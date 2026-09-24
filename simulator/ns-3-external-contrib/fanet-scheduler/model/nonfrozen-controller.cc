#include "nonfrozen-controller.h"

#include "bridge-records.h"

#include "ns3/simulator.h"

#include <algorithm>
#include <stdexcept>

namespace ns3::fanet
{
namespace
{
using namespace wire;

void
Require(bool ok, const char* message)
{
    if (!ok)
    {
        throw std::invalid_argument(message);
    }
}

Value
DeliveryRecord(const ControlDelivery& d)
{
    return Map{{"downlink", d.message.downlink}, {"node_id", d.message.node},
               {"epoch_id", d.message.epoch}, {"sequence", uint64_t{d.message.sequence}},
               {"sampled_at_ns", d.message.sampledAt}, {"started_at_ns", d.startedAt},
               {"ended_at_ns", d.endedAt}, {"received_at_ns", d.receivedAt},
               {"payload_bytes", d.payloadBytes}, {"wire_bytes", d.wireBytes}};
}

std::map<Id, Value>
NodePayloads(const Observation& sample, const Value& observation, const Value& physical)
{
    std::map<Id, Id> owners;
    for (const auto& q : sample.state.queues)
    {
        for (const auto& e : q.entries)
        {
            owners.emplace(e.packet, q.key.first);
        }
    }
    for (const auto& w : sample.state.waiting)
    {
        for (const auto& e : w.entries)
        {
            owners.emplace(e.packet, w.node);
        }
    }
    std::map<Id, Value> payloads;
    for (Id id : sample.config.nodes)
    {
        payloads.emplace(id, Map{{"packets", List{}}, {"queues", List{}},
            {"waiting_areas", List{}}, {"routes", List{}}, {"nodes", List{}},
            {"links", List{}}, {"gains", List{}},
            {"noise_w", physical.At("noise_w")}, {"channel_id", physical.At("channel_id")}});
    }
    // Explicit indices preserve canonical order without forwarding an unreceived global state.
    auto split = [&](const Value& source, const std::string& field, auto owner) {
        uint64_t index = 0;
        for (const auto& row : source.At(field).Array())
        {
            auto& value = std::get<Map>(payloads.at(owner(row)).data).at(field);
            std::get<List>(value.data).push_back(Map{{"index", index++}, {"value", row}});
        }
    };
    split(observation, "packets", [&](const Value& row) { return owners.at(row.At("packet_id").U()); });
    split(observation, "queues", [](const Value& row) { return row.At("key").Array()[0].U(); });
    for (const auto* field : {"waiting_areas", "routes"})
    {
        split(observation, field, [](const Value& row) { return row.At("node_id").U(); });
    }
    split(physical, "nodes", [](const Value& row) { return row.At("node_id").U(); });
    split(physical, "links", [](const Value& row) { return row.At("key").Array()[0].U(); });
    split(physical, "gains", [](const Value& row) { return row.At("sender").U(); });
    return payloads;
}

List
Merge(const std::map<Id, Value>& reports, const std::string& field)
{
    std::map<uint64_t, Value> rows;
    for (const auto& [id, report] : reports)
    {
        for (const auto& entry : report.At(field).Array())
        {
            entry.Object({"index", "value"});
            Require(rows.emplace(entry.At("index").U(), entry.At("value")).second,
                    "duplicate record across node reports");
        }
    }
    List result;
    for (const auto& [index, value] : rows)
    {
        Require(index == result.size(), "node reports omit a registered record");
        result.push_back(value);
    }
    return result;
}
} // namespace

NS_OBJECT_ENSURE_REGISTERED(NonfrozenController);

TypeId
NonfrozenController::GetTypeId()
{
    static TypeId tid = TypeId("ns3::fanet::NonfrozenController")
                            .SetParent<Object>().SetGroupName("FanetScheduler")
                            .AddConstructor<NonfrozenController>();
    return tid;
}

void
NonfrozenController::Configure(Ptr<ControlledEnvironment> env, Ptr<ScheduledRadio> radio,
                               const CollectionSettings& settings)
{
    Require(!m_disposed && !m_env && env && radio, "invalid controller ownership");
    const auto observation = env->Observe();
    Require(settings.maximumAge > 0 && settings.maximumAge < observation.config.period &&
                settings.reportPreparation >= 0 && settings.assembly >= 0 &&
                settings.distributionWindow >= 0 &&
                settings.distributionWindow <= observation.config.period,
            "invalid fresh-complete control timing configuration");
    auto ground = CreateObject<Node>();
    auto channel = CreateObject<ControlChannel>();
    std::map<Id, Ptr<Node>> nodes;
    for (Id id : observation.config.nodes)
    {
        Require(settings.channel.propagation.contains(id) &&
                    settings.channel.propagation.at(id) < observation.config.period,
                "control propagation must fit the declared complete-report envelope");
        nodes.emplace(id, env->GetNode(id));
    }
    channel->Configure(ground, nodes, settings.channel);
    m_env = env;
    m_radio = radio;
    m_ground = ground;
    m_channel = channel;
    m_settings = settings;
}

CollectedState
NonfrozenController::Collect(const Value& physical)
{
    Require(m_env && !m_current, "previous collected cycle is still active");
    const auto sample = m_env->Observe();
    Require(sample.remaining && sample.reference.epoch <= UINT32_MAX,
            "cannot collect an exhausted or unrepresentable epoch");
    const auto observation = WriteObservation(sample);
    const auto payloads = NodePayloads(sample, observation, physical);
    const auto sendAt = AddTime(sample.reference.sampledAt, m_settings.reportPreparation);
    Ns maximumArrival = sendAt;
    for (const auto& [id, payload] : payloads)
    {
        const auto size = EncodeValue(payload, 4 * 1024 * 1024).size() + 32;
        const auto duration = static_cast<Ns>(
            (uint64_t{size} * 1000000000 + m_settings.channel.rate - 1) / m_settings.channel.rate);
        maximumArrival = std::max(maximumArrival,
            AddTime(AddTime(sendAt, duration), m_settings.channel.propagation.at(id)));
    }
    const Ns expectedCut = AddTime(maximumArrival, m_settings.assembly);
    Require(expectedCut - sample.reference.sampledAt <= m_settings.maximumAge,
            "complete-report freshness envelope exceeded; no zero fill or silent stale reuse");
    m_env->BeginExternalCollection();
    std::map<Id, Value> received;
    List deliveries;
    for (const auto& [id, payload] : payloads)
    {
        m_channel->Send({false, id, sample.reference.epoch,
                         static_cast<uint32_t>(sample.reference.epoch), sample.reference.sampledAt,
                         payload}, sendAt,
            [&](const ControlDelivery& delivery) {
                Require(delivery.message.epoch == sample.reference.epoch &&
                            delivery.message.sampledAt == sample.reference.sampledAt &&
                            received.emplace(delivery.message.node, delivery.message.payload).second,
                        "wrong or repeated report during collection");
                deliveries.push_back(DeliveryRecord(delivery));
                m_arrivals.push_back(delivery);
                if (received.size() == sample.config.nodes.size())
                {
                    m_assemblyEvent = Simulator::Schedule(NanoSeconds(m_settings.assembly),
                                                          [] { Simulator::Stop(); });
                }
            });
    }
    Simulator::Run();
    Require(received.size() == sample.config.nodes.size() &&
                Simulator::Now().GetNanoSeconds() == expectedCut,
            "collection ended without a complete current report set");
    Map observed{{"reference", WriteReference(sample.reference)},
                 {"node_ids", observation.At("node_ids")},
                 {"packet_size_bytes", sample.config.packetBytes},
                 {"kind", "queue_state"}, {"pending_packet_ids", List{}},
                 {"periods_remaining", sample.remaining}};
    for (const auto* field : {"packets", "queues", "waiting_areas", "routes"})
    {
        observed.emplace(field, Merge(received, field));
    }
    Map measured{{"reference", WriteReference(sample.reference)},
                 {"noise_w", received.begin()->second.At("noise_w")},
                 {"channel_id", received.begin()->second.At("channel_id")}};
    for (const auto* field : {"nodes", "links", "gains"})
    {
        measured.emplace(field, Merge(received, field));
    }
    // This is an audit of decoded values, not a fallback to the original snapshot.
    Require(EncodeValue(observed, 128 * 1024 * 1024) == EncodeValue(observation, 128 * 1024 * 1024) &&
                EncodeValue(measured, 128 * 1024 * 1024) ==
                    EncodeValue(WritePhysical(physical, sample.reference), 128 * 1024 * 1024),
            "received reports disagree with their causal samples");
    m_current = CollectedState{sample, observed, measured,
        Map{{"profile", "fresh-complete-control-v1"},
            {"sampled_at_ns", sample.reference.sampledAt}, {"cut_at_ns", expectedCut},
            {"maximum_age_ns", m_settings.maximumAge}, {"reports", deliveries}}, expectedCut};
    return *m_current;
}

ControlExecution
NonfrozenController::Execute(const Action& action, Ns encodeTime, Ns decodeTime)
{
    Require(m_current && action.reference == m_current->sample.reference &&
                encodeTime >= 0 && decodeTime >= 0,
            "missing collection or invalid control computation timing");
    const auto deadline = AddTime(action.reference.sampledAt, m_current->sample.config.period);
    const auto created = AddTime(AddTime(m_current->cut, encodeTime), decodeTime);
    const auto executeAt = AddTime(created, m_settings.distributionWindow);
    m_env->AcceptExternalPlan(action);
    std::optional<ControlExecution> result;
    m_channel->ScheduleCommandWindow(action, created, executeAt, deadline,
        [&] { return m_radio->ExecutionLinksAt(Simulator::Now().GetNanoSeconds()); },
        [&](const ControlExecution& execution) {
            result = execution;
            m_radio->StartPreparedCycle(execution.execution);
        },
        [this](const ControlDelivery& delivery) { m_arrivals.push_back(delivery); });
    Simulator::Run();
    Require(result.has_value() && Simulator::Now().GetNanoSeconds() == deadline && !m_env->IsBusy(),
            "nonfrozen cycle did not settle at its original boundary");
    m_current.reset();
    return *result;
}

Value
NonfrozenController::TakeArrivals()
{
    List records;
    for (const auto& delivery : m_arrivals)
    {
        records.push_back(DeliveryRecord(delivery));
    }
    m_arrivals.clear();
    return records;
}

void
NonfrozenController::DoDispose()
{
    m_disposed = true;
    Simulator::Cancel(m_assemblyEvent);
    if (m_channel)
    {
        m_channel->Dispose();
    }
    if (m_ground)
    {
        m_ground->Dispose();
    }
    m_current.reset();
    m_arrivals.clear();
    m_channel = nullptr;
    m_ground = nullptr;
    m_radio = nullptr;
    m_env = nullptr;
    Object::DoDispose();
}
} // namespace ns3::fanet
