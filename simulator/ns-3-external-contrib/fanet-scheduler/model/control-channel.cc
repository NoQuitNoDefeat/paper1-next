#include "control-channel.h"

#include "ns3/header.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace ns3::fanet
{
namespace
{
void
Require(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::invalid_argument(message);
    }
}

class ControlHeader : public Header
{
  public:
    ControlMessage message;

    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::fanet::ControlHeader")
                                .SetParent<Header>().AddConstructor<ControlHeader>();
        return tid;
    }
    TypeId GetInstanceTypeId() const override
    {
        return GetTypeId();
    }
    uint32_t GetSerializedSize() const override
    {
        return 32;
    }
    void Serialize(Buffer::Iterator i) const override
    {
        i.WriteHtonU16(1);
        i.WriteHtonU16(message.downlink ? 2 : 1);
        i.WriteHtonU64(message.epoch);
        i.WriteHtonU64(message.node);
        i.WriteHtonU32(message.sequence);
        i.WriteHtonU64(static_cast<uint64_t>(message.sampledAt));
    }
    uint32_t Deserialize(Buffer::Iterator i) override
    {
        Require(i.ReadNtohU16() == 1, "unknown control header version");
        const auto kind = i.ReadNtohU16();
        Require(kind == 1 || kind == 2, "unknown control frame kind");
        message.downlink = kind == 2;
        message.epoch = i.ReadNtohU64();
        message.node = i.ReadNtohU64();
        message.sequence = i.ReadNtohU32();
        const auto sample = i.ReadNtohU64();
        Require(sample <= INT64_MAX, "control sample time overflow");
        message.sampledAt = static_cast<Ns>(sample);
        return 32;
    }
    void Print(std::ostream& out) const override
    {
        out << message.epoch << ':' << message.node << ':' << message.sequence;
    }
};
} // namespace

NS_OBJECT_ENSURE_REGISTERED(ControlChannel);

TypeId
ControlChannel::GetTypeId()
{
    static TypeId tid = TypeId("ns3::fanet::ControlChannel")
                            .SetParent<Object>().SetGroupName("FanetScheduler")
                            .AddConstructor<ControlChannel>();
    return tid;
}

void
ControlChannel::Configure(Ptr<Node> ground, const std::map<Id, Ptr<Node>>& nodes,
                          const ControlSettings& settings)
{
    Require(!m_disposed && !m_ground && ground && !nodes.empty() && settings.rate > 0 &&
                settings.rate <= 1000000000 && settings.propagation.size() == nodes.size(),
            "invalid control channel configuration");
    for (const auto& [id, node] : nodes)
    {
        Require(node && node != ground && settings.propagation.contains(id) &&
                    settings.propagation.at(id) >= 0,
                "control requires distinct ground station and finite registered pipes");
    }
    m_ground = ground;
    m_nodes = nodes;
    m_settings = settings;
}

Ns
ControlChannel::Send(const ControlMessage& message, Ns at,
                     const std::function<void(const ControlDelivery&)>& receiver)
{
    Require(m_ground && m_nodes.contains(message.node) && receiver &&
                at >= Simulator::Now().GetNanoSeconds() && message.sampledAt >= 0 &&
                message.sampledAt <= at,
            "invalid control frame destination, callback or causal time");
    const auto pipe = std::make_pair(message.downlink, message.node);
    const auto sequence = m_sequences.find(pipe);
    Require(sequence == m_sequences.end() || message.sequence > sequence->second,
            "control sequence must increase without wraparound");
    // Bound the serialized frame before multiplying bytes by integer nanoseconds.
    const auto encoded = wire::EncodeValue(message.payload, 4 * 1024 * 1024);
    auto packet = Create<ns3::Packet>(encoded.data(), static_cast<uint32_t>(encoded.size()));
    ControlHeader header;
    header.message = message;
    packet->AddHeader(header);
    const auto start = std::max(at, m_freeAt.contains(pipe) ? m_freeAt.at(pipe) : at);
    const auto duration = static_cast<Ns>(
        (uint64_t{packet->GetSize()} * 1000000000 + m_settings.rate - 1) / m_settings.rate);
    const auto end = AddTime(start, duration);
    const auto received = AddTime(end, m_settings.propagation.at(message.node));
    m_sequences[pipe] = message.sequence;
    m_freeAt[pipe] = end;
    std::erase_if(m_events, [](const auto& event) { return event.IsExpired(); });
    ++m_outstanding;
    auto delivery = std::make_shared<ControlDelivery>();
    auto phase = std::make_shared<unsigned>(0);
    // Register all three events in causal order, including zero propagation and exact equality.
    m_events.push_back(Simulator::Schedule(NanoSeconds(start) - Simulator::Now(),
        [delivery, phase] {
            Require(*phase == 0, "duplicate control transmission start");
            delivery->startedAt = Simulator::Now().GetNanoSeconds();
            *phase = 1;
        }));
    m_events.push_back(Simulator::Schedule(NanoSeconds(end) - Simulator::Now(),
        [delivery, phase] {
            Require(*phase == 1, "control transmission ended before starting");
            delivery->endedAt = Simulator::Now().GetNanoSeconds();
            *phase = 2;
        }));
    m_events.push_back(Simulator::Schedule(NanoSeconds(received) - Simulator::Now(),
        [this, packet, delivery, phase, receiver] {
            Require(*phase == 2, "control reception before complete transmission");
            delivery->wireBytes = packet->GetSize();
            ControlHeader decoded;
            Require(packet->RemoveHeader(decoded) == 32, "incomplete control header");
            std::vector<uint8_t> bytes(packet->GetSize());
            packet->CopyData(bytes.data(), bytes.size());
            delivery->message = decoded.message;
            delivery->message.payload = wire::DecodeValue(bytes);
            delivery->payloadBytes = bytes.size();
            delivery->receivedAt = Simulator::Now().GetNanoSeconds();
            *phase = 3;
            --m_outstanding;
            receiver(*delivery);
        }));
    return received;
}

void
ControlChannel::ScheduleCommandWindow(
    const Action& plan, Ns createdAt, Ns executeAt, Ns deadline,
    const std::function<std::set<Link>()>& physicalLinks,
    const std::function<void(const ControlExecution&)>& execute,
    const std::function<void(const ControlDelivery&)>& receipt)
{
    Require(m_ground && createdAt >= Simulator::Now().GetNanoSeconds() &&
                plan.reference.sampledAt >= 0 && !m_planEpochs.contains(plan.reference.epoch) &&
                createdAt >= plan.reference.sampledAt && executeAt >= createdAt &&
                deadline >= Simulator::Now().GetNanoSeconds() &&
                deadline > plan.reference.sampledAt && plan.reference.epoch <= UINT32_MAX &&
                physicalLinks && execute,
            "invalid common command window");
    std::set<Id> endpoints;
    wire::List links;
    for (const auto& [sender, receiver] : plan.links)
    {
        Require(m_nodes.contains(sender) && m_nodes.contains(receiver) &&
                    endpoints.insert(sender).second && endpoints.insert(receiver).second,
                "command plan violates single-radio endpoint resources");
        links.push_back(wire::List{sender, receiver});
    }
    // A plan formed after expiry is logged as a global miss, never transmitted later.
    const auto destinations = createdAt <= deadline ? endpoints : std::set<Id>{};
    for (Id id : destinations)
    {
        const auto previous = m_sequences.find({true, id});
        Require(previous == m_sequences.end() || plan.reference.epoch > previous->second,
                "command epoch was already sent on an endpoint pipe");
    }
    auto accepted = std::make_shared<std::map<Id, Ns>>();
    const wire::Value payload = wire::Map{{"links", links}, {"created_at_ns", createdAt},
                                          {"execute_at_ns", executeAt}, {"deadline_ns", deadline}};
    // Validate every pipe before any send, so a later overflow cannot leave a partial command.
    const auto size = wire::EncodeValue(payload, 4 * 1024 * 1024).size() + 32;
    const auto duration = static_cast<Ns>(
        (uint64_t{size} * 1000000000 + m_settings.rate - 1) / m_settings.rate);
    for (Id id : destinations)
    {
        const auto pipe = std::make_pair(true, id);
        const auto start = std::max(createdAt, m_freeAt.contains(pipe) ? m_freeAt.at(pipe) : createdAt);
        AddTime(AddTime(start, duration), m_settings.propagation.at(id));
    }
    m_planEpochs.insert(plan.reference.epoch);
    for (Id id : destinations)
    {
        Send({true, id, plan.reference.epoch, static_cast<uint32_t>(plan.reference.epoch),
              plan.reference.sampledAt, payload}, createdAt,
             [accepted, executeAt, receipt](const ControlDelivery& delivered) {
                 if (delivered.receivedAt <= executeAt)
                 {
                     accepted->emplace(delivered.message.node, delivered.receivedAt);
                 }
                 if (receipt)
                 {
                     receipt(delivered);
                 }
             });
    }
    // Late control receipts retain their original epoch, never delayed DATA authorizations.
    const auto at = std::min(executeAt, deadline);
    m_events.push_back(Simulator::Schedule(NanoSeconds(at) - Simulator::Now(),
        [plan, executeAt, deadline, accepted, physicalLinks, execute] {
            ControlExecution result{{plan.reference, {}}, executeAt,
                                     Simulator::Now().GetNanoSeconds(), {}, *accepted};
            const bool globalMiss = executeAt > deadline;
            const auto available = globalMiss ? std::set<Link>{} : physicalLinks();
            for (const auto& link : plan.links)
            {
                const auto reason = globalMiss ? "global_miss"
                    : (!accepted->contains(link.first) || !accepted->contains(link.second))
                        ? "command_miss"
                    : !available.contains(link) ? "physical_invalid" : "retained";
                result.reasons.emplace(link, reason);
                if (std::string(reason) == "retained")
                {
                    result.execution.links.push_back(link);
                }
            }
            execute(result);
        }));
}

uint64_t
ControlChannel::Outstanding() const
{
    return m_outstanding;
}

void
ControlChannel::DoDispose()
{
    m_disposed = true;
    for (const auto& event : m_events)
    {
        Simulator::Cancel(event);
    }
    m_events.clear();
    m_outstanding = 0;
    m_nodes.clear();
    m_ground = nullptr;
    Object::DoDispose();
}
} // namespace ns3::fanet
