#include "bridge-records.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace ns3::fanet::wire
{
namespace
{
void
Require(bool ok, const char* message)
{
    if (!ok)
    {
        throw std::invalid_argument(message);
    }
}

Link
ReadLink(const Value& v)
{
    Require(v.Array().size() == 2, "link requires two IDs");
    return {v.Array()[0].U(), v.Array()[1].U()};
}

Value
WriteLink(Link key)
{
    return List{key.first, key.second};
}

Vector
ReadVector(const Value& v)
{
    Require(v.Array().size() == 3, "motion vector requires three components");
    return {v.Array()[0].D(), v.Array()[1].D(), v.Array()[2].D()};
}

template <typename T>
Value
Opt(const std::optional<T>& value)
{
    return value ? Value(*value) : Value();
}

template <typename T, typename F>
List
Many(const std::vector<T>& values, F function)
{
    List result;
    for (const auto& v : values)
    {
        result.push_back(function(v));
    }
    return result;
}

List
Ids(const std::vector<Id>& values)
{
    return Many(values, [](auto v) { return Value(v); });
}

std::vector<Packet>
ReadPackets(const Value& v)
{
    std::vector<Packet> result;
    for (const auto& p : v.Array())
    {
        p.Object({"packet_id", "source", "destination", "born_at_ns", "deadline_ns"});
        Require(p.At("deadline_ns").Null(), "packet deadlines disabled in this scope");
        result.push_back({p.At("packet_id").U(),
                          p.At("source").U(),
                          p.At("destination").U(),
                          p.At("born_at_ns").I(),
                          std::nullopt});
    }
    return result;
}

std::vector<Entry>
ReadEntries(const Value& v, bool waiting)
{
    std::vector<Entry> result;
    const char* field = waiting ? "waiting_since_ns" : "enqueued_at_ns";
    for (const auto& e : v.Array())
    {
        e.Object({"packet_id", "node_arrived_at_ns", field});
        result.push_back({e.At("packet_id").U(), e.At("node_arrived_at_ns").I(), e.At(field).I()});
    }
    return result;
}

std::vector<Route>
ReadRoutes(const Value& v)
{
    std::vector<Route> result;
    for (const auto& r : v.Array())
    {
        r.Object({"node_id", "destination", "next_hop"});
        result.push_back(
            {r.At("node_id").U(),
             r.At("destination").U(),
             r.At("next_hop").Null() ? std::nullopt : std::optional<Id>(r.At("next_hop").U())});
    }
    return result;
}

List
Packets(const std::vector<Packet>& values)
{
    return Many(values, [](const auto& p) {
        return Map{{"packet_id", p.id},
                   {"source", p.source},
                   {"destination", p.destination},
                   {"born_at_ns", p.born},
                   {"deadline_ns", Opt(p.deadline)}};
    });
}

Map
Containers(const State& state, const Config& config, Ns at)
{
    List queues, waiting;
    for (std::size_t i = 0; i < state.queues.size(); ++i)
    {
        const auto& q = state.queues[i];
        auto entries = Many(q.entries, [](const auto& e) {
            return Map{{"packet_id", e.packet},
                       {"node_arrived_at_ns", e.arrived},
                       {"enqueued_at_ns", e.entered}};
        });
        queues.push_back(Map{{"key", WriteLink(q.key)},
                             {"entries", entries},
                             {"capacity_bytes", config.queues.at(i).capacity},
                             {"occupancy_bytes", Occupancy(q.entries.size(), config.packetBytes)},
                             {"hol_ns", q.entries.empty() ? Ns{0} : at - q.entries[0].entered}});
    }
    for (std::size_t i = 0; i < state.waiting.size(); ++i)
    {
        const auto& w = state.waiting[i];
        auto entries = Many(w.entries, [](const auto& e) {
            return Map{{"packet_id", e.packet},
                       {"node_arrived_at_ns", e.arrived},
                       {"waiting_since_ns", e.entered}};
        });
        waiting.push_back(
            Map{{"node_id", w.node},
                {"entries", entries},
                {"capacity_bytes", config.waiting.at(i).capacity},
                {"max_wait_ns", config.waiting.at(i).maxWait},
                {"occupancy_bytes", Occupancy(w.entries.size(), config.packetBytes)}});
    }
    return {{"queues", queues}, {"waiting_areas", waiting}, {"packets", Packets(state.packets)}};
}

void
ValidatePhysical(const Value& p, const Config& config, Ns at)
{
    p.Object({"at_ns", "nodes", "links", "gains", "noise_w", "channel_id"});
    Require(p.At("at_ns").I() == at && p.At("noise_w").D() > 0, "physical time/noise");
    const auto channel = p.At("channel_id").U();
    std::set<Id> nodes;
    for (const auto& n : p.At("nodes").Array())
    {
        n.Object({"node_id",
                  "position_m",
                  "velocity_m_s",
                  "transmit_power_w",
                  "radio_count",
                  "duplex_mode",
                  "channels"});
        Require(nodes.insert(n.At("node_id").U()).second, "duplicate physical node");
        for (const auto* field : {"position_m", "velocity_m_s"})
        {
            Require(n.At(field).Array().size() == 3, "coordinate length");
            for (const auto& coordinate : n.At(field).Array())
            {
                coordinate.D();
            }
        }
        Require(n.At("transmit_power_w").D() > 0 && n.At("radio_count").U() == 1 &&
                    n.At("duplex_mode").S() == "HD" && n.At("channels").Array().size() == 1 &&
                    n.At("channels").Array()[0].U() == channel,
                "physical resources");
    }
    Require(nodes == std::set<Id>(config.nodes.begin(), config.nodes.end()), "physical nodes");
    std::set<Link> edges, gains;
    for (const auto& e : p.At("links").Array())
    {
        e.Object({"key",
                  "base_quality_db",
                  "rate_bytes_per_second",
                  "sinr_threshold",
                  "success_probability"});
        auto key = ReadLink(e.At("key"));
        Require(key.first != key.second && nodes.contains(key.first) &&
                    nodes.contains(key.second) && edges.insert(key).second,
                "physical edge identity");
        e.At("base_quality_db").D();
        Require(e.At("rate_bytes_per_second").U() > 0 && e.At("sinr_threshold").D() > 0 &&
                    e.At("success_probability").D() >= 0 && e.At("success_probability").D() <= 1,
                "physical link range");
    }
    for (const auto& g : p.At("gains").Array())
    {
        g.Object({"sender", "receiver", "gain"});
        Link key{g.At("sender").U(), g.At("receiver").U()};
        Require(nodes.contains(key.first) && nodes.contains(key.second) &&
                    gains.insert(key).second && g.At("gain").D() >= 0,
                "gain identity/range");
    }
    for (auto [tx, unused] : edges)
    {
        for (auto [other, rx] : edges)
        {
            Require(gains.contains({tx, rx}), "missing complete interference path");
        }
    }
}
} // namespace

Initialization
ReadInitialization(const Value& v)
{
    Require(std::holds_alternative<Map>(v.data), "INIT requires an object");
    const auto& fields = std::get<Map>(v.data);
    const bool radio =
        fields.contains("wireless");
    const bool profile = fields.contains("execution_profile");
    const bool control = fields.contains("control");
    Require(fields.size() == 5 + static_cast<unsigned>(radio) + static_cast<unsigned>(profile) +
                                 static_cast<unsigned>(control),
            "unknown INIT fields");
    for (const auto* key : {"config", "initial", "trajectory", "physical_inputs", "metadata"})
    {
        v.At(key);
    }
    Initialization result;
    if (profile)
    {
        result.executionProfile = v.At("execution_profile").S();
        Require(result.executionProfile == "controlled-budget-v1" ||
                    result.executionProfile == "full-sinr-v1",
                "unknown execution profile");
    }
    const auto& c = v.At("config");
    const bool semantics = std::get<Map>(c.data).contains("stale_queue_policy");
    if (semantics)
    {
        c.Object({"node_ids",
                  "queues",
                  "waiting_areas",
                  "packet_size_bytes",
                  "start_ns",
                  "end_ns",
                  "period_ns",
                  "packet_deadlines_enabled",
                  "retransmissions_enabled",
                  "stale_queue_policy",
                  "waiting_restore"});
    }
    else
    {
        c.Object({"node_ids",
                  "queues",
                  "waiting_areas",
                  "packet_size_bytes",
                  "start_ns",
                  "end_ns",
                  "period_ns",
                  "packet_deadlines_enabled",
                  "retransmissions_enabled"});
    }
    auto& config = result.config;
    if (semantics)
    {
        const auto stale = c.At("stale_queue_policy").S();
        const auto restore = c.At("waiting_restore").S();
        Require((stale == "keep" || stale == "rehome") && (restore == "drop" || restore == "keep"),
                "unknown queue semantics");
        config.rehomeStaleQueues = stale == "rehome";
        config.keepRestoredWaiting = restore == "keep";
    }
    config.packetBytes = c.At("packet_size_bytes").U();
    config.start = c.At("start_ns").I();
    config.end = c.At("end_ns").I();
    config.period = c.At("period_ns").I();
    config.packetDeadlines = c.At("packet_deadlines_enabled").B();
    config.retransmissions = c.At("retransmissions_enabled").B();
    for (const auto& n : c.At("node_ids").Array())
    {
        config.nodes.push_back(n.U());
    }
    for (const auto& q : c.At("queues").Array())
    {
        q.Object({"key", "capacity_bytes"});
        config.queues.push_back({ReadLink(q.At("key")), q.At("capacity_bytes").U()});
    }
    for (const auto& w : c.At("waiting_areas").Array())
    {
        w.Object({"node_id", "capacity_bytes", "max_wait_ns"});
        config.waiting.push_back(
            {w.At("node_id").U(), w.At("capacity_bytes").U(), w.At("max_wait_ns").I()});
    }
    const auto& state = v.At("initial");
    state.Object({"packets", "queues", "waiting_areas", "routes"});
    result.state.packets = ReadPackets(state.At("packets"));
    result.state.routes = ReadRoutes(state.At("routes"));
    for (const auto& q : state.At("queues").Array())
    {
        q.Object({"key", "entries"});
        result.state.queues.push_back({ReadLink(q.At("key")), ReadEntries(q.At("entries"), false)});
    }
    for (const auto& w : state.At("waiting_areas").Array())
    {
        w.Object({"node_id", "entries"});
        result.state.waiting.push_back({w.At("node_id").U(), ReadEntries(w.At("entries"), true)});
    }
    v.At("trajectory").Object({"frames"});
    for (const auto& f : v.At("trajectory").At("frames").Array())
    {
        f.Object({"start_ns", "services", "births", "route_updates"});
        Frame frame;
        frame.start = f.At("start_ns").I();
        frame.births = ReadPackets(f.At("births"));
        frame.routeUpdates = ReadRoutes(f.At("route_updates"));
        for (const auto& s : f.At("services").Array())
        {
            s.Object(
                {"key", "budget_bytes", "completed_at_ns", "available", "rate_bytes_per_second"});
            frame.services.push_back(
                {ReadLink(s.At("key")),
                 s.At("budget_bytes").U(),
                 s.At("completed_at_ns").I(),
                 s.At("available").B(),
                 s.At("rate_bytes_per_second").Null()
                     ? std::nullopt
                     : std::optional<Bytes>(s.At("rate_bytes_per_second").U())});
        }
        result.frames.push_back(std::move(frame));
    }
    ValidateInitial(config, result.state);
    ValidateTrajectory(config, result.state, result.frames);
    result.physical = v.At("physical_inputs").Array();
    Require(result.physical.size() == result.frames.size() + 1, "missing final physical input");
    for (std::size_t i = 0; i < result.physical.size(); ++i)
    {
        ValidatePhysical(result.physical[i], config, config.start + i * config.period);
        Require(result.physical[i].At("channel_id").U() == result.physical[0].At("channel_id").U(),
                "channel changes unsupported");
    }
    result.metadata = v.At("metadata");
    if (radio)
    {
        auto settings = v.At("wireless");
        const bool actualAck = settings.At("profile").S() == "transaction-ack-v1";
        if (std::get<Map>(settings.data).contains("motion"))
        {
            const auto& moving = settings.At("motion");
            moving.Object({"profile", "low_m", "high_m", "reference_distance_m",
                           "path_loss_exponent", "sinr_threshold"});
            Require(actualAck && moving.At("profile").S() ==
                                     "continuous-motion-frame-quasistatic-v1",
                    "unknown or incompatible moving channel profile");
            result.radioMotion = MotionSettings{ReadVector(moving.At("low_m")),
                                                 ReadVector(moving.At("high_m")),
                                                 moving.At("reference_distance_m").D(),
                                                 moving.At("path_loss_exponent").D(),
                                                 moving.At("sinr_threshold").D(), {}};
            std::get<Map>(settings.data).erase("motion");
        }
        Require(actualAck || settings.At("profile").S() == "ideal-spectrum-v1",
                "unknown radio profile");
        if (actualAck)
        {
            settings.Object({"profile", "center_hz", "bandwidth_hz", "propagation_ns",
                             "overhead_bytes", "frames", "ack_bytes",
                             "ack_rate_bytes_per_second", "turnaround_ns"});
        }
        else
        {
            settings.Object({"profile", "center_hz", "bandwidth_hz", "propagation_ns",
                             "overhead_bytes", "frames"});
        }
        result.radioSettings = RadioSettings{settings.At("center_hz").D(),
                                             settings.At("bandwidth_hz").D(),
                                             settings.At("propagation_ns").I(),
                                             settings.At("overhead_bytes").U()};
        if (actualAck)
        {
            result.radioSettings->actualAck = true;
            result.radioSettings->ackBytes = settings.At("ack_bytes").U();
            result.radioSettings->ackRate = settings.At("ack_rate_bytes_per_second").U();
            result.radioSettings->turnaroundNs = settings.At("turnaround_ns").I();
        }
        Require(settings.At("frames").Array().size() == result.physical.size(),
                "incomplete private radio frames");
        for (std::size_t i = 0; i < result.physical.size(); ++i)
        {
            const auto& source = settings.At("frames").Array()[i];
            ValidatePhysical(source, config, config.start + i * config.period);
            RadioFrame frame;
            frame.at = source.At("at_ns").I();
            frame.noiseW = source.At("noise_w").D();
            for (const auto& n : source.At("nodes").Array())
            {
                frame.powersW.emplace(n.At("node_id").U(), n.At("transmit_power_w").D());
            }
            if (result.radioMotion)
            {
                std::map<Id, MotionState> states;
                for (const auto& node : source.At("nodes").Array())
                {
                    states.emplace(node.At("node_id").U(),
                                   MotionState{ReadVector(node.At("position_m")),
                                               ReadVector(node.At("velocity_m_s"))});
                }
                result.radioMotion->boundaries.push_back(std::move(states));
            }
            for (const auto& gain : source.At("gains").Array())
            {
                if (gain.At("sender").U() != gain.At("receiver").U())
                {
                    frame.gains.emplace(Link{gain.At("sender").U(), gain.At("receiver").U()},
                                        gain.At("gain").D());
                }
            }
            for (const auto& link : source.At("links").Array())
            {
                Require(!result.radioMotion || link.At("sinr_threshold").D() ==
                                                   result.radioMotion->sinrThreshold,
                        "moving channel requires registered single-MCS threshold");
                frame.rates.emplace(ReadLink(link.At("key")), link.At("rate_bytes_per_second").U());
            }
            const auto& observed = result.physical[i];
            Require(source.At("links").Array().size() == observed.At("links").Array().size(),
                    "radio physical edge set must remain frozen");
            for (const auto& link : observed.At("links").Array())
            {
                const auto key = ReadLink(link.At("key"));
                Require(frame.rates.contains(key) &&
                            frame.rates.at(key) == link.At("rate_bytes_per_second").U(),
                        "radio changed nominal PHY rate");
                for (const auto& actual : source.At("links").Array())
                {
                    if (ReadLink(actual.At("key")) == key)
                    {
                        Require(actual.At("sinr_threshold").D() == link.At("sinr_threshold").D(),
                                "radio changed nominal SINR threshold");
                    }
                }
            }
            for (const auto& node : observed.At("nodes").Array())
            {
                Require(frame.powersW.at(node.At("node_id").U()) == node.At("transmit_power_w").D(),
                        "radio changed fixed tx power");
            }
            result.radioFrames.push_back(std::move(frame));
        }
    }
    result.metadata.Object(
        {"feature_schema_sha256", "history_sha256", "reward_sha256", "checkpoint_sha256"});
    for (const auto& [key, token] : std::get<Map>(result.metadata.data))
    {
        Require(token.Null() ||
                    (token.S().size() == 64 &&
                     token.S().find_first_not_of("0123456789abcdef") == std::string::npos),
                "metadata requires null or SHA256 token");
    }
    if (control)
    {
        Require(result.radioMotion.has_value(), "nonfrozen control requires moving actual-ACK radio");
        const auto& settings = v.At("control");
        settings.Object({"profile", "rate_bytes_per_second", "propagation_ns", "maximum_age_ns",
                         "report_preparation_ns", "assembly_ns", "distribution_window_ns"});
        Require(settings.At("profile").S() == "fresh-complete-control-v1",
                "unknown nonfrozen control profile");
        CollectionSettings decoded;
        decoded.channel.rate = settings.At("rate_bytes_per_second").U();
        for (const auto& delay : settings.At("propagation_ns").Array())
        {
            delay.Object({"node_id", "delay_ns"});
            Require(decoded.channel.propagation.emplace(delay.At("node_id").U(),
                                                        delay.At("delay_ns").I()).second,
                    "duplicate control pipe delay");
        }
        decoded.maximumAge = settings.At("maximum_age_ns").I();
        decoded.reportPreparation = settings.At("report_preparation_ns").I();
        decoded.assembly = settings.At("assembly_ns").I();
        decoded.distributionWindow = settings.At("distribution_window_ns").I();
        result.control = decoded;
        for (std::size_t i = 0; i < result.physical.size(); ++i)
        {
            for (const auto& node : result.physical[i].At("nodes").Array())
            {
                const auto& actual = result.radioMotion->boundaries[i].at(node.At("node_id").U());
                Require(ReadVector(node.At("position_m")) == actual.position &&
                            ReadVector(node.At("velocity_m_s")) == actual.velocity,
                        "complete-report controller requires actual boundary kinematics");
            }
        }
    }
    return result;
}

std::set<Link>
FailedSinrLinks(const Value& physical, const Action& action)
{
    std::map<Id, double> powers;
    std::map<Link, double> gains, thresholds;
    for (const auto& node : physical.At("nodes").Array())
    {
        powers.emplace(node.At("node_id").U(), node.At("transmit_power_w").D());
    }
    for (const auto& gain : physical.At("gains").Array())
    {
        gains.emplace(Link{gain.At("sender").U(), gain.At("receiver").U()}, gain.At("gain").D());
    }
    for (const auto& link : physical.At("links").Array())
    {
        thresholds.emplace(ReadLink(link.At("key")), link.At("sinr_threshold").D());
    }
    std::set<Link> failed;
    for (const auto& target : action.links)
    {
        double interference = 0;
        // Match the declared left-to-right binary64 action order, without Python outcomes.
        for (const auto& source : action.links)
        {
            if (source != target)
            {
                // Force the product's binary64 rounding before addition, including on FMA CPUs.
                volatile double contribution =
                    powers.at(source.first) * gains.at({source.first, target.second});
                interference += contribution;
            }
        }
        const double denominator = physical.At("noise_w").D() + interference;
        const double sinr = powers.at(target.first) * gains.at(target) / denominator;
        Require(std::isfinite(denominator) && denominator > 0 && std::isfinite(sinr),
                "nonfinite execution SINR");
        if (sinr < thresholds.at(target))
        {
            failed.insert(target);
        }
    }
    return failed;
}

uint64_t
RequiredResultCapacity(const Initialization& inputs, uint32_t capacity)
{
    // Cap additions above the largest relevant array bound before summing uint64 capacities.
    // Published active packets occupy exactly one queue/wait slot; only this cycle's births
    // enlarge its risk population. Relays do not create a second logical packet identity.
    constexpr uint64_t beyondLimit = 100001;
    auto add = [beyondLimit](uint64_t a, uint64_t b) {
        return std::min(beyondLimit, a + std::min(beyondLimit, b));
    };
    uint64_t lifetime = std::min(beyondLimit, uint64_t{inputs.state.packets.size()});
    uint64_t slots = 0;
    uint64_t peakBirths = 0;
    std::set<Link> routeKeys;
    for (const auto& queue : inputs.config.queues)
    {
        slots = add(slots, queue.capacity / inputs.config.packetBytes);
    }
    for (const auto& wait : inputs.config.waiting)
    {
        slots = add(slots, wait.capacity / inputs.config.packetBytes);
    }
    for (const auto& route : inputs.state.routes)
    {
        routeKeys.insert({route.node, route.destination});
    }
    for (const auto& frame : inputs.frames)
    {
        lifetime = add(lifetime, frame.births.size());
        peakBirths = std::max(peakBirths, uint64_t{frame.births.size()});
        for (const auto& route : frame.routeUpdates)
        {
            routeKeys.insert({route.node, route.destination});
        }
    }
    const auto packets = std::min(lifetime, add(slots, peakBirths));
    const auto routes = routeKeys.size();
    Require(packets <= 50000 && routes <= 100000, "result array count exceeds wire limit");
    const bool ack = inputs.radioSettings && inputs.radioSettings->actualAck;
    Require(!inputs.radioSettings || packets <= (ack ? 12500 : 20000),
            "radio event array bound exceeded");
    uint64_t physicalBytes = 0;
    for (const auto& p : inputs.physical)
    {
        physicalBytes = std::max(physicalBytes, uint64_t{EncodeValue(p, capacity).size()});
    }
    const auto maximumLinks = inputs.config.nodes.size() / 2;
    const auto radioBytes = inputs.radioSettings
        ? (ack ? 3200 : 2000) * packets + 512 * inputs.config.queues.size() +
              4096 + 256 * maximumLinks * maximumLinks
        : 0;
    const auto motionBytes = inputs.radioMotion ? 1000 * packets : 0;
    const auto controlBytes = inputs.control ? 4096 + 4096 * inputs.config.nodes.size() +
        1024 * inputs.config.queues.size() : 0;
    return 16384 + radioBytes + motionBytes + controlBytes + physicalBytes + 4500 * packets +
           1500 * inputs.config.queues.size() + 1000 * inputs.config.waiting.size() + 150 * routes +
           20 * inputs.config.nodes.size();
}

Value
WriteReference(const Reference& ref)
{
    return Map{{"episode_id", ref.run},
               {"period_index", ref.epoch},
               {"sampled_at_ns", ref.sampledAt},
               {"stage", "decision"}};
}

Value
WriteExecutionChannel(const ExecutionChannel& sample)
{
    List links, powers, gains;
    for (const auto& link : sample.links)
    {
        links.push_back(List{link.first, link.second});
    }
    for (const auto& [id, power] : sample.powersW)
    {
        powers.push_back(Map{{"node_id", id}, {"power_w", power}});
    }
    for (const auto& [link, gain] : sample.gains)
    {
        gains.push_back(Map{{"sender", link.first}, {"receiver", link.second}, {"gain", gain}});
    }
    return Map{{"sampled_at_ns", sample.at}, {"executed_links", links},
               {"noise_w", sample.noiseW}, {"powers", powers}, {"gains", gains}};
}

Value
WriteRadioEvents(const std::vector<RadioEvent>& events)
{
    return Many(events, [](const auto& e) {
        return Map{{"kind", e.kind},
                   {"packet_id", e.packet},
                   {"link", WriteLink(e.link)},
                   {"at_ns", e.at},
                   {"payload_bytes", e.payloadBytes},
                   {"frame_bytes", e.frameBytes}};
    });
}

Value
WriteReceptionSamples(const std::vector<ReceptionSample>& samples)
{
    return Many(samples, [](const auto& s) {
        return Map{{"packet_id", s.packet}, {"link", WriteLink(s.link)},
                   {"sampled_at_ns", s.at}, {"gain", s.gain},
                   {"received_power_w", s.receivedPowerW},
                   {"sender_position_m", List{s.senderPosition.x, s.senderPosition.y,
                                              s.senderPosition.z}},
                   {"receiver_position_m", List{s.receiverPosition.x, s.receiverPosition.y,
                                                s.receiverPosition.z}}};
    });
}

Action
ReadPlan(const Value& v, const Reference& reference, Ns period, bool nonfrozen)
{
    v.Object({"action", "timing"});
    const auto& action = v.At("action");
    action.Object({"reference", "links"});
    const auto& r = action.At("reference");
    r.Object({"episode_id", "period_index", "sampled_at_ns", "stage"});
    Reference ref{r.At("episode_id").S(), r.At("period_index").U(), r.At("sampled_at_ns").I()};
    Require(ref == reference && r.At("stage").S() == "decision", "stale plan reference");
    const auto& timing = v.At("timing");
    if (nonfrozen)
    {
        timing.Object({"sampled_at_ns", "collection_completed_at_ns", "encode_ns", "decode_ns",
                       "plan_created_at_ns", "execute_at_ns", "deadline_ns"});
        Require(timing.At("sampled_at_ns").I() == ref.sampledAt &&
                    timing.At("collection_completed_at_ns").I() >= ref.sampledAt &&
                    timing.At("plan_created_at_ns").I() == AddTime(
                        AddTime(timing.At("collection_completed_at_ns").I(), timing.At("encode_ns").I()),
                        timing.At("decode_ns").I()) &&
                    timing.At("execute_at_ns").I() >= timing.At("plan_created_at_ns").I(),
                "noncausal nonfrozen plan timing");
    }
    else
    {
        timing.Object({"sampled_at_ns",
                   "report_received_at_ns",
                   "plan_created_at_ns",
                   "command_received_at_ns",
                   "execute_at_ns",
                   "deadline_ns"});
        for (const auto* key : {"sampled_at_ns",
                            "report_received_at_ns",
                            "plan_created_at_ns",
                            "command_received_at_ns",
                            "execute_at_ns"})
        {
            Require(timing.At(key).I() == ref.sampledAt, "synchronous plan timing mismatch");
        }
    }
    Require(timing.At("deadline_ns").I() == AddTime(ref.sampledAt, period), "plan deadline");
    Action result{ref, {}};
    for (const auto& key : action.At("links").Array())
    {
        result.links.push_back(ReadLink(key));
    }
    return result;
}

Value
WriteObservation(const Observation& o)
{
    auto result = Containers(o.state, o.config, o.reference.sampledAt);
    result["reference"] = WriteReference(o.reference);
    result["node_ids"] = Ids(o.config.nodes);
    result["packet_size_bytes"] = o.config.packetBytes;
    result["kind"] = "queue_state";
    result["pending_packet_ids"] = List{};
    result["periods_remaining"] = o.remaining;
    result["routes"] = Many(o.state.routes, [](const auto& r) {
        return Map{{"node_id", r.node},
                   {"destination", r.destination},
                   {"next_hop", Opt(r.nextHop)}};
    });
    return result;
}

Value
WritePhysical(const Value& source, const Reference& ref)
{
    auto value = std::get<Map>(source.data);
    value.erase("at_ns");
    value["reference"] = WriteReference(ref);
    return value;
}

Value
WriteCycle(const CycleResult& c)
{
    auto post = Containers(c.post.state, c.next.config, c.post.sampledAt);
    post["reference"] = WriteReference(c.post.reference);
    post["sampled_at_ns"] = c.post.sampledAt;
    post["stage"] = "post_service";
    post["pending"] = Many(c.post.pending, [](const auto& p) {
        return Map{{"packet_id", p.packet},
                   {"node_id", p.node},
                   {"node_arrived_at_ns", p.arrived},
                   {"origin", p.origin}};
    });
    return Map{{"action",
                Map{{"reference", WriteReference(c.action.reference)},
                    {"links", Many(c.action.links, WriteLink)}}},
               {"ended_at_ns", c.endedAt},
               {"post_service", post},
               {"initial_ids", Ids(c.initialIds)},
               {"new_source_ids", Ids(c.newIds)},
               {"risk_ids", Ids(c.riskIds)},
               {"next_observation", WriteObservation(c.next)},
               {"reward_status", "not_computed"},
               {"services",
                Many(c.services,
                     [](const auto& s) {
                         return Map{{"key", WriteLink(s.key)},
                                    {"packet_ids", Ids(s.packets)},
                                    {"bytes_served", s.bytes}};
                     })},
               {"events", Many(c.events, [](const auto& e) {
                    return Map{{"kind", e.kind},
                               {"packet_id", e.packet},
                               {"at_ns", e.at},
                               {"node_id", e.node},
                               {"link", e.link ? WriteLink(*e.link) : Value()},
                               {"origin", Opt(e.origin)},
                               {"terminal_class", Opt(e.terminalClass)},
                               {"reason", Opt(e.reason)},
                               {"end_to_end_ns", Opt(e.endToEnd)}};
                })}};
}
} // namespace ns3::fanet::wire
