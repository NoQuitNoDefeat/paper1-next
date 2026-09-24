#include "controlled-state.h"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>

namespace ns3::fanet
{
namespace
{
void
Require(bool condition, const std::string& context)
{
    if (!condition)
    {
        throw std::invalid_argument(context);
    }
}

void
Node(Id id, const Config& config)
{
    Require(std::ranges::find(config.nodes, id) != config.nodes.end(), "unregistered node");
}

void
LinkKey(Link key, const Config& config)
{
    Node(key.first, config);
    Node(key.second, config);
    Require(key.first != key.second, "self link is not a next-hop queue");
}
} // namespace

Ns
AddTime(Ns a, Ns b)
{
    Require(a >= 0 && b >= 0 && a <= std::numeric_limits<Ns>::max() - b,
            "negative or overflowing time");
    return a + b;
}

Bytes
Occupancy(std::size_t count, Bytes packetBytes)
{
    Require(packetBytes > 0 && count <= std::numeric_limits<Bytes>::max() / packetBytes,
            "zero packet size or overflowing occupancy");
    return count * packetBytes;
}

void
ValidateRoutes(const Config& config, const std::vector<Route>& routes)
{
    std::set<Link> seen;
    for (const auto& route : routes)
    {
        Node(route.node, config);
        Node(route.destination, config);
        Require(seen.emplace(route.node, route.destination).second, "duplicate route query");
        if (route.nextHop)
        {
            Node(*route.nextHop, config);
            const Link key{route.node, *route.nextHop};
            Require(std::ranges::any_of(config.queues, [&](const auto& q) { return q.key == key; }),
                    "route next hop needs a registered queue");
        }
    }
}

void
ValidateInitial(const Config& config, const State& state)
{
    std::set<Id> nodes(config.nodes.begin(), config.nodes.end());
    Require(nodes.size() == config.nodes.size(), "duplicate node ID");
    Require(config.packetBytes > 0, "packet bytes must be positive");
    Require(config.start >= 0 && config.end > config.start && config.period > 0,
            "invalid finite time range");
    Require((config.end - config.start) % config.period == 0, "range must contain whole periods");
    Require(!config.packetDeadlines && !config.retransmissions, "unsupported mechanism enabled");
    std::map<Link, Bytes> queueCaps;
    for (const auto& queue : config.queues)
    {
        LinkKey(queue.key, config);
        Require(queue.capacity > 0, "queue capacity must be positive");
        Require(queueCaps.emplace(queue.key, queue.capacity).second, "duplicate queue config");
    }
    std::map<Id, WaitingConfig> waitConfigs;
    for (const auto& area : config.waiting)
    {
        Node(area.node, config);
        Require(area.capacity > 0 && area.maxWait > 0, "invalid waiting limits");
        Require(area.maxWait % config.period == 0, "waiting limit must use whole periods");
        // Admission can occur at the final boundary too. Never defer overflow to execution.
        AddTime(config.end, area.maxWait);
        Require(waitConfigs.emplace(area.node, area).second, "duplicate waiting config");
    }
    std::map<Id, Packet> packets;
    for (const auto& packet : state.packets)
    {
        Node(packet.source, config);
        Node(packet.destination, config);
        Require(packet.born >= 0 && packet.born <= config.start, "initial birth out of range");
        Require(!packet.deadline, "packet deadline not supported");
        Require(packets.emplace(packet.id, packet).second, "duplicate packet ID");
    }
    std::set<Id> active;
    auto entryCheck = [&](const Entry& entry, Id node) {
        Require(packets.contains(entry.packet), "unregistered packet entry");
        Require(active.insert(entry.packet).second, "packet occupies multiple positions");
        const auto& packet = packets.at(entry.packet);
        Require(packet.born <= entry.arrived && entry.arrived <= entry.entered &&
                    entry.entered <= config.start,
                "require birth <= arrival <= entry <= boundary");
        Require(packet.destination != node, "active packet already at destination");
    };
    std::set<Link> seenQueues;
    for (const auto& queue : state.queues)
    {
        Require(queueCaps.contains(queue.key) && seenQueues.insert(queue.key).second,
                "unregistered or duplicate queue state");
        Ns last = -1;
        for (const auto& entry : queue.entries)
        {
            entryCheck(entry, queue.key.first);
            Require(entry.entered >= last, "FIFO entry times decrease");
            last = entry.entered;
        }
        Require(Occupancy(queue.entries.size(), config.packetBytes) <= queueCaps.at(queue.key),
                "initial queue capacity exceeded");
    }
    Require(seenQueues.size() == queueCaps.size(), "missing explicit queue state");
    std::set<Id> seenWaiting;
    for (const auto& area : state.waiting)
    {
        Require(waitConfigs.contains(area.node) && seenWaiting.insert(area.node).second,
                "unregistered or duplicate waiting state");
        const auto& limits = waitConfigs.at(area.node);
        for (const auto& entry : area.entries)
        {
            entryCheck(entry, area.node);
            Require((config.start - entry.entered) % config.period == 0,
                    "waiting entry not aligned to boundary");
            Require(AddTime(entry.entered, limits.maxWait) > config.start,
                    "initial waiting packet already due");
        }
        Require(Occupancy(area.entries.size(), config.packetBytes) <= limits.capacity,
                "initial waiting capacity exceeded");
    }
    Require(seenWaiting.size() == waitConfigs.size(), "missing explicit waiting state");
    Require(active.size() == packets.size(), "packet without active position");
    ValidateRoutes(config, state.routes);
}

Ns
PacketDuration(Bytes packetBytes, Bytes rate)
{
    Require(packetBytes > 0 && rate > 0, "packet bytes and rate must be positive");
    const auto divisor = std::gcd(packetBytes, rate);
    const auto numerator = packetBytes / divisor;
    const auto denominator = rate / divisor;
    Require(1000000000 % denominator == 0, "packet duration must be exact integer ns");
    const auto factor = 1000000000 / denominator;
    Require(numerator <= static_cast<Bytes>(std::numeric_limits<Ns>::max()) / factor,
            "packet duration overflow");
    return static_cast<Ns>(numerator * factor);
}

Bytes
RateBudget(Bytes rate, Ns duration)
{
    Require(rate > 0 && duration >= 0, "invalid rate or service duration");
    // Split both operands around 1e9 so no intermediate needs nonstandard 128-bit integers.
    constexpr Bytes scale = 1000000000;
    const auto t = static_cast<Bytes>(duration);
    const auto whole = Occupancy(t / scale, rate);
    const auto part = rate / scale == 0 ? 0 : Occupancy(t % scale, rate / scale);
    const auto tail = (t % scale) * (rate % scale) / scale;
    Require(part <= std::numeric_limits<Bytes>::max() - tail, "rate budget overflow");
    Require(whole <= std::numeric_limits<Bytes>::max() - part - tail, "rate budget overflow");
    return whole + part + tail;
}

void
ValidateTrajectory(const Config& config, const State& initial, const std::vector<Frame>& frames)
{
    Require(frames.size() == static_cast<uint64_t>((config.end - config.start) / config.period),
            "trajectory must cover the complete finite range");
    std::set<Id> ids;
    for (const auto& packet : initial.packets)
    {
        ids.insert(packet.id);
    }
    Ns start = config.start;
    for (const auto& frame : frames)
    {
        const auto end = AddTime(start, config.period);
        Require(frame.start == start, "trajectory frames must be contiguous and ordered");
        std::set<Link> seen;
        for (const auto& condition : frame.services)
        {
            Require(std::ranges::any_of(config.queues,
                                       [&](const auto& q) { return q.key == condition.key; }) &&
                        seen.insert(condition.key).second,
                    "unregistered or duplicate service queue");
            Require(start < condition.completedAt && condition.completedAt <= end,
                    "service completion outside (start, end]");
            if (condition.rate)
            {
                PacketDuration(config.packetBytes, *condition.rate);
                Require(condition.budget == RateBudget(*condition.rate, config.period),
                        "budget inconsistent with constant rate");
            }
        }
        Require(seen.size() == config.queues.size(), "missing explicit service condition");
        for (const auto& packet : frame.births)
        {
            Node(packet.source, config);
            Node(packet.destination, config);
            Require(packet.source != packet.destination, "local birth outside transport scope");
            Require(start < packet.born && packet.born <= end, "birth outside (start, end]");
            Require(!packet.deadline, "packet deadline not supported");
            Require(ids.insert(packet.id).second, "packet ID reused within trajectory");
        }
        ValidateRoutes(config, frame.routeUpdates);
        start = end;
    }
}
} // namespace ns3::fanet
