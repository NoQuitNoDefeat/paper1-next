#ifndef FANET_CONTROLLED_STATE_H
#define FANET_CONTROLLED_STATE_H

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ns3::fanet
{

using Id = uint64_t;             //!< Stable research ID; zero is valid, never a Node/Packet UID.
using Bytes = uint64_t;          //!< Integer application bytes, not protocol bytes.
using Ns = int64_t;              //!< Nonnegative nanoseconds within ns-3 Time's signed range.
using Link = std::pair<Id, Id>;   //!< Registered directed next-hop queue.

/** Immutable packet identity; its containing state owns its active location. */
struct Packet
{
    Id id{};
    Id source{};
    Id destination{};
    Ns born{};
    std::optional<Ns> deadline;
    bool operator==(const Packet&) const = default;
};

/** Arrival and admission times at the current node; FIFO order is explicit. */
struct Entry
{
    Id packet{};
    Ns arrived{};
    Ns entered{};
    bool operator==(const Entry&) const = default;
};

struct QueueConfig
{
    Link key;
    Bytes capacity{};
    bool operator==(const QueueConfig&) const = default;
};

struct WaitingConfig
{
    Id node{};
    Bytes capacity{};
    Ns maxWait{};
    bool operator==(const WaitingConfig&) const = default;
};

/** Explicit supported configuration; flags are rejected when enabled. */
struct Config
{
    std::vector<Id> nodes;
    std::vector<QueueConfig> queues;
    std::vector<WaitingConfig> waiting;
    Bytes packetBytes{};
    Ns start{};
    Ns end{};
    Ns period{};
    bool packetDeadlines{false};
    bool retransmissions{false};
    // paper1-next semantics (optional INIT fields; defaults keep the original ledger):
    // re-admit queued packets whose next hop changed when routes are updated, and keep a
    // restorable waiting packet in its area when the target queue is full.
    bool rehomeStaleQueues{false};
    bool keepRestoredWaiting{false};
    bool operator==(const Config&) const = default;
};

struct Queue
{
    Link key;
    std::vector<Entry> entries;
    bool operator==(const Queue&) const = default;
};

struct Waiting
{
    Id node{};
    std::vector<Entry> entries;
    bool operator==(const Waiting&) const = default;
};

/** Missing record and explicit no-route (null nextHop) have different meanings. */
struct Route
{
    Id node{};
    Id destination{};
    std::optional<Id> nextHop;
    bool operator==(const Route&) const = default;
};

/** Settled decision state. Post-service snapshots separately include pending arrivals. */
struct State
{
    std::vector<Packet> packets;
    std::vector<Queue> queues;
    std::vector<Waiting> waiting;
    std::vector<Route> routes;
    bool operator==(const State&) const = default;
};

struct Reference
{
    std::string run;
    uint64_t epoch{};
    Ns sampledAt{};
    bool operator==(const Reference&) const = default;
};

/** Value snapshot: callers cannot mutate the environment through its output. */
struct Observation
{
    Reference reference;
    Config config;
    State state;
    uint64_t remaining{};
    bool operator==(const Observation&) const = default;
};

/** Private service conditions, never expected packet outcomes or policy observations. */
struct ServiceCondition
{
    Link key;
    Bytes budget{};
    Ns completedAt{};
    bool available{false};
    std::optional<Bytes> rate;
};

struct Frame
{
    Ns start{};
    std::vector<ServiceCondition> services;
    std::vector<Packet> births;
    std::vector<Route> routeUpdates;
};

struct Action
{
    Reference reference;
    std::vector<Link> links;
};

struct Arrival
{
    Id packet{};
    Id node{};
    Ns arrived{};
    std::string origin;
};

struct PacketEvent
{
    std::string kind;
    Id packet{};
    Ns at{};
    Id node{};
    std::optional<Link> link;
    std::optional<std::string> origin;
    std::optional<std::string> terminalClass;
    std::optional<std::string> reason;
    std::optional<Ns> endToEnd;
    std::size_t order{}; //!< Semantic presentation order, distinct from raw callback order.
};

struct LinkService
{
    Link key;
    std::vector<Id> packets;
    Bytes bytes{};
};

struct PostService
{
    Reference reference;
    Ns sampledAt{};
    State state;
    std::vector<Arrival> pending;
};

/** Raw controller trace; records only this controller's owned events, not radio events. */
struct CallbackRecord
{
    Ns at{};
    std::string stage;
    uint64_t outstanding{};
};

struct CycleResult
{
    Action action;
    Ns endedAt{};
    std::vector<LinkService> services;
    std::vector<PacketEvent> events;
    PostService post;
    std::vector<Id> initialIds;
    std::vector<Id> newIds;
    std::vector<Id> riskIds;
    Observation next;
    std::vector<CallbackRecord> callbacks;
};

/** @return Checked nonnegative sum; throws before signed overflow. */
Ns AddTime(Ns a, Ns b);
/** @return Checked byte occupancy; throws before unsigned overflow. */
Bytes Occupancy(std::size_t count, Bytes packetBytes);
/** Validate a settled initial boundary without modifying or sorting its records. */
void ValidateInitial(const Config& config, const State& state);
/** Validate route keys and registered next-hop targets, including duplicate queries. */
void ValidateRoutes(const Config& config, const std::vector<Route>& routes);
/** Validate the entire private finite trajectory without consuming its frames. */
void ValidateTrajectory(const Config& config,
                        const State& initial,
                        const std::vector<Frame>& frames);
/** @return Exact integer packet duration; rejects overflow and fractional nanoseconds. */
Ns PacketDuration(Bytes packetBytes, Bytes rate);
/** @return floor(rate * duration / 1e9), checked without overflowing the product. */
Bytes RateBudget(Bytes rate, Ns duration);

} // namespace ns3::fanet
#endif
