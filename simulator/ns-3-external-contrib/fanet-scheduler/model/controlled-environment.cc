#include "controlled-environment.h"

#include "ns3/simulator.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <tuple>

namespace ns3::fanet
{
NS_OBJECT_ENSURE_REGISTERED(ControlledEnvironment);

std::array<std::string, 3>
ControlledBuildInfo()
{
    return {FANET_PROBE_SOURCE_SHA256, FANET_PROBE_NS3_COMMIT, FANET_PROBE_COMPILER};
}

TypeId
ControlledEnvironment::GetTypeId()
{
    static TypeId tid = TypeId("ns3::fanet::ControlledEnvironment")
                            .SetParent<Object>()
                            .SetGroupName("FanetScheduler")
                            .AddConstructor<ControlledEnvironment>();
    return tid;
}

void
ControlledEnvironment::Reset(const Config& config,
                             const State& state,
                             const std::string& run,
                             const std::optional<std::vector<Frame>>& trajectory)
{
    const bool printable = run.size() <= 128 && std::ranges::all_of(run, [](unsigned char ch) {
                               return ch >= 33 && ch < 127;
                           });
    if (m_busy || m_disposed || run.empty() || !printable ||
        std::ranges::find(m_runs, run) != m_runs.end())
    {
        throw std::invalid_argument("reset requires live object and fresh printable ASCII run ID");
    }
    if (Time::GetResolution() != Time::NS || Simulator::Now().GetNanoSeconds() > config.start)
    {
        throw std::invalid_argument("reset requires nanosecond resolution and no past start");
    }
    ValidateInitial(config, state);
    if (trajectory)
    {
        ValidateTrajectory(config, state, *trajectory);
    }
    Observation next{{run, 0, config.start},
                     config,
                     state,
                     static_cast<uint64_t>((config.end - config.start) / config.period)};
    // Configuration order defines the published container order, not map iteration or input order.
    next.state.queues.clear();
    for (const auto& q : config.queues)
    {
        next.state.queues.push_back(*std::ranges::find(state.queues, q.key, &Queue::key));
    }
    next.state.waiting.clear();
    for (const auto& w : config.waiting)
    {
        next.state.waiting.push_back(*std::ranges::find(state.waiting, w.node, &Waiting::node));
    }
    // Prepare all owning copies before replacing the prior published episode.
    auto nextTrajectory = trajectory;
    auto runs = m_runs;
    runs.push_back(run);
    std::map<Id, Ptr<Node>> nodes;
    for (Id id : config.nodes)
    {
        nodes.emplace(id, CreateObject<Node>());
    }
    for (auto& [id, node] : m_nodes)
    {
        node->Dispose();
    }
    m_nodes = std::move(nodes);
    m_observation = std::move(next);
    m_runs = std::move(runs);
    m_trajectory = std::move(nextTrajectory);
    m_result.reset();
    m_failure = nullptr;
}

Observation
ControlledEnvironment::Observe(const std::optional<Reference>& expected) const
{
    if (m_failure)
    {
        std::rethrow_exception(m_failure);
    }
    if (m_busy)
    {
        throw std::logic_error("observation unavailable during a cycle");
    }
    if (m_disposed || !m_observation)
    {
        throw std::logic_error("observe requires an initialized live environment");
    }
    if (expected && *expected != m_observation->reference)
    {
        throw std::invalid_argument("stale observation reference");
    }
    return *m_observation;
}

Ptr<Node>
ControlledEnvironment::GetNode(Id id) const
{
    Observe();
    return m_nodes.at(id);
}

void
ControlledEnvironment::DoDispose()
{
    RemoveEvents();
    m_busy = false;
    m_disposed = true;
    m_observation.reset();
    for (auto& [id, node] : m_nodes)
    {
        node->Dispose();
    }
    m_nodes.clear();
    m_trajectory.reset();
    m_result.reset();
    m_work = {};
    m_pending.clear();
    Object::DoDispose();
}

namespace
{
std::vector<Route>
MergeRoutes(const State& state, const Frame& frame)
{
    std::map<Link, Route> routes;
    for (const auto& route : state.routes)
    {
        routes[{route.node, route.destination}] = route;
    }
    for (const auto& route : frame.routeUpdates)
    {
        routes[{route.node, route.destination}] = route;
    }
    std::vector<Route> result;
    for (const auto& [key, route] : routes)
    {
        result.push_back(route);
    }
    return result;
}

const Packet&
FindPacket(const State& state, Id id)
{
    auto found = std::ranges::find(state.packets, id, &Packet::id);
    if (found == state.packets.end())
    {
        throw std::logic_error("packet missing from active registry");
    }
    return *found;
}

const Route&
FindRoute(const std::vector<Route>& routes, Id node, Id destination)
{
    auto found = std::ranges::find_if(routes, [&](const auto& route) {
        return route.node == node && route.destination == destination;
    });
    if (found == routes.end())
    {
        throw std::invalid_argument("missing route query is not explicit no-route");
    }
    return *found;
}

std::vector<Id>
PacketIds(const State& state)
{
    std::vector<Id> ids;
    for (const auto& packet : state.packets)
    {
        ids.push_back(packet.id);
    }
    std::ranges::sort(ids);
    return ids;
}
} // namespace

void
ControlledEnvironment::ValidateAction(const Action& action) const
{
    const auto observation = Observe(action.reference);
    ValidateAgainstSample(action, observation);
}

void
ControlledEnvironment::ValidateAgainstSample(const Action& action,
                                               const Observation& observation) const
{
    if (action.reference != observation.reference)
    {
        throw std::invalid_argument("stale action reference");
    }
    if (!observation.remaining || !m_trajectory)
    {
        throw std::invalid_argument("advance requires remaining horizon and private trajectory");
    }
    const auto& config = observation.config;
    const auto& state = observation.state;
    const auto& frame = m_trajectory->at(action.reference.epoch);
    Occupancy(observation.state.packets.size() + frame.births.size(), config.packetBytes);
    std::set<Link> selected;
    for (const auto& key : action.links)
    {
        if (!selected.insert(key).second ||
            !std::ranges::any_of(config.queues, [&](const auto& q) { return q.key == key; }))
        {
            throw std::invalid_argument("duplicate or unregistered action link");
        }
    }
    const auto routes = MergeRoutes(state, frame);
    std::set<Link> queries;
    const auto end = AddTime(frame.start, config.period);
    for (const auto& packet : frame.births)
    {
        queries.emplace(packet.source, packet.destination);
    }
    for (std::size_t i = 0; i < state.waiting.size(); ++i)
    {
        for (const auto& entry : state.waiting[i].entries)
        {
            if (AddTime(entry.entered, config.waiting[i].maxWait) > end)
            {
                queries.emplace(state.waiting[i].node, FindPacket(state, entry.packet).destination);
            }
        }
    }
    for (const auto& queue : state.queues)
    {
        const auto& condition =
            *std::ranges::find(frame.services, queue.key, &ServiceCondition::key);
        if (!selected.contains(queue.key) || !condition.available)
        {
            continue;
        }
        const auto count =
            std::min<uint64_t>(queue.entries.size(), condition.budget / config.packetBytes);
        for (std::size_t i = 0; i < count; ++i)
        {
            const auto& packet = FindPacket(state, queue.entries[i].packet);
            if (packet.destination != queue.key.second)
            {
                queries.emplace(queue.key.second, packet.destination);
            }
        }
    }
    for (const auto& [node, destination] : queries)
    {
        const auto& route = FindRoute(routes, node, destination);
        if (!route.nextHop &&
            !std::ranges::any_of(config.waiting, [&](const auto& w) { return w.node == node; }))
        {
            throw std::invalid_argument("no-route needs a registered waiting area");
        }
    }
}

void
ControlledEnvironment::RemoveEvents()
{
    for (const auto& event : m_events)
    {
        if (event.IsPending())
        {
            Simulator::Remove(event);
        }
    }
    m_events.clear();
}

void
ControlledEnvironment::Record(const std::string& stage)
{
    m_result->callbacks.push_back({Simulator::Now().GetNanoSeconds(), stage, m_outstanding});
}

void
ControlledEnvironment::Guard(const std::function<void()>& work)
{
    try
    {
        if (!m_busy || m_disposed || m_failure)
        {
            throw std::logic_error("fact outside a live cycle");
        }
        work();
    }
    catch (...)
    {
        m_failure = std::current_exception();
        RemoveEvents();
        m_busy = false;
        Simulator::Stop();
    }
}

void
ControlledEnvironment::ScheduleWork(Ns at,
                                    const std::string& stage,
                                    const std::function<void()>& work)
{
    if (at < Simulator::Now().GetNanoSeconds() || at > m_result->endedAt)
    {
        throw std::logic_error("owned work outside current execution window");
    }
    ++m_outstanding;
    m_events.push_back(Simulator::Schedule(NanoSeconds(at) - Simulator::Now(), [this, stage, work] {
        Guard([&] {
            Record(stage);
            work();
            // A parent must register any same-time child before releasing this ticket.
            --m_outstanding;
            FinishIfReady();
        });
    }));
}

void
ControlledEnvironment::StartCycle(const Action& action,
                                  bool boundaryFirst,
                                  const std::function<void()>& externalExecutor,
                                  const std::set<Link>& failedLinks)
{
    ValidateAction(action);
    for (const auto& key : failedLinks)
    {
        if (externalExecutor || std::ranges::find(action.links, key) == action.links.end())
        {
            throw std::invalid_argument("failed links require a selected controlled execution");
        }
    }
    if (Simulator::Now().GetNanoSeconds() > action.reference.sampledAt)
    {
        throw std::invalid_argument("simulator has passed the decision boundary");
    }
    RemoveEvents();
    const auto& config = m_observation->config;
    const auto& frame = m_trajectory->at(action.reference.epoch);
    m_work = m_observation->state;
    m_pending.clear();
    m_admissionRoutes = MergeRoutes(m_work, frame);
    m_serviceTimes.clear();
    m_result = CycleResult{};
    m_result->action = action;
    m_result->endedAt = AddTime(frame.start, config.period);
    m_result->initialIds = PacketIds(m_work);
    for (const auto& packet : frame.births)
    {
        m_result->newIds.push_back(packet.id);
    }
    std::ranges::sort(m_result->newIds);
    m_result->riskIds = m_result->initialIds;
    m_result->riskIds.insert(m_result->riskIds.end(),
                             m_result->newIds.begin(),
                             m_result->newIds.end());
    std::ranges::sort(m_result->riskIds);
    m_order = 0;
    m_outstanding = 0;
    m_boundaryRequested = false;
    m_external = static_cast<bool>(externalExecutor);
    m_awaitingPlan = false;
    m_busy = true;
    Guard([&] {
        auto registerBoundary = [&] {
            m_events.push_back(
                Simulator::Schedule(NanoSeconds(m_result->endedAt) - Simulator::Now(), [this] {
                    Guard([&] {
                        m_boundaryRequested = true;
                        Record("close_requested");
                        FinishIfReady();
                    });
                }));
            // Route and expiry callbacks record readiness; only Settle applies their semantics.
            ScheduleWork(m_result->endedAt, "boundary_inputs", [this] {
                ScheduleWork(m_result->endedAt, "boundary_inputs_complete", [] {});
            });
        };
        if (boundaryFirst)
        {
            registerBoundary();
        }
        for (const auto& packet : frame.births)
        {
            const auto order = m_order++;
            ScheduleWork(packet.born, "birth_ready", [this, packet, order] {
                ScheduleWork(packet.born, "birth_complete", [this, packet, order] {
                    m_work.packets.push_back(packet);
                    m_pending.push_back({packet.id, packet.source, packet.born, "source"});
                    m_result->events.push_back({"birth",
                                                packet.id,
                                                packet.born,
                                                packet.source,
                                                std::nullopt,
                                                "source",
                                                std::nullopt,
                                                std::nullopt,
                                                std::nullopt,
                                                order});
                });
            });
        }
        for (std::size_t i = 0; i < config.queues.size(); ++i)
        {
            const auto key = config.queues[i].key;
            const auto& condition = *std::ranges::find(frame.services, key, &ServiceCondition::key);
            const bool selected = std::ranges::find(action.links, key) != action.links.end();
            const auto count = !m_external && selected && condition.available &&
                                       !failedLinks.contains(key)
                                   ? std::min<uint64_t>(m_work.queues[i].entries.size(),
                                                        condition.budget / config.packetBytes)
                                   : 0;
            m_result->services.push_back({key, {}, 0});
            Ns at = frame.start;
            for (std::size_t n = 0; n < count; ++n)
            {
                at = condition.rate
                         ? AddTime(at, PacketDuration(config.packetBytes, *condition.rate))
                         : condition.completedAt;
                const auto packet = m_work.queues[i].entries[n].packet;
                m_serviceTimes.emplace(std::make_pair(i, packet), at);
                const auto order = m_order;
                m_order += 2;
                ScheduleWork(at, "service_ready", [this, i, packet, order, at] {
                    ScheduleWork(at, "service_complete", [this, i, packet, order] {
                        CompleteService(i, packet, order);
                    });
                });
            }
        }
        if (!boundaryFirst)
        {
            registerBoundary();
        }
        if (m_external)
        {
            ++m_outstanding;
            externalExecutor();
        }
    });
    if (m_failure)
    {
        std::rethrow_exception(m_failure);
    }
}

Observation
ControlledEnvironment::BeginExternalCollection()
{
    const auto sample = Observe();
    StartCycle({sample.reference, {}}, true, [] {});
    m_awaitingPlan = true;
    return sample;
}

Observation
ControlledEnvironment::ExternalSample(const Reference& reference) const
{
    if (m_failure)
    {
        std::rethrow_exception(m_failure);
    }
    if (!m_busy || !m_external || !m_observation || m_observation->reference != reference)
    {
        throw std::logic_error("no active external cycle for this immutable sample");
    }
    return *m_observation;
}

void
ControlledEnvironment::AcceptExternalPlan(const Action& action)
{
    const auto sample = ExternalSample(action.reference);
    if (!m_awaitingPlan || Simulator::Now().GetNanoSeconds() >= m_result->endedAt)
    {
        throw std::logic_error("external plan is late or already accepted");
    }
    ValidateAgainstSample(action, sample);
    m_result->action = action;
    m_awaitingPlan = false;
}

void
ControlledEnvironment::ValidateExternalExecution(const Action& action) const
{
    const auto sample = ExternalSample(action.reference);
    if (m_awaitingPlan)
    {
        throw std::logic_error("external DATA before a complete plan");
    }
    ValidateAgainstSample(action, sample);
    for (const auto& link : action.links)
    {
        if (std::ranges::find(m_result->action.links, link) == m_result->action.links.end())
        {
            throw std::invalid_argument("external execution cannot add a link to the plan");
        }
    }
}

void
ControlledEnvironment::ConfirmExternalService(Link key, Id packet)
{
    Guard([&] {
        if (!m_external)
        {
            throw std::logic_error("external service without execution ticket");
        }
        const auto& queues = m_observation->config.queues;
        const auto found = std::ranges::find(queues, key, &QueueConfig::key);
        if (found == queues.end())
        {
            throw std::logic_error("external service for unknown queue");
        }
        const auto index = static_cast<std::size_t>(found - queues.begin());
        const auto at = Simulator::Now().GetNanoSeconds();
        m_serviceTimes.emplace(std::make_pair(index, packet), at);
        Record("wireless_confirmation");
        CompleteService(index, packet, m_order);
        m_order += 2;
    });
    if (m_failure)
    {
        std::rethrow_exception(m_failure);
    }
}

void
ControlledEnvironment::FinishExternalExecution()
{
    Guard([&] {
        if (!m_external || m_outstanding == 0)
        {
            throw std::logic_error("external execution already released");
        }
        m_external = false;
        --m_outstanding;
        Record("wireless_drained");
        FinishIfReady();
    });
    if (m_failure)
    {
        std::rethrow_exception(m_failure);
    }
}

void
ControlledEnvironment::CompleteService(std::size_t queueIndex, Id packetId, std::size_t order)
{
    auto& queue = m_work.queues.at(queueIndex);
    const auto at = Simulator::Now().GetNanoSeconds();
    const auto& action = m_result->action;
    const bool selected = std::ranges::find(action.links, queue.key) != action.links.end();
    const auto ticket = m_serviceTimes.find({queueIndex, packetId});
    if (!selected || at <= action.reference.sampledAt || at > m_result->endedAt ||
        queue.entries.empty() || queue.entries.front().packet != packetId ||
        ticket == m_serviceTimes.end() || ticket->second != at)
    {
        throw std::logic_error("duplicate, non-FIFO or unauthorized service completion");
    }
    const auto packet = FindPacket(m_work, packetId);
    m_serviceTimes.erase(ticket);
    queue.entries.erase(queue.entries.begin());
    auto& service = m_result->services.at(queueIndex);
    service.packets.push_back(packetId);
    service.bytes = Occupancy(service.packets.size(), m_observation->config.packetBytes);
    m_result->events.push_back({"service",
                                packetId,
                                at,
                                queue.key.second,
                                queue.key,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                order});
    if (queue.key.second == packet.destination)
    {
        m_result->events.push_back({"delivery",
                                    packetId,
                                    at,
                                    packet.destination,
                                    queue.key,
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    at - packet.born,
                                    order + 1});
        std::erase_if(m_work.packets, [&](const auto& p) { return p.id == packetId; });
    }
    else
    {
        m_pending.push_back({packetId, queue.key.second, at, "relay"});
    }
}

void
ControlledEnvironment::FinishIfReady()
{
    if (m_boundaryRequested && m_outstanding == 0)
    {
        if (!m_serviceTimes.empty() || Simulator::Now().GetNanoSeconds() != m_result->endedAt)
        {
            throw std::logic_error("settlement did not occur at the exact boundary");
        }
        Settle();
        m_busy = false;
        // The caller may Run again for the next period; no time epsilon is inserted.
        Simulator::Stop();
    }
}

void
ControlledEnvironment::Settle()
{
    const auto& config = m_observation->config;
    const auto end = m_result->endedAt;
    auto sortPending = [&] {
        std::ranges::sort(m_pending, [](const auto& a, const auto& b) {
            return std::tie(a.arrived, a.packet) < std::tie(b.arrived, b.packet);
        });
    };
    std::ranges::sort(m_work.packets, {}, &Packet::id);
    sortPending();
    Record("post_service");
    m_result->post = {m_observation->reference, end, m_work, m_pending};
    auto terminate = [&](Id packet,
                         Id node,
                         const std::string& category,
                         const std::string& reason,
                         const std::string& origin,
                         std::optional<Link> link = std::nullopt) {
        m_result->events.push_back({"terminal",
                                    packet,
                                    end,
                                    node,
                                    link,
                                    origin,
                                    category,
                                    reason,
                                    std::nullopt,
                                    m_order++});
        std::erase_if(m_work.packets, [&](const auto& p) { return p.id == packet; });
    };
    Record("expiry_and_restoration");
    for (std::size_t i = 0; i < m_work.waiting.size(); ++i)
    {
        auto& area = m_work.waiting[i];
        std::vector<Entry> survivors;
        for (const auto& entry : area.entries)
        {
            if (AddTime(entry.entered, config.waiting[i].maxWait) <= end)
            {
                terminate(entry.packet, area.node, "deadline", "route_wait_timeout", "waiting");
            }
            else if (FindRoute(m_admissionRoutes,
                               area.node,
                               FindPacket(m_work, entry.packet).destination)
                         .nextHop)
            {
                m_pending.push_back({entry.packet, area.node, entry.arrived, "waiting"});
            }
            else
            {
                survivors.push_back(entry);
            }
        }
        area.entries = std::move(survivors);
    }
    Record("unified_admission");
    sortPending();
    for (const auto& arrival : m_pending)
    {
        const auto& packet = FindPacket(m_work, arrival.packet);
        const auto hop = FindRoute(m_admissionRoutes, arrival.node, packet.destination).nextHop;
        if (hop)
        {
            const Link key{arrival.node, *hop};
            auto queue = std::ranges::find(m_work.queues, key, &Queue::key);
            const auto capacity =
                std::ranges::find(config.queues, key, &QueueConfig::key)->capacity;
            if (queue->entries.size() >= capacity / config.packetBytes)
            {
                terminate(arrival.packet,
                          arrival.node,
                          "overflow",
                          "queue_capacity",
                          arrival.origin,
                          key);
            }
            else
            {
                queue->entries.push_back({arrival.packet, arrival.arrived, end});
                m_result->events.push_back({"queue_admit",
                                            arrival.packet,
                                            end,
                                            arrival.node,
                                            key,
                                            arrival.origin,
                                            std::nullopt,
                                            std::nullopt,
                                            std::nullopt,
                                            m_order++});
            }
        }
        else
        {
            auto area = std::ranges::find(m_work.waiting, arrival.node, &Waiting::node);
            const auto capacity =
                std::ranges::find(config.waiting, arrival.node, &WaitingConfig::node)->capacity;
            if (area->entries.size() >= capacity / config.packetBytes)
            {
                terminate(arrival.packet,
                          arrival.node,
                          "overflow",
                          "waiting_capacity",
                          arrival.origin);
            }
            else
            {
                area->entries.push_back({arrival.packet, arrival.arrived, end});
                m_result->events.push_back({"wait_admit",
                                            arrival.packet,
                                            end,
                                            arrival.node,
                                            std::nullopt,
                                            arrival.origin,
                                            std::nullopt,
                                            std::nullopt,
                                            std::nullopt,
                                            m_order++});
            }
        }
    }
    std::vector<Id> active;
    for (const auto& q : m_work.queues)
    {
        for (const auto& entry : q.entries)
        {
            active.push_back(entry.packet);
        }
    }
    for (const auto& w : m_work.waiting)
    {
        for (const auto& entry : w.entries)
        {
            active.push_back(entry.packet);
        }
    }
    std::ranges::sort(active);
    auto accounted = active;
    for (const auto& event : m_result->events)
    {
        if (event.kind == "terminal" || event.kind == "delivery")
        {
            accounted.push_back(event.packet);
        }
    }
    std::ranges::sort(accounted);
    if (active != PacketIds(m_work) || accounted != m_result->riskIds)
    {
        throw std::logic_error("active/terminal/delivery packet conservation failed");
    }
    for (std::size_t i = 0; i < config.queues.size(); ++i)
    {
        const auto before =
            Occupancy(m_observation->state.queues[i].entries.size(), config.packetBytes);
        const auto after =
            Occupancy(m_result->post.state.queues[i].entries.size(), config.packetBytes);
        if (before < m_result->services[i].bytes || before - m_result->services[i].bytes != after)
        {
            throw std::logic_error("per-queue service byte conservation failed");
        }
    }
    std::ranges::sort(m_work.packets, {}, &Packet::id);
    m_work.routes = m_admissionRoutes;
    std::ranges::sort(m_result->events, [](const auto& a, const auto& b) {
        return std::tie(a.at, a.order) < std::tie(b.at, b.order);
    });
    m_result->next = {{m_observation->reference.run, m_observation->reference.epoch + 1, end},
                      config,
                      m_work,
                      m_observation->remaining - 1};
    Record("published");
    m_observation = m_result->next;
}

CycleResult
ControlledEnvironment::Result() const
{
    if (m_failure)
    {
        std::rethrow_exception(m_failure);
    }
    if (m_disposed || m_busy || !m_result)
    {
        throw std::logic_error("complete final report is unavailable");
    }
    return *m_result;
}

bool
ControlledEnvironment::IsBusy() const
{
    return m_busy;
}
} // namespace ns3::fanet
