#include "ns3/controlled-environment.h"
#include "ns3/simulator.h"
#include "ns3/test.h"

#include <functional>
#include <limits>
#include <stdexcept>

using namespace ns3;
using namespace ns3::fanet;

namespace
{
Config
BaseConfig()
{
    return {{0, 9}, {{{0, 9}, 2000}}, {{0, 1000, 100000000}}, 1000,
            50000000, 150000000, 50000000, false, false};
}

State
BaseState()
{
    return {{{0, 0, 9, 0, std::nullopt}}, {{{0, 9}, {{0, 0, 0}}}},
            {{0, {}}}, {{0, 9, 9}}};
}

bool
Rejected(const std::function<void()>& operation)
{
    try
    {
        operation();
    }
    catch (const std::exception&)
    {
        return true;
    }
    return false;
}

class InitialCase : public TestCase
{
  public:
    explicit InitialCase(std::string name)
        : TestCase("controlled initialization " + name),
          m_name(std::move(name))
    {
    }

  private:
    void DoRun() override
    {
        Simulator::Destroy();
        auto env = CreateObject<ControlledEnvironment>();
        auto config = BaseConfig();
        auto state = BaseState();
        env->Reset(config, state, "native-1");
        const auto original = env->Observe();
        if (m_name == "value snapshot and zero identity")
        {
            auto copy = env->Observe();
            copy.state.queues.front().entries.clear();
            NS_TEST_ASSERT_MSG_EQ(env->Observe() == original, true, "Output is not mutable state");
            NS_TEST_ASSERT_MSG_EQ(env->GetNode(0) != env->GetNode(9), true,
                                  "Explicit node mapping");
            NS_TEST_ASSERT_MSG_EQ(original.state.packets.front().id, 0, "Zero is a real packet ID");
            NS_TEST_ASSERT_MSG_EQ(original.remaining, 2, "Full finite horizon");
            NS_TEST_ASSERT_MSG_EQ(Simulator::Now().GetNanoSeconds(), 0, "Reset consumes no time");
        }
        else if (m_name == "fresh reset and stale reference")
        {
            env->Reset(config, state, "native-2");
            NS_TEST_ASSERT_MSG_EQ(Rejected([&] { env->Observe(original.reference); }),
                                  true, "Old run rejected");
            NS_TEST_ASSERT_MSG_EQ(Rejected([&] { env->Reset(config, state, "native-1"); }),
                                  true, "Run identity cannot be reused");
        }
        else if (m_name == "uninitialized and disposed")
        {
            auto empty = CreateObject<ControlledEnvironment>();
            NS_TEST_ASSERT_MSG_EQ(Rejected([&] { empty->Observe(); }), true, "No fake observation");
            empty->Dispose();
            env->Dispose();
            NS_TEST_ASSERT_MSG_EQ(Rejected([&] { env->Observe(); }), true, "Disposed is closed");
        }
        else
        {
            if (m_name == "duplicate node")
            {
                config.nodes.push_back(0);
            }
            else if (m_name == "duplicate packet location")
            {
                state.waiting.front().entries.push_back({0, 0, 0});
            }
            else if (m_name == "missing empty container")
            {
                state.waiting.clear();
            }
            else if (m_name == "FIFO order")
            {
                state.packets.push_back({2, 0, 9, 0, std::nullopt});
                state.queues.front().entries = {{0, 0, 20}, {2, 0, 10}};
            }
            else if (m_name == "capacity")
            {
                config.queues.front().capacity = 999;
            }
            else if (m_name == "overdue wait")
            {
                config.waiting.front().maxWait = 50000000;
                state.queues.front().entries.clear();
                state.waiting.front().entries.push_back({0, 0, 0});
            }
            else if (m_name == "future birth")
            {
                state.packets.front().born = 50000001;
            }
            else if (m_name == "invalid route")
            {
                state.routes.front().nextHop = 0;
            }
            else if (m_name == "unlocated packet")
            {
                state.queues.front().entries.clear();
            }
            else if (m_name == "time overflow")
            {
                config.waiting.front().maxWait =
                    (std::numeric_limits<Ns>::max() / 50000000) * 50000000;
            }
            else if (m_name == "unsupported mechanism")
            {
                config.retransmissions = true;
            }
            NS_TEST_ASSERT_MSG_EQ(Rejected([&] { env->Reset(config, state, "invalid"); }),
                                  true, "Malformed initialization explicitly fails");
            NS_TEST_ASSERT_MSG_EQ(env->Observe() == original, true, "Failed reset is atomic");
        }
        env->Dispose();
        Simulator::Destroy();
    }
    std::string m_name;
};

class FaultEnvironment : public ControlledEnvironment
{
  public:
    void DuplicateFact()
    {
        Guard([&] {
            CompleteService(0, 0, 0);
            CompleteService(0, 0, 2);
        });
    }
};

class CycleCase : public TestCase
{
  public:
    explicit CycleCase(std::string name)
        : TestCase("controlled cycle " + name),
          m_name(std::move(name))
    {
    }

  private:
    void DoRun() override
    {
        Simulator::Destroy();
        auto env = CreateObject<FaultEnvironment>();
        auto config = BaseConfig();
        config.end = 100000000;
        auto state = BaseState();
        Frame frame{config.start, {{{0, 9}, 1000, config.end, true, std::nullopt}}, {}, {}};
        if (m_name == "short budget")
        {
            frame.services.front().budget = 999;
        }
        if (m_name == "disconnected backlog")
        {
            frame.services.front().available = false;
        }
        if (m_name == "expiry before recovery")
        {
            config.waiting.front().maxWait = config.period;
            state.queues.front().entries.clear();
            state.waiting.front().entries = {{0, 0, config.start}};
            state.routes.front().nextHop = std::nullopt;
            frame.routeUpdates = {{0, 9, 9}};
        }
        if (m_name == "no same-cycle second hop")
        {
            config.nodes.push_back(20);
            config.queues.push_back({{9, 20}, 1000});
            state.packets.front().destination = 20;
            state.queues.push_back({{9, 20}, {}});
            state.routes.push_back({9, 20, 20});
            frame.services.push_back({{9, 20}, 1000, config.end, true, std::nullopt});
        }
        if (m_name == "exact constant rate")
        {
            frame.services.front().rate = 40000;
            frame.services.front().budget = 2000;
            state.packets.push_back({2, 0, 9, 0, std::nullopt});
            state.queues.front().entries.push_back({2, 0, 0});
        }
        env->Reset(config, state, "cycle-1", std::vector<Frame>{frame});
        const auto before = env->Observe();
        Action action{before.reference, {{0, 9}}};
        if (m_name == "no same-cycle second hop")
        {
            action.links.push_back({9, 20});
        }
        if (m_name == "invalid action atomic")
        {
            auto invalid = action;
            invalid.links.push_back({0, 9});
            NS_TEST_ASSERT_MSG_EQ(Rejected([&] { env->StartCycle(invalid); }), true,
                                  "Duplicate plan rejected before events");
            NS_TEST_ASSERT_MSG_EQ(env->Observe() == before, true, "Preflight preserves state");
        }
        if (m_name == "duplicate internal completion")
        {
            Simulator::Schedule(NanoSeconds(config.end), &FaultEnvironment::DuplicateFact, env);
        }
        env->StartCycle(action, m_name != "boundary registered last");
        NS_TEST_ASSERT_MSG_EQ(Rejected([&] { env->Observe(); }), true, "No partial observation");
        NS_TEST_ASSERT_MSG_EQ(Rejected([&] { env->Result(); }), true, "No incomplete result");
        if (m_name == "dispose cancels only owned events")
        {
            bool unrelated = false;
            Simulator::Schedule(NanoSeconds(config.end), [&] { unrelated = true; });
            env->Dispose();
            Simulator::Run();
            NS_TEST_ASSERT_MSG_EQ(unrelated, true, "Another owner's event survives disposal");
            NS_TEST_ASSERT_MSG_EQ(Rejected([&] { env->Result(); }), true, "No false final report");
        }
        else
        {
            Simulator::Run();
            if (m_name == "duplicate internal completion")
            {
                NS_TEST_ASSERT_MSG_EQ(Rejected([&] { env->Result(); }), true, "Execution poisoned");
                NS_TEST_ASSERT_MSG_EQ(Rejected([&] { env->Observe(); }), true, "Reset required");
                config.start = config.end;
                config.end += config.period;
                env->Reset(config, state, "recovered");
                NS_TEST_ASSERT_MSG_EQ(env->Observe().reference.run, "recovered", "Explicit reset");
            }
            else
            {
                const auto result = env->Result();
                NS_TEST_ASSERT_MSG_EQ(result.endedAt, 100000000, "Exact boundary");
                NS_TEST_ASSERT_MSG_EQ(Simulator::Now().GetNanoSeconds(), 100000000,
                                      "No time epsilon");
                NS_TEST_ASSERT_MSG_EQ(result.next.remaining, 0, "Final observation retained");
                NS_TEST_ASSERT_MSG_EQ(result.callbacks.back().stage, "published", "Publish last");
                NS_TEST_ASSERT_MSG_EQ(result.callbacks.back().outstanding, 0,
                                      "All tickets released");
                NS_TEST_ASSERT_MSG_EQ(Rejected([&] { env->StartCycle(action); }), true,
                                      "Completed action cannot consume another cycle");
                if (m_name == "short budget" || m_name == "disconnected backlog")
                {
                    NS_TEST_ASSERT_MSG_EQ(result.services.front().bytes, 0, "Normal zero service");
                    NS_TEST_ASSERT_MSG_EQ(result.next.state.queues.front().entries.front().entered,
                                          0, "HOL origin preserved");
                    NS_TEST_ASSERT_MSG_EQ(result.events.size(), 0, "No fictitious termination");
                }
                else if (m_name == "expiry before recovery")
                {
                    NS_TEST_ASSERT_MSG_EQ(result.post.state.waiting.front().entries.size(), 1,
                                          "Snapshot precedes expiry");
                    NS_TEST_ASSERT_MSG_EQ(result.next.state.packets.empty(), true, "Timeout wins");
                    NS_TEST_ASSERT_MSG_EQ(*result.events.front().reason, "route_wait_timeout",
                                          "Specific reason, shared deadline class");
                }
                else if (m_name == "no same-cycle second hop")
                {
                    NS_TEST_ASSERT_MSG_EQ(result.services[1].bytes, 0, "No second-hop service");
                    NS_TEST_ASSERT_MSG_EQ(result.post.pending.size(), 1, "Relay pending");
                    NS_TEST_ASSERT_MSG_EQ(result.next.state.queues[1].entries.front().entered,
                                          100000000, "Admission is the ending boundary");
                }
                else if (m_name == "exact constant rate")
                {
                    NS_TEST_ASSERT_MSG_EQ(result.services.front().bytes, 2000, "Two whole packets");
                    NS_TEST_ASSERT_MSG_EQ(result.events[0].at, 75000000, "First exact completion");
                    NS_TEST_ASSERT_MSG_EQ(result.events[2].at, 100000000, "Second at boundary");
                }
                else
                {
                    NS_TEST_ASSERT_MSG_EQ(result.services.front().bytes, 1000, "Whole service");
                    NS_TEST_ASSERT_MSG_EQ(result.events[1].kind, "delivery", "Final hop delivered");
                    NS_TEST_ASSERT_MSG_EQ(result.next.state.packets.empty(), true,
                                          "No active copy");
                }
            }
        }
        env->Dispose();
        Simulator::Destroy();
    }
    std::string m_name;
};

class BoundaryCase : public TestCase
{
  public:
    explicit BoundaryCase(bool numeric)
        : TestCase(numeric ? "controlled integer arithmetic limits"
                           : "controlled reset and consecutive periods"),
          m_numeric(numeric)
    {
    }
  private:
    void DoRun() override
    {
        if (m_numeric)
        {
            const auto maximum = std::numeric_limits<Bytes>::max();
            NS_TEST_ASSERT_MSG_EQ(RateBudget(maximum, 1000000000), maximum,
                                  "Valid quotient with overflowing naive product");
            NS_TEST_ASSERT_MSG_EQ(RateBudget(50000, 50000000), 2500, "Exact byte budget");
            NS_TEST_ASSERT_MSG_EQ(Rejected([&] { RateBudget(maximum, 1000000001); }),
                                  true, "True budget overflow rejected");
            NS_TEST_ASSERT_MSG_EQ(Rejected([&] { PacketDuration(maximum, 1); }),
                                  true, "Packet duration overflow rejected");
            NS_TEST_ASSERT_MSG_EQ(Rejected([&] { PacketDuration(1000, 13000000); }),
                                  true, "Fractional nanoseconds rejected");
            NS_TEST_ASSERT_MSG_EQ(Rejected([&] { Occupancy(2, maximum); }),
                                  true, "Byte occupancy overflow rejected");
            return;
        }
        Simulator::Destroy();
        auto env = CreateObject<ControlledEnvironment>();
        auto config = BaseConfig();
        auto state = BaseState();
        std::vector<Frame> frames{
            {50000000, {{{0, 9}, 600, 100000000, true, std::nullopt}}, {}, {}},
            {100000000, {{{0, 9}, 600, 150000000, true, std::nullopt}}, {}, {}}};
        env->Reset(config, state, "old", frames);
        const auto old = env->Observe();
        auto invalidFrames = frames;
        invalidFrames.back().services.front().completedAt = config.end + 1;
        const auto rejected = Rejected([&] {
            env->Reset(config, state, "invalid", invalidFrames);
        });
        NS_TEST_ASSERT_MSG_EQ(rejected, true,
                              "Invalid future completion rejected before replacement");
        NS_TEST_ASSERT_MSG_EQ(env->Observe() == old, true,
                              "Invalid trajectory preserves old state");
        env->Reset(config, state, "fresh", frames);
        for (uint64_t epoch = 0; epoch < 2; ++epoch)
        {
            const auto observation = env->Observe();
            env->StartCycle({observation.reference, {{0, 9}}}, epoch == 0);
            NS_TEST_ASSERT_MSG_EQ(Rejected([&] { env->Reset(config, state, "busy"); }),
                                  true, "Reset cannot race an owned cycle");
            Simulator::Run();
            const auto result = env->Result();
            NS_TEST_ASSERT_MSG_EQ(result.services.front().bytes, 0, "Budget never carries");
            NS_TEST_ASSERT_MSG_EQ(result.next.reference.epoch, epoch + 1, "One advance per plan");
            NS_TEST_ASSERT_MSG_EQ(result.next.state.queues.front().entries.front().entered, 0,
                                  "Unserved HOL does not reset");
        }
        NS_TEST_ASSERT_MSG_EQ(env->Observe().remaining, 0, "Full final observation");
        NS_TEST_ASSERT_MSG_EQ(old.reference.epoch, 0, "Old value snapshot unchanged");
        NS_TEST_ASSERT_MSG_EQ(Simulator::Now().GetNanoSeconds(), 150000000, "Exact final clock");
        env->Dispose();
        Simulator::Destroy();
    }
    bool m_numeric;
};

class ControlledSuite : public TestSuite
{
  public:
    ControlledSuite()
        : TestSuite("fanet-controlled-environment", Type::UNIT)
    {
        AddTestCase(new BoundaryCase(true), TestCase::Duration::QUICK);
        AddTestCase(new BoundaryCase(false), TestCase::Duration::QUICK);
        for (const auto& name : {"value snapshot and zero identity",
                                 "fresh reset and stale reference",
                                 "uninitialized and disposed", "duplicate node",
                                 "duplicate packet location", "missing empty container",
                                 "FIFO order",
                                 "capacity", "overdue wait", "future birth", "invalid route",
                                 "unlocated packet", "time overflow", "unsupported mechanism"})
        {
            AddTestCase(new InitialCase(name), TestCase::Duration::QUICK);
        }
        for (const auto& name : {"boundary registered first", "boundary registered last",
                                 "short budget", "disconnected backlog", "expiry before recovery",
                                 "no same-cycle second hop", "exact constant rate",
                                 "invalid action atomic", "duplicate internal completion",
                                 "dispose cancels only owned events"})
        {
            AddTestCase(new CycleCase(name), TestCase::Duration::QUICK);
        }
    }
} g_controlledSuite;
} // namespace
