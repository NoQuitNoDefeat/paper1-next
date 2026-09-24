#include "ns3/control-channel.h"
#include "ns3/simulator.h"
#include "ns3/test.h"

#include <stdexcept>

namespace ns3::fanet
{
namespace
{
class ControlCase : public TestCase
{
  public:
    explicit ControlCase(bool dispose)
        : TestCase(dispose ? "cancel-owned-control-events" : "serialized-control-bytes-and-order"),
          m_dispose(dispose)
    {
    }

  private:
    void DoRun() override
    {
        auto channel = CreateObject<ControlChannel>();
        auto ground = CreateObject<Node>();
        std::map<Id, Ptr<Node>> nodes{{17, CreateObject<Node>()}, {81, CreateObject<Node>()}};
        channel->Configure(ground, nodes, {1000000, {{17, 100}, {81, 1000}}});
        std::vector<ControlDelivery> received;
        auto callback = [&](const ControlDelivery& frame) { received.push_back(frame); };
        ControlMessage message{false, 17, 5, 0, 0, wire::List{uint64_t{42}}};
        NS_TEST_ASSERT_MSG_EQ(channel->Send(message, 0, callback), 46100,
                              "32-byte header plus canonical 14-byte payload and propagation");
        message.sequence = 1;
        NS_TEST_ASSERT_MSG_EQ(channel->Send(message, 0, callback), 92100,
                              "Second frame queues behind the first on the same pipe");
        bool rejected = false;
        try
        {
            channel->Send(message, 0, callback);
        }
        catch (const std::invalid_argument&)
        {
            rejected = true;
        }
        NS_TEST_ASSERT_MSG_EQ(rejected, true, "Duplicate sequence rejected before scheduling");
        message.downlink = true;
        message.sequence = 0;
        NS_TEST_ASSERT_MSG_EQ(channel->Send(message, 0, callback), 46100,
                              "Independent reverse control pipe, not data-channel contention");
        message.downlink = false;
        message.node = 81;
        NS_TEST_ASSERT_MSG_EQ(channel->Send(message, 0, callback), 47000,
                              "Node-specific finite propagation is explicit");
        NS_TEST_ASSERT_MSG_EQ(channel->Outstanding(), 4, "Only accepted sends own events");
        if (m_dispose)
        {
            channel->Dispose();
        }
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(received.size(), m_dispose ? 0 : 4, "Owned cancellation or delivery");
        NS_TEST_ASSERT_MSG_EQ(channel->Outstanding(), 0, "Every frame drained exactly once");
        for (const auto& frame : received)
        {
            NS_TEST_ASSERT_MSG_EQ(frame.message.payload.Array()[0].U(), 42,
                                  "Payload reconstructed from actual received packet bytes");
            NS_TEST_ASSERT_MSG_EQ(frame.message.epoch, 5, "Epoch survives header serialization");
            NS_TEST_ASSERT_MSG_EQ(frame.payloadBytes, 14, "Application bytes measured");
            NS_TEST_ASSERT_MSG_EQ(frame.wireBytes, 46, "Control header is charged on the wire");
            NS_TEST_ASSERT_MSG_EQ(frame.endedAt - frame.startedAt, 46000,
                                  "Serialization time is distinct from propagation");
        }
        if (!m_dispose)
        {
            NS_TEST_ASSERT_MSG_EQ(received.back().message.sequence, 1,
                                  "Queued later frame cannot overtake its predecessor");
            NS_TEST_ASSERT_MSG_EQ(received.back().receivedAt, 92100, "No extra sampled delay");
        }
        channel->Dispose();
        ground->Dispose();
        for (const auto& [id, node] : nodes)
        {
            node->Dispose();
        }
        Simulator::Destroy();
    }

    bool m_dispose;
};

class WindowCase : public TestCase
{
  public:
    explicit WindowCase(bool globalMiss)
        : TestCase(globalMiss ? "global-miss-precedes-all-endpoint-reasons"
                              : "common-window-exact-arrival-and-exclusive-reasons"),
          m_globalMiss(globalMiss)
    {
    }

  private:
    void DoRun() override
    {
        auto channel = CreateObject<ControlChannel>();
        auto ground = CreateObject<Node>();
        std::map<Id, Ptr<Node>> nodes;
        ControlSettings settings{1000000, {}};
        for (Id id = 1; id <= 6; ++id)
        {
            nodes[id] = CreateObject<Node>();
            settings.propagation[id] = id == 6 ? 1 : 0;
        }
        channel->Configure(ground, nodes, settings);
        const Action plan{{"control-window", 0, 0}, {{1, 2}, {3, 4}, {5, 6}}};
        const Ns target = m_globalMiss ? 200001 : 200000;
        const Ns deadline = m_globalMiss ? 200000 : 1000000;
        std::vector<ControlExecution> decisions;
        std::vector<ControlDelivery> receipts;
        unsigned physicalCalls = 0;
        Ns physicalAt = -1;
        channel->ScheduleCommandWindow(plan, 0, target, deadline,
            [&] {
                ++physicalCalls;
                physicalAt = Simulator::Now().GetNanoSeconds();
                return std::set<Link>{{1, 2}};
            },
            [&](const ControlExecution& result) { decisions.push_back(result); },
            [&](const ControlDelivery& frame) { receipts.push_back(frame); });
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(decisions.size(), 1, "Exactly one common execution event");
        const auto result = decisions.at(0);
        NS_TEST_ASSERT_MSG_EQ(result.evaluatedAt, 200000, "Never shift the common DATA event");
        NS_TEST_ASSERT_MSG_EQ(result.reasons.size(), 3, "One exclusive reason per planned link");
        NS_TEST_ASSERT_MSG_EQ(result.execution.links.size(), m_globalMiss ? 0 : 1,
                              "Only retained original links obtain execution");
        NS_TEST_ASSERT_MSG_EQ(physicalCalls, m_globalMiss ? 0 : 1,
                              "Global miss does not inspect future physical state");
        NS_TEST_ASSERT_MSG_EQ(physicalAt, m_globalMiss ? -1 : target,
                              "Physical state is read at common execution, not planning");
        NS_TEST_ASSERT_MSG_EQ(result.reasons.at({1, 2}), m_globalMiss ? "global_miss" : "retained",
                              "Exactly-on-window command arrivals are accepted");
        NS_TEST_ASSERT_MSG_EQ(result.reasons.at({3, 4}),
                              m_globalMiss ? "global_miss" : "physical_invalid",
                              "Physical invalidity only after complete command delivery");
        NS_TEST_ASSERT_MSG_EQ(result.reasons.at({5, 6}),
                              m_globalMiss ? "global_miss" : "command_miss",
                              "Command miss takes precedence over simultaneous physical failure");
        NS_TEST_ASSERT_MSG_EQ(receipts.size(), 6, "Late command is delivered and logged");
        NS_TEST_ASSERT_MSG_EQ(receipts.back().receivedAt, 200001,
                              "A one-nanosecond late receipt cannot authorize another DATA event");
        for (const auto& receipt : receipts)
        {
            // Canonical four-field payload is 168 bytes; the versioned control header adds 32.
            NS_TEST_ASSERT_MSG_EQ(receipt.wireBytes, 200, "Independent serialized command size");
        }
        channel->Dispose();
        ground->Dispose();
        for (const auto& [id, node] : nodes)
        {
            node->Dispose();
        }
        Simulator::Destroy();
    }

    bool m_globalMiss;
};

class WindowGuardCase : public TestCase
{
  public:
    WindowGuardCase() : TestCase("invalid-window-leaves-no-partial-control-events")
    {
    }

  private:
    void DoRun() override
    {
        auto channel = CreateObject<ControlChannel>();
        auto ground = CreateObject<Node>();
        std::map<Id, Ptr<Node>> nodes{{1, CreateObject<Node>()}, {2, CreateObject<Node>()}};
        channel->Configure(ground, nodes, {1000000, {{1, 0}, {2, INT64_MAX}}});
        unsigned decisions = 0;
        unsigned receipts = 0;
        auto physical = [] { return std::set<Link>{}; };
        auto execute = [&](const ControlExecution&) { ++decisions; };
        auto receipt = [&](const ControlDelivery&) { ++receipts; };
        bool rejected = false;
        try
        {
            channel->ScheduleCommandWindow({{"guard", 0, 0}, {{1, 2}}}, 0, 10, 20,
                                           physical, execute, receipt);
        }
        catch (const std::invalid_argument&)
        {
            rejected = true;
        }
        NS_TEST_ASSERT_MSG_EQ(rejected, true, "Second endpoint arrival overflows");
        NS_TEST_ASSERT_MSG_EQ(channel->Outstanding(), 0,
                              "No first endpoint send remains after second endpoint rejection");
        rejected = false;
        try
        {
            channel->ScheduleCommandWindow({{"guard", 0, -1}, {}}, 0, 10, 20,
                                           physical, execute, receipt);
        }
        catch (const std::invalid_argument&)
        {
            rejected = true;
        }
        NS_TEST_ASSERT_MSG_EQ(rejected, true, "Negative sampling time rejected even for empty plan");
        channel->ScheduleCommandWindow({{"guard", 0, 0}, {}}, 0, 10, 20,
                                       physical, execute, receipt);
        rejected = false;
        try
        {
            channel->ScheduleCommandWindow({{"guard", 0, 0}, {}}, 0, 10, 20,
                                           physical, execute, receipt);
        }
        catch (const std::invalid_argument&)
        {
            rejected = true;
        }
        NS_TEST_ASSERT_MSG_EQ(rejected, true, "Empty plan also permits only one execution window");
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(decisions, 1, "Only the valid window executes");
        NS_TEST_ASSERT_MSG_EQ(receipts, 0, "Rejected commands never arrive");
        channel->Dispose();
        ground->Dispose();
        for (const auto& [id, node] : nodes)
        {
            node->Dispose();
        }
        Simulator::Destroy();
    }
};

class ControlSuite : public TestSuite
{
  public:
    ControlSuite() : TestSuite("fanet-control-channel", Type::UNIT)
    {
        AddTestCase(new ControlCase(false));
        AddTestCase(new ControlCase(true));
        AddTestCase(new WindowCase(false));
        AddTestCase(new WindowCase(true));
        AddTestCase(new WindowGuardCase());
    }
};
ControlSuite suite;
} // namespace
} // namespace ns3::fanet
