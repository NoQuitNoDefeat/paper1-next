#include "ns3/scheduled-radio.h"
#include "ns3/control-channel.h"
#include "ns3/simulator.h"
#include "ns3/test.h"

#include <algorithm>
#include <set>

namespace ns3::fanet
{
namespace
{
class RadioCase : public TestCase
{
  public:
    RadioCase(const std::string& name,
              unsigned interferers,
              bool weak = false,
              int boundary = 0,
              bool boundaryFirst = true,
              bool empty = false,
              bool exactShannon = false)
        : TestCase(name),
          m_interferers(interferers),
          m_weak(weak),
          m_boundary(boundary),
          m_boundaryFirst(boundaryFirst),
          m_empty(empty),
          m_exactShannon(exactShannon)
    {
    }

  private:
    void DoRun() override
    {
        auto env = CreateObject<ControlledEnvironment>();
        Config config;
        config.nodes = {1, 2, 3, 4, 5, 6};
        config.packetBytes = 1000;
        config.start = 0;
        config.end = 100000000;
        config.period = 50000000;
        State state;
        Frame frame;
        RadioFrame radioFrame;
        radioFrame.noiseW = 1;
        for (Id id : config.nodes)
        {
            radioFrame.powersW[id] = 1;
            for (Id other : config.nodes)
            {
                if (other != id)
                {
                    radioFrame.gains[{id, other}] = 0;
                }
            }
        }
        std::vector<Link> selected;
        for (unsigned n = 0; n <= m_interferers; ++n)
        {
            Link key{2 * n + 1, 2 * n + 2};
            const Id packet = 100 + n;
            config.queues.push_back({key, 3000});
            state.queues.push_back({key, {{packet, 0, 0}}});
            state.packets.push_back({packet, key.first, key.second, 0, std::nullopt});
            frame.services.push_back({key, 50000, config.period, true, 1000000});
            radioFrame.rates[key] = 1000000;
            radioFrame.gains[key] = (m_weak && n == 0) ? 0 : (m_exactShannon ? 3 : 10);
            if (n != 0)
            {
                radioFrame.gains[{key.first, 2}] = 2;
            }
            if (!m_empty)
            {
                selected.push_back(key);
            }
        }
        Frame second = frame;
        second.start = config.period;
        for (auto& service : second.services)
        {
            service.completedAt = config.end;
        }
        auto middle = radioFrame;
        middle.at = config.period;
        auto last = radioFrame;
        last.at = config.end;
        env->Reset(config, state, "radio-reference", std::vector<Frame>{frame, second});
        auto radio = CreateObject<ScheduledRadio>();
        radio->Configure(env, {2400000000., 4000000., 100, 24}, {radioFrame, middle, last});
        // 1024 wire bytes / 1e6 bytes/s + 100ns propagation; independent integer answer.
        const Ns duration = 1024100;
        const Ns offset = m_boundary ? config.period - duration + (m_boundary == 2 ? 1 : 0) : 0;
        radio->StartCycle({env->Observe().reference, selected}, m_boundaryFirst, offset);
        Simulator::Run();
        auto result = env->Result();
        const bool expected =
            !m_empty && !m_weak && !m_exactShannon && m_interferers < 2 && m_boundary != 2;
        NS_TEST_ASSERT_MSG_EQ(result.services[0].bytes,
                              expected ? 1000 : 0,
                              "independent Shannon outcome with accumulated interference");
        NS_TEST_ASSERT_MSG_EQ(result.next.state.queues[0].entries.size(),
                              expected ? 0 : 1,
                              "only a confirmed head is removed");
        const auto& events = radio->Events();
        const auto tx = std::ranges::count(events, std::string("tx_start"), &RadioEvent::kind);
        const auto confirmed =
            std::ranges::count(events, std::string("ideal_confirmation"), &RadioEvent::kind);
        NS_TEST_ASSERT_MSG_EQ(tx,
                              (m_empty || m_boundary == 2) ? 0 : m_interferers + 1,
                              "no unauthorized or out-of-window DATA");
        NS_TEST_ASSERT_MSG_EQ(confirmed,
                              expected ? m_interferers + 1
                                       : ((!m_empty && m_boundary != 2) ? m_interferers : 0),
                              "actual intended receptions alone confirm service");
        for (const auto& event : events)
        {
            if (event.kind == "tx_start")
            {
                NS_TEST_ASSERT_MSG_EQ(event.at, offset, "concurrent DATA intervals start together");
                NS_TEST_ASSERT_MSG_EQ(event.frameBytes, 1024, "header consumes wire time");
            }
            if (event.kind == "rx_ok" || event.kind == "rx_error")
            {
                NS_TEST_ASSERT_MSG_EQ(event.at, offset + duration, "real PHY reception time");
            }
        }
        NS_TEST_ASSERT_MSG_EQ(Simulator::Now().GetNanoSeconds(),
                              config.period,
                              "settle at exact cycle boundary");
        // Empty next plan: no leftover radio event may confirm a packet in another cycle.
        radio->StartCycle({env->Observe().reference, {}});
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(radio->Events().size(), 0, "no old radio event after boundary");
        NS_TEST_ASSERT_MSG_EQ(env->Result().services[0].bytes, 0, "no second removal");
        radio->Dispose();
        env->Dispose();
        Simulator::Destroy();
    }

    unsigned m_interferers;
    bool m_weak;
    int m_boundary;
    bool m_boundaryFirst;
    bool m_empty;
    bool m_exactShannon;
};

class AckCase : public TestCase
{
  public:
    AckCase(bool loss, bool boundaryFirst)
        : TestCase(std::string("actual-ack-") + (loss ? "loss-retry" : "exact-boundary") +
                   (boundaryFirst ? "-close-first" : "-close-last")),
          m_loss(loss),
          m_boundaryFirst(boundaryFirst)
    {
    }

  private:
    void DoRun() override
    {
        constexpr Ns transaction = 1072200; // 1024B DATA + 32B ACK + 16us + 2*100ns.
        Config config;
        config.nodes = {1, 2};
        config.queues = {{{1, 2}, 3000}};
        config.packetBytes = 1000;
        config.period = m_loss ? 50000000 : transaction;
        config.end = 2 * config.period;
        State state;
        state.packets = {{100, 1, 2, 0, std::nullopt}, {101, 1, 2, 0, std::nullopt}};
        state.queues = {{{1, 2}, {{100, 0, 0}, {101, 0, 0}}}};
        Frame first;
        first.services = {{{1, 2}, static_cast<Bytes>(config.period / 1000), config.period,
                           true, 1000000}};
        Frame second = first;
        second.start = config.period;
        second.services[0].completedAt = config.end;
        RadioFrame initial;
        initial.noiseW = 1;
        initial.powersW = {{1, 1}, {2, 1}};
        initial.gains = {{{1, 2}, 10}, {{2, 1}, m_loss ? 0.0 : 10.0}};
        initial.rates = {{{1, 2}, 1000000}};
        auto middle = initial;
        middle.at = config.period;
        middle.gains[{2, 1}] = 10;
        auto final = middle;
        final.at = config.end;
        auto env = CreateObject<ControlledEnvironment>();
        env->Reset(config, state, "actual-ack-test", std::vector<Frame>{first, second});
        auto radio = CreateObject<ScheduledRadio>();
        radio->Configure(env, {2400000000., 4000000., 100, 24, true, 32, 1000000, 16000},
                         {initial, middle, final});
        radio->StartCycle({env->Observe().reference, {{1, 2}}}, m_boundaryFirst);
        Simulator::Run();
        const auto firstResult = env->Result();
        NS_TEST_ASSERT_MSG_EQ(firstResult.services[0].bytes, m_loss ? 0 : 1000,
                              "Only actual ACK commits service, including the exact boundary");
        NS_TEST_ASSERT_MSG_EQ(firstResult.next.state.queues[0].entries[0].packet,
                              m_loss ? 100 : 101, "Failed FIFO head blocks followers");
        const auto& events = radio->Events();
        NS_TEST_ASSERT_MSG_EQ(std::ranges::count(events, std::string("rx_ok"), &RadioEvent::kind),
                              1, "DATA reception does not suffice for service");
        for (const auto& event : events)
        {
            if (event.kind == "transaction_commit" || event.kind == "ack_timeout")
            {
                NS_TEST_ASSERT_MSG_EQ(event.at, transaction, "Independent transaction duration");
            }
        }
        radio->StartCycle({env->Observe().reference, {{1, 2}}}, m_boundaryFirst);
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(env->Result().services[0].bytes, m_loss ? 2000 : 1000,
                              "Later cycle retries original ID once, then serves its follower");
        NS_TEST_ASSERT_MSG_EQ(env->Result().next.state.packets.size(), 0,
                              "No extra authoritative receive copy remains");
        radio->Dispose();
        env->Dispose();
        Simulator::Destroy();
    }

    bool m_loss;
    bool m_boundaryFirst;
};

class CollectionCase : public TestCase
{
  public:
    CollectionCase() : TestCase("control-collection-births-remain-pending-before-data-ack")
    {
    }

  private:
    void DoRun() override
    {
        Config config;
        config.nodes = {1, 2};
        config.queues = {{{1, 2}, 4000}};
        config.packetBytes = 1000;
        config.period = config.end = 50000000;
        State state;
        state.packets = {{100, 1, 2, 0, std::nullopt}};
        state.queues = {{{1, 2}, {{100, 0, 0}}}};
        state.routes = {{1, 2, 2}};
        Frame frame;
        frame.services = {{{1, 2}, 50000, config.end, true, 1000000}};
        frame.births = {{101, 1, 2, 10000, std::nullopt}};
        RadioFrame physical;
        physical.noiseW = 1;
        physical.powersW = {{1, 1}, {2, 1}};
        physical.gains = {{{1, 2}, 10}, {{2, 1}, 10}};
        physical.rates = {{{1, 2}, 1000000}};
        auto final = physical;
        final.at = config.end;
        auto env = CreateObject<ControlledEnvironment>();
        env->Reset(config, state, "collection-cycle", std::vector<Frame>{frame});
        auto radio = CreateObject<ScheduledRadio>();
        radio->Configure(env, {2400000000., 4000000., 100, 24, true, 32, 1000000, 16000},
                         {physical, final});
        auto channel = CreateObject<ControlChannel>();
        auto ground = CreateObject<Node>();
        channel->Configure(ground, {{1, env->GetNode(1)}, {2, env->GetNode(2)}},
                           {1000000, {{1, 100}, {2, 100}}});
        const auto sample = env->BeginExternalCollection();
        unsigned reports = 0;
        for (Id id : config.nodes)
        {
            channel->Send({false, id, 0, 0, 0, wire::List{uint64_t{42}}}, 0,
                [&](const ControlDelivery&) {
                    if (++reports == 2)
                    {
                        Simulator::Stop();
                    }
                });
        }
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(Simulator::Now().GetNanoSeconds(), 46100,
                              "Actual report collection advances simulation time");
        NS_TEST_ASSERT_MSG_EQ(env->ExternalSample(sample.reference).state.packets.size(), 1,
                              "Immutable causal sample excludes births during collection");
        Action plan{sample.reference, {{1, 2}}};
        env->AcceptExternalPlan(plan);
        unsigned commands = 0;
        for (Id id : config.nodes)
        {
            channel->Send({true, id, 0, 0, 0,
                           wire::List{wire::List{uint64_t{1}, uint64_t{2}}}}, 46100,
                [&](const ControlDelivery&) {
                    if (++commands == 2)
                    {
                        Simulator::Stop();
                    }
                });
        }
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(Simulator::Now().GetNanoSeconds(), 106200,
                              "Command bytes occupy the independent control pipe");
        radio->StartPreparedCycle(plan);
        Simulator::Run();
        const auto result = env->Result();
        NS_TEST_ASSERT_MSG_EQ(result.services[0].bytes, 1000, "Only the original head is served");
        NS_TEST_ASSERT_MSG_EQ(result.next.state.queues[0].entries[0].packet, 101,
                              "Born during control, admitted only at the cycle boundary");
        for (const auto& event : result.events)
        {
            if (event.kind == "birth")
            {
                NS_TEST_ASSERT_MSG_EQ(event.at, 10000, "Original birth time is not backfilled");
            }
            if (event.kind == "delivery")
            {
                NS_TEST_ASSERT_MSG_EQ(event.at, 1178400, "Control plus actual DATA/ACK time");
            }
        }
        NS_TEST_ASSERT_MSG_EQ(channel->Outstanding(), 0, "All control frames drained");
        channel->Dispose();
        ground->Dispose();
        radio->Dispose();
        env->Dispose();
        Simulator::Destroy();
    }
};

class RadioSuite : public TestSuite
{
  public:
    RadioSuite()
        : TestSuite("fanet-scheduled-radio", Type::UNIT)
    {
        AddTestCase(new RadioCase("single-link-and-next-cycle", 0));
        AddTestCase(new RadioCase("failed-reception-retains-head", 0, true));
        AddTestCase(new RadioCase("two-concurrent-links", 1));
        AddTestCase(new RadioCase("two-interferers-cause-victim-failure", 2));
        AddTestCase(new RadioCase("boundary-close-registered-first", 0, false, 1, true));
        AddTestCase(new RadioCase("boundary-close-registered-last", 0, false, 1, false));
        AddTestCase(new RadioCase("one-nanosecond-short-does-not-transmit", 0, false, 2));
        AddTestCase(new RadioCase("empty-plan-with-backlog", 0, false, 0, true, true));
        // B*log2(1+3) == 8Mbps: upstream requires strictly more decodable bytes.
        // The planning >= SINR predicate passes at 3; do not hide the PHY discrepancy.
        AddTestCase(
            new RadioCase("exact-shannon-capacity-is-not-success", 0, false, 0, true, false, true));
        for (bool loss : {false, true})
        {
            for (bool boundaryFirst : {false, true})
            {
                AddTestCase(new AckCase(loss, boundaryFirst));
            }
        }
        AddTestCase(new CollectionCase());
    }
};

RadioSuite suite;
} // namespace
} // namespace ns3::fanet
