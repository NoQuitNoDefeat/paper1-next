#include "ns3/moving-channel.h"
#include "ns3/control-channel.h"
#include "ns3/scheduled-radio.h"
#include "ns3/simulator.h"
#include "ns3/test.h"

#include <cmath>
#include <stdexcept>

namespace ns3::fanet
{
namespace
{
class MotionCase : public TestCase
{
  public:
    MotionCase() : TestCase("reflecting-motion-causal-knots-and-gain-ratio")
    {
    }

  private:
    void DoRun() override
    {
        MotionSettings settings{{0, 0, 0}, {10, 10, 10}, 1, 2, 3,
            {{{1, {{1, 5, 5}, {-4, 0, 0}}}, {2, {{9, 5, 5}, {30, 0, 0}}}},
             {{1, {{3, 5, 5}, {4, 0, 0}}}, {2, {{1, 5, 5}, {-30, 0, 0}}}}}};
        auto trace = std::make_shared<const MotionTrace>(settings, 0, 1000000000,
                                                        std::vector<Id>{1, 2});
        auto mobility = CreateObject<TraceMobility>();
        mobility->Configure(trace, 1);
        auto copy = mobility->Copy();
        auto at = trace->At(1, 250000000);
        NS_TEST_ASSERT_MSG_EQ(at.position.x, 0, "Exact low boundary without teleporting");
        NS_TEST_ASSERT_MSG_EQ(at.velocity.x, 4, "Velocity points inward at reflection");
        NS_TEST_ASSERT_MSG_EQ(trace->At(2, 1000000000).position.x, 1,
                              "Multiple crossings preserve the supplied final knot");
        NS_TEST_ASSERT_MSG_EQ_TOL(trace->GainRatio({1, 2}, 250000000), 256.0 / 49, 1e-12,
                                  "Independent hand-computed ratio for 8m to 3.5m");
        bool rejected = false;
        auto broken = settings;
        broken.boundaries[1][1].position.x += 0.1;
        try
        {
            MotionTrace invalid(broken, 0, 1000000000, {1, 2});
        }
        catch (const std::invalid_argument&)
        {
            rejected = true;
        }
        NS_TEST_ASSERT_MSG_EQ(rejected, true, "Boundary teleport must be rejected");
        Vector sampled;
        Simulator::Schedule(MilliSeconds(750), [&] { sampled = copy->GetPosition(); });
        Simulator::Run();
        NS_TEST_ASSERT_MSG_EQ(sampled.x, 2, "Copied mobility still follows actual simulator time");
        rejected = false;
        try
        {
            mobility->SetPosition({5, 5, 5});
        }
        catch (const std::logic_error&)
        {
            rejected = true;
        }
        NS_TEST_ASSERT_MSG_EQ(rejected, true, "Immutable trajectory cannot be silently moved");
        copy->Dispose();
        mobility->Dispose();
        Simulator::Destroy();
    }
};

class MovingAckCase : public TestCase
{
  public:
    explicit MovingAckCase(bool fast, bool control = false)
        : TestCase(control ? "collected-plan-loses-link-before-common-execution"
                    : fast ? "frame-held-data-but-moving-reverse-ack-fails"
                           : "data-and-ack-use-distinct-current-channel-samples"),
          m_fast(fast), m_control(control)
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
        RadioFrame physical;
        physical.noiseW = 1;
        physical.powersW = {{1, 1}, {2, 1}};
        physical.gains = {{{1, 2}, 10}, {{2, 1}, 10}};
        physical.rates = {{{1, 2}, 1000000}};
        auto final = physical;
        final.at = config.end;
        const double speed = m_fast ? 1000 : 1;
        const double finalDistance = 1 + speed * .05;
        final.gains = {{{1, 2}, 10 / (finalDistance * finalDistance)},
                       {{2, 1}, 10 / (finalDistance * finalDistance)}};
        MotionSettings motion{{-1, -1, -1}, {100, 1, 1}, 1, 2, 3,
            {{{1, {{0, 0, 0}, {0, 0, 0}}}, {2, {{1, 0, 0}, {speed, 0, 0}}}},
             {{1, {{0, 0, 0}, {0, 0, 0}}}, {2, {{1 + speed * .05, 0, 0}, {speed, 0, 0}}}}}};
        auto env = CreateObject<ControlledEnvironment>();
        env->Reset(config, state, "moving-ack", std::vector<Frame>{frame});
        auto radio = CreateObject<ScheduledRadio>();
        radio->Configure(env, {2400000000., 4000000., 100, 24, true, 32, 1000000, 16000},
                         {physical, final}, motion);
        const auto sample = env->Observe();
        if (m_control)
        {
            auto control = CreateObject<ControlChannel>();
            auto ground = CreateObject<Node>();
            control->Configure(ground, {{1, env->GetNode(1)}, {2, env->GetNode(2)}},
                               {1000000, {{1, 100}, {2, 100}}});
            env->BeginExternalCollection();
            unsigned collected = 0;
            for (Id id : config.nodes)
            {
                control->Send({false, id, 0, 0, 0, wire::List{uint64_t{42}}}, 0,
                    [&](const ControlDelivery&) {
                        if (++collected == 2)
                        {
                            Simulator::Stop();
                        }
                    });
            }
            Simulator::Run();
            const Action plan{sample.reference, {{1, 2}}};
            env->AcceptExternalPlan(plan);
            std::optional<ControlExecution> execution;
            control->ScheduleCommandWindow(plan, 46100, 1000000, config.end,
                [&] { return radio->ExecutionLinksAt(Simulator::Now().GetNanoSeconds()); },
                [&](const ControlExecution& decision) {
                    execution = decision;
                    radio->StartPreparedCycle(decision.execution);
                });
            Simulator::Run();
            const auto result = env->Result();
            NS_TEST_ASSERT_MSG_EQ(execution->reasons.at({1, 2}), "physical_invalid",
                                  "Complete delivered commands do not rescue an aged link");
            NS_TEST_ASSERT_MSG_EQ(execution->commandReceipts.size(), 2,
                                  "Both actual endpoint deliveries precede the common time");
            NS_TEST_ASSERT_MSG_EQ(result.action.links.size(), 1, "Original plan remains auditable");
            NS_TEST_ASSERT_MSG_EQ(result.services[0].bytes, 0, "No service after physical removal");
            NS_TEST_ASSERT_MSG_EQ(radio->Events().size(), 0, "No actual DATA after physical removal");
            NS_TEST_ASSERT_MSG_EQ(result.next.state.queues[0].entries[0].packet, 100,
                                  "Physical removal preserves backlog");
            control->Dispose();
            ground->Dispose();
            radio->Dispose();
            env->Dispose();
            Simulator::Destroy();
            return;
        }
        radio->StartCycle({sample.reference, {{1, 2}}});
        Simulator::Run();
        const auto result = env->Result();
        const auto samples = radio->ReceptionSamples();
        NS_TEST_ASSERT_MSG_EQ(samples.size(), 2, "Both actual frame starts recorded");
        NS_TEST_ASSERT_MSG_EQ(samples[0].at, 100, "DATA samples at Rx start, not Tx start");
        NS_TEST_ASSERT_MSG_EQ(samples[1].at, 1040200, "ACK recomputes after DATA and turnaround");
        const double firstDistance = 1 + speed * 1e-7;
        const double ackDistance = 1 + speed * .0010402;
        NS_TEST_ASSERT_MSG_EQ_TOL(samples[0].gain, 10 / (firstDistance * firstDistance), 1e-12,
                                  "Independent reception-time DATA gain");
        NS_TEST_ASSERT_MSG_EQ_TOL(samples[1].gain, 10 / (ackDistance * ackDistance), 1e-12,
                                  "Independent reception-time reverse ACK gain");
        NS_TEST_ASSERT_MSG_EQ_TOL(samples[1].senderPosition.x, ackDistance, 1e-12,
                                  "Real sender mobility advanced during DATA");
        NS_TEST_ASSERT_MSG_EQ(result.services[0].bytes, m_fast ? 0 : 1000,
                              "ACK reception determines the authoritative service");
        bool dataOk = false, ackFailed = false;
        for (const auto& event : radio->Events())
        {
            dataOk = dataOk || event.kind == "rx_ok";
            ackFailed = ackFailed || event.kind == "ack_rx_error";
        }
        NS_TEST_ASSERT_MSG_EQ(dataOk, true, "DATA keeps its own start-sampled power within frame");
        NS_TEST_ASSERT_MSG_EQ(ackFailed, m_fast, "ACK cannot reuse earlier DATA propagation");
        if (m_fast)
        {
            NS_TEST_ASSERT_MSG_EQ(result.next.state.queues[0].entries[0].packet, 100,
                                  "Failed ACK retains the original packet identity");
            NS_TEST_ASSERT_MSG_EQ(result.next.state.queues[0].entries[0].entered, 0,
                                  "Failed ACK retains the original HOL start");
        }
        radio->Dispose();
        env->Dispose();
        Simulator::Destroy();
    }

    bool m_fast;
    bool m_control;
};

class MotionSuite : public TestSuite
{
  public:
    MotionSuite() : TestSuite("fanet-moving-channel", Type::UNIT)
    {
        AddTestCase(new MotionCase());
        AddTestCase(new MovingAckCase(false));
        AddTestCase(new MovingAckCase(true));
        AddTestCase(new MovingAckCase(true, true));
    }
};
MotionSuite suite;
} // namespace
} // namespace ns3::fanet
