#include "ns3/probe-txop.h"
#include "ns3/test.h"
#include "ns3/wifi-execution-probe.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

using namespace ns3;

namespace
{

std::vector<WifiProbeEvent>
Select(const WifiProbeResult& result,
       const std::string& kind,
       const std::string& frame = "",
       uint32_t node = 0)
{
    std::vector<WifiProbeEvent> selected;
    for (const auto& event : result.events)
    {
        if (event.kind == kind && (frame.empty() || event.frame == frame) &&
            (node == 0 || event.node == node))
        {
            selected.push_back(event);
        }
    }
    return selected;
}

class ProbeCase : public TestCase
{
  public:
    explicit ProbeCase(const std::string& scenario)
        : TestCase("Wi-Fi execution probe " + scenario),
          m_scenario(scenario)
    {
    }

  private:
    void DoRun() override
    {
        auto config = MakeWifiProbeConfig(m_scenario);
        auto result = RunWifiExecutionProbe(config);
        const auto data = Select(result, "tx_begin", "data");
        const auto ack = Select(result, "tx_begin", "ack");
        const auto confirmed = Select(result, "acked_mpdu");
        const auto delivered = Select(result, "upper_rx");
        const auto final = Select(result, "final_queue", "", 1);
        NS_TEST_ASSERT_MSG_EQ(final.size(), 1, "A complete final snapshot is required");
        if (final.empty())
        {
            return;
        }
        for (std::size_t i = 0; i < result.events.size(); ++i)
        {
            NS_TEST_ASSERT_MSG_EQ(result.events[i].sequence, i, "Stable callback sequence");
            if (i > 0)
            {
                NS_TEST_ASSERT_MSG_EQ(result.events[i].timeNs >= result.events[i - 1].timeNs,
                                      true,
                                      "Monotonic simulation time");
            }
        }

        if (m_scenario == "empty" || m_scenario == "window-short")
        {
            NS_TEST_ASSERT_MSG_EQ(data.size(), 0, "No data without a usable grant");
            NS_TEST_ASSERT_MSG_EQ(ack.size(), 0, "No fabricated ACK");
            NS_TEST_ASSERT_MSG_EQ(final.front().queuePackets, 1, "Unserved buffer retained");
            if (m_scenario == "window-short")
            {
                auto outcomes = Select(result, "grant_result");
                NS_TEST_ASSERT_MSG_EQ(outcomes.size(), 1, "One explicit rejection");
                NS_TEST_ASSERT_MSG_EQ(outcomes.front().detail, "no_time", "Whole transaction fit");
            }
            return;
        }
        if (m_scenario == "expiry")
        {
            NS_TEST_ASSERT_MSG_EQ(data.size(), 0, "Expired diagnostic buffer cannot transmit");
            NS_TEST_ASSERT_MSG_EQ(Select(result, "queue_expired").size(),
                                  1,
                                  "Observe actual expiry");
            NS_TEST_ASSERT_MSG_EQ(final.front().queuePackets, 0, "Upstream expiry removes MPDU");
            return;
        }
        if (m_scenario == "gated")
        {
            NS_TEST_ASSERT_MSG_EQ(data.size(), 2, "Exactly two one-MPDU grants");
            NS_TEST_ASSERT_MSG_EQ(Select(result, "tx_begin", "data", 3).size(),
                                  0,
                                  "Peer not selected");
            NS_TEST_ASSERT_MSG_EQ(final.front().queuePackets, 1, "No autonomous next-packet send");
            auto other = Select(result, "final_queue", "", 3);
            NS_TEST_ASSERT_MSG_EQ(other.front().queuePackets, 3, "Unselected backlog retained");
            if (data.size() == 2)
            {
                NS_TEST_ASSERT_MSG_EQ(data[0].timeNs, 5000000, "First actual grant");
                NS_TEST_ASSERT_MSG_EQ(data[1].timeNs, 10000000, "Reopen only at the second grant");
                NS_TEST_ASSERT_MSG_EQ(data[0].packetId, 1001, "Stable FIFO identity");
                NS_TEST_ASSERT_MSG_EQ(data[1].packetId, 1002, "FIFO next identity");
            }
            return;
        }
        if (m_scenario == "parallel" || m_scenario == "dcf")
        {
            NS_TEST_ASSERT_MSG_EQ(data.size(), 2, "Two actual transmissions");
            NS_TEST_ASSERT_MSG_EQ(confirmed.size(), 2, "Two actual ACK confirmations");
            if (data.size() == 2)
            {
                if (m_scenario == "parallel")
                {
                    NS_TEST_ASSERT_MSG_EQ(data[0].timeNs, 5000000, "First planned start");
                    NS_TEST_ASSERT_MSG_EQ(data[1].timeNs, 5000000, "Actual concurrent start");
                }
                else
                {
                    NS_TEST_ASSERT_MSG_EQ(data[0].timeNs < 5000000, true, "DCF ignores plan time");
                    NS_TEST_ASSERT_MSG_EQ(data[1].timeNs < 5000000,
                                          true,
                                          "Both DCF sends bypass the explicit plan");
                    // An idle-channel simultaneous first burst can start concurrently in DCF.
                    // Do not assume that contention always serializes every input.
                }
            }
            return;
        }
        if (m_scenario.starts_with("interference-"))
        {
            const auto active = static_cast<std::size_t>(m_scenario.back() - '0') + 1;
            NS_TEST_ASSERT_MSG_EQ(data.size(), active, "Only selected interferers transmit");
            for (const auto& event : data)
            {
                NS_TEST_ASSERT_MSG_EQ(event.timeNs, 5000000, "Shared physical start time");
            }
            // Reception counts are observations, not a deterministic SINR threshold oracle.
            return;
        }

        NS_TEST_ASSERT_MSG_EQ(data.size(), 1, "Single DATA attempt");
        NS_TEST_ASSERT_MSG_EQ(ack.size(), 1, "Receiver emits a real MAC ACK");
        NS_TEST_ASSERT_MSG_EQ(delivered.size(), 1, "Receiver upper delivery");
        if (data.size() != 1 || ack.size() != 1 || delivered.size() != 1)
        {
            return;
        }
        NS_TEST_ASSERT_MSG_EQ(data.front().timeNs, 5000000, "No DCF delay");
        NS_TEST_ASSERT_MSG_EQ(data.front().bytes, 1036, "1000 + LLC 8 + MAC 24 + FCS 4");
        NS_TEST_ASSERT_MSG_EQ(data.front().durationNs, 1408000, "Independent OFDM duration");
        NS_TEST_ASSERT_MSG_EQ(ack.front().durationNs, 44000, "Independent ACK duration");
        NS_TEST_ASSERT_MSG_EQ(ack.front().bytes, 14, "ACK includes FCS");
        NS_TEST_ASSERT_MSG_EQ(delivered.front().bytes, 1000, "Only upper payload bytes");
        NS_TEST_ASSERT_MSG_EQ(delivered.front().packetId, 1001, "Identity survives lower copies");
        NS_TEST_ASSERT_MSG_EQ(ack.front().packetId, 0, "ACK has no invented business tag");
        NS_TEST_ASSERT_MSG_EQ(ack.front().timeNs, 6424100, "DATA RX then SIFS before ACK");
        if (m_scenario == "ack-loss")
        {
            NS_TEST_ASSERT_MSG_EQ(confirmed.size(), 0, "DATA delivery does not imply ACK success");
            NS_TEST_ASSERT_MSG_EQ(Select(result, "response_timeout").size(), 1, "Actual timeout");
            NS_TEST_ASSERT_MSG_EQ(Select(result, "mac_drop").size(), 1, "One-attempt retry limit");
            NS_TEST_ASSERT_MSG_EQ(final.front().queuePackets, 0, "Diagnostic MAC drop, not Qij");
            return;
        }
        NS_TEST_ASSERT_MSG_EQ(confirmed.size(), 1, "Sender confirms once");
        if (confirmed.empty())
        {
            return;
        }
        NS_TEST_ASSERT_MSG_EQ(confirmed.front().timeNs, 6468200, "ACK arrives after second delay");
        if (m_scenario == "window-overrun")
        {
            NS_TEST_ASSERT_MSG_EQ(confirmed.front().timeNs > config.windows.front().endNs,
                                  true,
                                  "Start-only gating leaves an ACK across the boundary");
        }
        if (m_scenario == "boundary")
        {
            auto close = Select(result, "window_close", "", 1);
            auto followup = Select(result, "close_followup", "", 1);
            NS_TEST_ASSERT_MSG_EQ(close.front().timeNs, confirmed.front().timeNs, "Same timestamp");
            NS_TEST_ASSERT_MSG_EQ(close.front().queuePackets, 1, "Earlier event ID sees in-flight");
            NS_TEST_ASSERT_MSG_EQ(followup.front().queuePackets, 0, "Later event observes dequeue");
        }
        if (m_scenario == "overflow")
        {
            NS_TEST_ASSERT_MSG_EQ(Select(result, "queue_rejected").size(),
                                  1,
                                  "Internal capacity drop");
        }
    }

    std::string m_scenario;
};

class ProbeInputCase : public TestCase
{
  public:
    ProbeInputCase()
        : TestCase("Wi-Fi probe rejects complete invalid plans before simulation")
    {
    }

  private:
    void DoRun() override
    {
        auto unattached = CreateObject<ProbeTxop>();
        for (bool disposed : {false, true})
        {
            if (disposed)
            {
                unattached->Dispose();
            }
            bool rejected = false;
            try
            {
                unattached->Grant(MilliSeconds(1), NanoSeconds(100), false);
            }
            catch (const std::logic_error&)
            {
                rejected = true;
            }
            NS_TEST_ASSERT_MSG_EQ(rejected, true, "Reject unattached or disposed grant safely");
        }
        auto base = MakeWifiProbeConfig("parallel");
        std::vector<WifiProbeConfig> invalid;
        auto bad = base;
        bad.windows.front().links = {{1, 2}, {1, 4}};
        invalid.push_back(bad);
        bad = base;
        bad.windows.front().links = {{1, 2}, {3, 2}};
        invalid.push_back(bad);
        bad = base;
        bad.windows.front().links = {{1, 5}};
        invalid.push_back(bad);
        bad = base;
        bad.windows.front().endNs = bad.windows.front().startNs;
        invalid.push_back(bad);
        bad = base;
        bad.desiredLossDb = std::numeric_limits<double>::quiet_NaN();
        invalid.push_back(bad);
        bad = base;
        bad.queueMaxPackets = 0;
        invalid.push_back(bad);
        bad = base;
        bad.payloadBytes = 100000;
        invalid.push_back(bad);
        bad = base;
        bad.seed = 0;
        invalid.push_back(bad);
        for (const auto& config : invalid)
        {
            bool rejected = false;
            try
            {
                RunWifiExecutionProbe(config);
            }
            catch (const std::invalid_argument&)
            {
                rejected = true;
            }
            NS_TEST_ASSERT_MSG_EQ(rejected, true, "Input rejected with no partial simulation");
        }
        auto good = RunWifiExecutionProbe(MakeWifiProbeConfig("single"));
        NS_TEST_ASSERT_MSG_EQ(Select(good, "acked_mpdu").size(),
                              1,
                              "Fresh run after invalid input");
    }
};

class ProbeLifecycleCase : public TestCase
{
  public:
    ProbeLifecycleCase()
        : TestCase("Wi-Fi probe repeated lifetime and reverse grant order")
    {
    }

  private:
    void DoRun() override
    {
        auto config = MakeWifiProbeConfig("parallel");
        const auto first = RunWifiExecutionProbe(config);
        const auto second = RunWifiExecutionProbe(config);
        NS_TEST_ASSERT_MSG_EQ((first.events == second.events),
                              true,
                              "Same-environment full replay");
        std::reverse(config.windows.front().links.begin(), config.windows.front().links.end());
        auto reversed = RunWifiExecutionProbe(config);
        for (uint32_t node : {1, 3})
        {
            auto data = Select(reversed, "tx_begin", "data", node);
            auto ack = Select(reversed, "acked_mpdu", "", node);
            NS_TEST_ASSERT_MSG_EQ(data.size(), 1, "Both senders remain active");
            NS_TEST_ASSERT_MSG_EQ(ack.size(), 1, "Reversed registration preserves this outcome");
            if (!data.empty())
            {
                NS_TEST_ASSERT_MSG_EQ(data.front().timeNs, 5000000, "No event-order start skew");
            }
        }
        auto busy = MakeWifiProbeConfig("single");
        busy.allowOverrun = true;
        busy.windows = {{5000000, 5100000, {{1, 2}}}, {5200000, 7000000, {{1, 2}}}};
        auto busyResult = RunWifiExecutionProbe(busy);
        auto outcomes = Select(busyResult, "grant_result");
        NS_TEST_ASSERT_MSG_EQ(outcomes.size(), 2, "Both diagnostic grants reported");
        if (outcomes.size() == 2)
        {
            NS_TEST_ASSERT_MSG_EQ(outcomes[1].detail,
                                  "busy",
                                  "Cannot overwrite in-flight exchange");
        }
        NS_TEST_ASSERT_MSG_EQ(Select(busyResult, "tx_begin", "data").size(),
                              1,
                              "No second transmission while PHY is busy");

        auto noPackets = MakeWifiProbeConfig("single");
        noPackets.packetsPerSender = 0;
        auto emptyResult = RunWifiExecutionProbe(noPackets);
        auto emptyOutcomes = Select(emptyResult, "grant_result");
        NS_TEST_ASSERT_MSG_EQ(emptyOutcomes.front().detail, "empty", "Empty grant is explicit");
        NS_TEST_ASSERT_MSG_EQ(Select(emptyResult, "tx_begin").size(),
                              0,
                              "No fabricated frame");
    }
};

class WifiExecutionProbeSuite : public TestSuite
{
  public:
    WifiExecutionProbeSuite()
        : TestSuite("fanet-wifi-execution-probe", Type::UNIT)
    {
        for (const auto& name : {"single",
                                 "empty",
                                 "gated",
                                 "parallel",
                                 "dcf",
                                 "interference-0",
                                 "interference-1",
                                 "interference-2",
                                 "ack-loss",
                                 "window-short",
                                 "window-overrun",
                                 "boundary",
                                 "expiry",
                                 "overflow"})
        {
            AddTestCase(new ProbeCase(name), TestCase::Duration::QUICK);
        }
        AddTestCase(new ProbeInputCase(), TestCase::Duration::QUICK);
        AddTestCase(new ProbeLifecycleCase(), TestCase::Duration::QUICK);
    }
};

static WifiExecutionProbeSuite g_wifiExecutionProbeSuite;

} // namespace
