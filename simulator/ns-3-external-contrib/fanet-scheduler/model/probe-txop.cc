#include "probe-txop.h"

#include "ns3/frame-exchange-manager.h"
#include "ns3/simulator.h"
#include "ns3/wifi-mac-queue.h"
#include "ns3/wifi-mac.h"
#include "ns3/wifi-mpdu.h"
#include "ns3/wifi-phy.h"
#include "ns3/wifi-remote-station-manager.h"
#include "ns3/wifi-utils.h"

#include <stdexcept>

namespace ns3
{

NS_OBJECT_ENSURE_REGISTERED(ProbeTxop);

TypeId
ProbeTxop::GetTypeId()
{
    static TypeId tid = TypeId("ns3::ProbeTxop")
                            .SetParent<Txop>()
                            .SetGroupName("Wifi")
                            .AddConstructor<ProbeTxop>()
                            .AddTraceSource("EnqueueRejected",
                                            "Diagnostic MAC queue Enqueue returned false.",
                                            MakeTraceSourceAccessor(&ProbeTxop::m_enqueueRejected),
                                            "ns3::WifiMpdu::TracedCallback");
    return tid;
}

void
ProbeTxop::Queue(Ptr<WifiMpdu> mpdu)
{
    ValidateContext();
    if (!mpdu || !mpdu->GetHeader().IsData() || mpdu->GetHeader().IsQosData() ||
        mpdu->GetHeader().GetAddr1().IsGroup())
    {
        throw std::invalid_argument("Probe queue accepts only unicast non-QoS DATA");
    }
    if (!m_queue->Enqueue(mpdu))
    {
        // The locked queue can return false before its generic DropBeforeEnqueue trace.
        m_enqueueRejected(mpdu);
    }
}

void
ProbeTxop::ValidateContext() const
{
    if (!m_mac || !m_queue || m_mac->GetNLinks() != 1 || !m_mac->GetWifiPhy(0))
    {
        throw std::logic_error("ProbeTxop is not attached to a live single-link MAC");
    }
}

void
ProbeTxop::NotifyChannelReleased(uint8_t linkId)
{
    GetLink(linkId).access = NOT_REQUESTED;
    // Deliberately do not call Txop::NotifyChannelReleased: it schedules RequestAccess.
    m_mac->NotifyChannelReleased(this, linkId);
}

Time
ProbeTxop::GetTransactionDuration(Time oneWayDelay) const
{
    ValidateContext();
    if (oneWayDelay.IsStrictlyNegative())
    {
        throw std::invalid_argument("Negative propagation delay in probe reservation");
    }
    auto mpdu = m_queue->Peek(0);
    if (!mpdu)
    {
        return Time(0);
    }
    auto phy = m_mac->GetWifiPhy(0);
    auto manager = m_mac->GetWifiRemoteStationManager(0);
    auto data = manager->GetDataTxVector(mpdu->GetHeader(), phy->GetChannelWidth());
    auto ack = manager->GetAckTxVector(mpdu->GetHeader().GetAddr1(), data);
    return WifiPhy::CalculateTxDuration(mpdu->GetSize(), data, phy->GetPhyBand()) +
           phy->GetSifs() + WifiPhy::CalculateTxDuration(GetAckSize(), ack, phy->GetPhyBand()) +
           2 * oneWayDelay;
}

ProbeTxop::GrantResult
ProbeTxop::Grant(Time end, Time oneWayDelay, bool allowOverrun)
{
    ValidateContext();
    if (end <= Simulator::Now() || oneWayDelay.IsStrictlyNegative())
    {
        throw std::invalid_argument("Probe grant requires a future end and nonnegative delay");
    }
    auto phy = m_mac->GetWifiPhy(0);
    if (GetAccessStatus(0) != NOT_REQUESTED || phy->IsStateTx() || phy->IsStateRx() ||
        phy->IsStateSwitching() || phy->IsStateSleep() || phy->IsStateOff())
    {
        return GrantResult::BUSY;
    }
    m_queue->WipeAllExpiredMpdus();
    if (!m_queue->Peek(0))
    {
        return GrantResult::EMPTY;
    }
    if (!allowOverrun && GetTransactionDuration(oneWayDelay) > end - Simulator::Now())
    {
        return GrantResult::NO_TIME;
    }
    const bool started =
        m_mac->GetFrameExchangeManager(0)->StartTransmission(this, phy->GetChannelWidth());
    return started ? GrantResult::STARTED : GrantResult::EMPTY;
}

} // namespace ns3
