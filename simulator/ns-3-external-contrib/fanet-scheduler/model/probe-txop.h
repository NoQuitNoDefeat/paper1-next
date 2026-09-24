#ifndef FANET_PROBE_TXOP_H
#define FANET_PROBE_TXOP_H

#include "ns3/txop.h"

namespace ns3
{

/**
 * @brief Non-QoS, single-link component probe with explicit one-MPDU grants.
 *
 * Queueing and release never request DCF access. A grant invokes the existing frame exchange
 * manager, retaining its DATA/ACK processing. This is not a research queue or a complete MAC.
 * Only the fixed, unfragmented 802.11a/ConstantRate configuration in RunWifiExecutionProbe is
 * supported. Sleep, switching, QoS, aggregation and autonomous retries are outside this probe.
 */
class ProbeTxop : public Txop
{
  public:
    /** @return the registered object type */
    static TypeId GetTypeId();

    /** Outcomes of one explicit attempt; none denotes research service or terminal status. */
    enum class GrantResult
    {
        STARTED,
        EMPTY,
        BUSY,
        NO_TIME
    };

    /**
     * @brief Enqueue in the diagnostic MAC buffer without requesting channel access.
     * @param mpdu reference-counted MPDU; the queue retains it if accepted
     */
    void Queue(Ptr<WifiMpdu> mpdu) override;

    /**
     * @brief Release access without scheduling the next MPDU.
     * @param linkId Wi-Fi link ID; only zero is supported
     */
    void NotifyChannelReleased(uint8_t linkId) override;

    /**
     * @brief Try one MPDU now, with an optional successful DATA/ACK airtime reservation.
     * @param end absolute end of the component window (ns-3 time)
     * @param oneWayDelay bound on one-way propagation time for the selected pair
     * @param allowOverrun explicitly bypass reservation for boundary diagnostics only
     * @return attempt outcome; EMPTY may trigger the upstream lazy expiry trace
     *
     * Does not abort a frame in flight or promise that an ACK timeout has settled by end.
     * The caller validates the complete plan and peer before granting any transmitter.
     */
    GrantResult Grant(Time end, Time oneWayDelay, bool allowOverrun);

    /**
     * @brief Predict airtime for a successful, unfragmented DATA/SIFS/ACK transaction.
     * @param oneWayDelay bound on each propagation leg
     * @return duration, or zero for an empty queue; no queue mutation
     */
    Time GetTransactionDuration(Time oneWayDelay) const;

  private:
    /** Reject use before attachment or after Dispose, instead of dereferencing stale state. */
    void ValidateContext() const;
    TracedCallback<Ptr<const WifiMpdu>> m_enqueueRejected; //!< Actual Enqueue(false) outcome.
};

} // namespace ns3

#endif // FANET_PROBE_TXOP_H
