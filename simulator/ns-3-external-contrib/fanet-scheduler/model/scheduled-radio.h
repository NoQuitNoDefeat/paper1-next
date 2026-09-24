#ifndef FANET_SCHEDULED_RADIO_H
#define FANET_SCHEDULED_RADIO_H

#include "controlled-environment.h"
#include "moving-channel.h"

#include <memory>

namespace ns3::fanet
{
/** Explicit planned PHY settings; neither profile is an IEEE 802.11 MAC. */
struct RadioSettings
{
    double centerHz{};
    double bandwidthHz{};
    Ns propagationNs{};
    Bytes overheadBytes{}; //!< Includes the 24-byte identity header; remaining bytes are padding.
    bool actualAck{false}; //!< Versioned centralized transaction, committed on actual ACK reception.
    Bytes ackBytes{};      //!< Whole ACK frame including its 24-byte identity header.
    Bytes ackRate{};       //!< Nominal ACK bytes per second, independent of DATA goodput.
    Ns turnaroundNs{};    //!< DATA receive completion to ACK transmission start.
};

/** Frozen private execution channel, independent from policy-side estimated channel inputs. */
struct RadioFrame
{
    Ns at{};
    double noiseW{};
    std::map<Id, double> powersW;
    std::map<Link, double> gains; //!< Every directed pair of distinct nodes, including zero gains.
    std::map<Link, Bytes> rates;  //!< Application-selected nominal PHY byte rates, not goodput.
};

/** Packet-related event captured from the actual Spectrum/PHY execution path. */
struct RadioEvent
{
    std::string kind;
    Id packet{};
    Link link;
    Ns at{};
    Bytes payloadBytes{};
    Bytes frameBytes{};
    bool operator==(const RadioEvent&) const = default;
};

/** Actual intended reception-start propagation sample, distinct from end-of-frame success. */
struct ReceptionSample
{
    Id packet{};
    Link link; //!< Actual direction; reverse for an ACK.
    Ns at{};
    double gain{};
    double receivedPowerW{};
    Vector senderPosition;
    Vector receiverPosition;
};

/** Private channel facts sampled at the actual grant event, not inferred from PHY success. */
struct ExecutionChannel
{
    Ns at{};
    std::vector<Link> links;
    double noiseW{};
    std::map<Id, double> powersW;
    std::map<Link, double> gains; //!< Every selected transmitter to every selected receiver.
};

/**
 * Plan-driven single-radio execution using ns-3 Spectrum and its Shannon error model.
 * The receiver synchronizes only to its scheduled peer; all other signals remain interference.
 * The legacy profile has ideal confirmation; the actual-ACK profile commits the centralized
 * ledger only after the reverse wireless ACK. Failed heads can retry in later cycles only.
 * Configure once per episode. The caller owns Simulator::Run/Destroy; on failure dispose both
 * radio and environment and Destroy before any further Run.
 */
class ScheduledRadio : public Object
{
  public:
    static TypeId GetTypeId();
    ScheduledRadio();
    ~ScheduledRadio() override;
    /**
     * Validate inputs before attaching PHYs. Gains are private execution facts.
     * @param env Initialized, independently owned packet ledger.
     * @param settings Explicit frequency, bandwidth, propagation and frame overhead.
     * @param frames Every boundary including the final boundary, copied on success.
     */
    void Configure(Ptr<ControlledEnvironment> env,
                   const RadioSettings& settings,
                   const std::vector<RadioFrame>& frames,
                   const std::optional<MotionSettings>& motion = std::nullopt);
    /**
     * Begin one complete plan. The caller must run the simulator to its boundary.
     * @param action Complete selected link set for the current observation.
     * @param boundaryFirst Register boundary closure before radio events for ordering tests.
     * @param startOffset Declared diagnostic execution delay in integer nanoseconds.
     */
    void StartCycle(const Action& action, bool boundaryFirst = true, Ns startOffset = 0);
    /** Execute a subset now, after an externally collected cycle has accepted its full plan. */
    void StartPreparedCycle(const Action& execution);
    /** @return Actual events for the latest settled cycle; rejects unfinished execution. */
    const std::vector<RadioEvent>& Events() const;
    /** Sample current physical availability for original reported edges; no policy recomputation. */
    std::set<Link> ExecutionLinksAt(Ns at) const;
    /** @return Intended DATA/ACK reception-start samples for the latest settled moving cycle. */
    const std::vector<ReceptionSample>& ReceptionSamples() const;
    /** @return Complete selected-set channel facts for offline audit of the settled cycle. */
    const ExecutionChannel& ExecutionSnapshot() const;

  protected:
    void DoDispose() override;

  private:
    void Begin(const Action& action, bool boundaryFirst, Ns startOffset, bool prepared);
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
} // namespace ns3::fanet
#endif
