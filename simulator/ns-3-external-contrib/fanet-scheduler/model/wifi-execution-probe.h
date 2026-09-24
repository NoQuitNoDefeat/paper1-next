#ifndef FANET_WIFI_EXECUTION_PROBE_H
#define FANET_WIFI_EXECUTION_PROBE_H

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ns3
{

/** A complete component window; endpoints are stable local IDs in [1, nodeCount]. */
struct ProbeWindow
{
    int64_t startNs{5000000}; //!< Absolute grant time.
    int64_t endNs{7000000};   //!< Absolute reservation boundary.
    std::vector<std::pair<uint32_t, uint32_t>> links{{1, 2}}; //!< Ordered directed pairs.
};

/**
 * @brief Bounded development configuration, never a formal experiment default.
 *
 * The radio profile is deliberately fixed: 802.11a, channel 36/20 MHz, OFDM 6 Mbps DATA,
 * constant-rate control, 16 dBm, 7 dB noise figure, NistErrorRateModel, threshold preamble
 * detection (4 dB/-82 dBm), no capture, RTS/fragmentation disabled for the allowed packet
 * sizes, non-QoS, one transmission attempt. Matrix losses and static positions are explicit.
 */
struct WifiProbeConfig
{
    std::string scenario{"single"}; //!< A validated development case name.
    uint32_t nodeCount{2};          //!< Even, in [2, 6].
    uint32_t packetsPerSender{1};   //!< Diagnostic burst in each odd-ID sender's buffer.
    uint32_t payloadBytes{1000};    //!< MSDU payload before LLC/MAC/FCS.
    uint32_t queueMaxPackets{16};   //!< Internal diagnostic queue capacity, not Qij.
    int64_t queueLifetimeNs{10000000000}; //!< Upstream expiry, longer than normal probe.
    int64_t enqueueNs{1000000};     //!< Absolute injection time.
    int64_t stopNs{20000000};       //!< Stop after all normal diagnostic events.
    double desiredLossDb{60};      //!< Loss between (1,2), (3,4), (5,6), both directions.
    double crossLossDb{84};        //!< Every other directed propagation path.
    bool dcf{false};               //!< Unmodified contention baseline.
    bool dropAckAtSenders{false};  //!< Post-reception ACK corruption, diagnostics only.
    bool allowOverrun{false};      //!< Explicit start-only boundary diagnostic.
    uint32_t seed{41};             //!< ns-3 seed, set before random objects exist.
    uint64_t run{1};               //!< ns-3 run number.
    std::vector<ProbeWindow> windows{{}};
};

/** One raw event. Zero identity/size/duration means not applicable; signal/noise may be absent. */
struct WifiProbeEvent
{
    uint64_t sequence{0};           //!< Callback order, including equal timestamps.
    int64_t timeNs{0};              //!< Actual Simulator::Now, integer nanoseconds.
    std::string kind;               //!< Trace or explicit component event.
    uint32_t node{0};               //!< Stable local observer/transmitter ID, zero for global.
    uint32_t peer{0};               //!< Frame Addr1; upper_rx sender; grant receiver; 0 if absent.
    uint64_t packetId{0};           //!< Explicit tag on DATA; ACK has no business packet tag.
    std::string frame{"none"};      //!< data, ack, other, or none.
    uint32_t bytes{0};              //!< Observed bytes at this event's layer.
    int64_t durationNs{0};          //!< Actual TX duration or reservation, depending on kind.
    uint32_t queuePackets{0};       //!< Only meaningful for queue snapshots.
    std::string detail;            //!< Result, mode, or upstream reason.
    std::optional<double> signalDbm; //!< Monitor receive signal, not full-window SINR.
    std::optional<double> noiseDbm; //!< Monitor receive noise/interference sample.
    bool operator==(const WifiProbeEvent&) const = default;
};

/** Completed component run. Contains no CycleResult, rewards, or research service accounting. */
struct WifiProbeResult
{
    WifiProbeConfig config;
    int64_t streamsAssigned{0}; //!< WifiHelper streams beginning at zero.
    std::vector<WifiProbeEvent> events;
};

/**
 * @param name one of the documented development case names
 * @return explicit case configuration, before simulation or result inspection
 * @throws std::invalid_argument for an unknown case
 */
WifiProbeConfig MakeWifiProbeConfig(const std::string& name);

/**
 * @brief Validate the whole input before allocating nodes or scheduling transmissions.
 * @param config bounded component input
 * @throws std::invalid_argument for unsupported, inconsistent or unsafe input
 */
void ValidateWifiProbeConfig(const WifiProbeConfig& config);

/**
 * @brief Execute a fresh component simulation, then destroy owned ns-3 state.
 * @param config validated again at the public boundary
 * @return raw trace with stable local IDs, not Packet::GetUid
 * @throws std::exception on invalid configuration or trace setup
 *
 * Requires exclusive ownership of the process-global Simulator/RNG; never call concurrently
 * or inside another active simulation. No files, shared memory, or Python processes are used.
 */
WifiProbeResult RunWifiExecutionProbe(const WifiProbeConfig& config);

/**
 * @brief Serialize a completed diagnostic report as JSON schema 1.
 * @param out caller-owned stream; stream failures propagate to the caller
 * @param result complete trace and explicit configuration
 */
void WriteWifiProbeJson(std::ostream& out, const WifiProbeResult& result);

} // namespace ns3

#endif // FANET_WIFI_EXECUTION_PROBE_H
