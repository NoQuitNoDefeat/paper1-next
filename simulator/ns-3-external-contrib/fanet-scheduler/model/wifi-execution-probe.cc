#include "wifi-execution-probe.h"

#include "probe-txop.h"

#include "ns3/boolean.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/double.h"
#include "ns3/error-model.h"
#include "ns3/mobility-helper.h"
#include "ns3/net-device-container.h"
#include "ns3/node-container.h"
#include "ns3/propagation-delay-model.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/simulator.h"
#include "ns3/single-model-spectrum-channel.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/string.h"
#include "ns3/tag.h"
#include "ns3/uinteger.h"
#include "ns3/wifi-mac-helper.h"
#include "ns3/wifi-mac-queue.h"
#include "ns3/wifi-mac.h"
#include "ns3/wifi-mpdu.h"
#include "ns3/wifi-net-device.h"
#include "ns3/wifi-psdu.h"

#include <cmath>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace ns3
{
namespace
{

// A local diagnostic identity travels with DATA copies; it adds no on-air bytes.
class ProbePacketTag : public Tag
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::ProbePacketTag")
                                .SetParent<Tag>()
                                .AddConstructor<ProbePacketTag>();
        return tid;
    }
    TypeId GetInstanceTypeId() const override { return GetTypeId(); }
    uint32_t GetSerializedSize() const override { return 8; }
    void Serialize(TagBuffer buffer) const override { buffer.WriteU64(m_id); }
    void Deserialize(TagBuffer buffer) override { m_id = buffer.ReadU64(); }
    void Print(std::ostream& out) const override { out << m_id; }
    void Set(uint64_t id) { m_id = id; }
    uint64_t Get() const { return m_id; }

  private:
    uint64_t m_id{0};
};

NS_OBJECT_ENSURE_REGISTERED(ProbePacketTag);

// The post-reception hook receives the MPDU payload WITHOUT its MAC header. In this
// DATA/ACK-only profile the zero-length payload received at a sender belongs to its ACK.
// This selector must not be reused for a profile containing other empty control frames.
class ProbeAckErrorModel : public ErrorModel
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::ProbeAckErrorModel")
                                .SetParent<ErrorModel>()
                                .AddConstructor<ProbeAckErrorModel>();
        return tid;
    }

  private:
    bool DoCorrupt(Ptr<Packet> packet) override
    {
        return packet->GetSize() == 0;
    }
    void DoReset() override {}
};

NS_OBJECT_ENSURE_REGISTERED(ProbeAckErrorModel);

class ProbeMacHelper : public WifiMacHelper
{
  public:
    explicit ProbeMacHelper(bool dcf)
    {
        if (!dcf)
        {
            m_dcf.SetTypeId(ProbeTxop::GetTypeId());
        }
        SetType("ns3::AdhocWifiMac",
                "QosSupported",
                BooleanValue(false),
                "FrameRetryLimit",
                UintegerValue(1));
    }
};

std::string
FrameKind(const WifiMacHeader& header)
{
    return header.IsData() ? "data" : (header.IsAck() ? "ack" : "other");
}

uint64_t
PacketId(Ptr<const Packet> packet)
{
    ProbePacketTag tag;
    return packet->PeekPacketTag(tag) ? tag.Get() : 0;
}

const std::set<std::string> CASES{"single",
                                 "empty",
                                 "gated",
                                 "dcf",
                                 "parallel",
                                 "interference-0",
                                 "interference-1",
                                 "interference-2",
                                 "ack-loss",
                                 "window-short",
                                 "window-overrun",
                                 "boundary",
                                 "expiry",
                                 "overflow"};

// The guard is destroyed before the recorder and its callback targets.
struct SimulatorGuard
{
    ~SimulatorGuard() { Simulator::Destroy(); }
};

class ProbeRun
{
  public:
    explicit ProbeRun(const WifiProbeConfig& config)
        : m_result{config, 0, {}}
    {
    }

    WifiProbeResult Execute()
    {
        SimulatorGuard guard;
        const auto& config = m_result.config;
        RngSeedManager::SetSeed(config.seed);
        RngSeedManager::SetRun(config.run);
        if (Time::GetResolution() != Time::NS)
        {
            throw std::invalid_argument("Probe requires the existing time resolution to be NS");
        }
        m_nodes.Create(config.nodeCount);
        MobilityHelper mobility;
        mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        mobility.Install(m_nodes);
        for (uint32_t i = 0; i < config.nodeCount; ++i)
        {
            m_nodes.Get(i)->GetObject<MobilityModel>()->SetPosition(Vector(30.0 * i, 0, 0));
        }

        auto loss = CreateObject<MatrixPropagationLossModel>();
        loss->SetDefaultLoss(config.crossLossDb);
        for (uint32_t i = 0; i < config.nodeCount; i += 2)
        {
            loss->SetLoss(m_nodes.Get(i)->GetObject<MobilityModel>(),
                          m_nodes.Get(i + 1)->GetObject<MobilityModel>(),
                          config.desiredLossDb,
                          true);
        }
        m_delay = CreateObject<ConstantSpeedPropagationDelayModel>();
        m_delay->SetAttribute("Speed", DoubleValue(300000000));
        auto channel = CreateObject<SingleModelSpectrumChannel>();
        channel->AddPropagationLossModel(loss);
        channel->SetPropagationDelayModel(m_delay);
        SpectrumWifiPhyHelper phy;
        phy.SetChannel(channel);
        phy.Set("ChannelSettings", StringValue("{36, 20, BAND_5GHZ, 0}"));
        phy.Set("TxPowerStart", DoubleValue(16));
        phy.Set("TxPowerEnd", DoubleValue(16));
        phy.Set("TxPowerLevels", UintegerValue(1));
        phy.Set("TxGain", DoubleValue(0));
        phy.Set("RxGain", DoubleValue(0));
        phy.Set("RxNoiseFigure", DoubleValue(7));
        phy.Set("RxSensitivity", DoubleValue(-101));
        phy.Set("CcaEdThreshold", DoubleValue(-62));
        phy.Set("CcaSensitivity", DoubleValue(-82));
        phy.SetErrorRateModel("ns3::NistErrorRateModel");
        phy.SetPreambleDetectionModel("ns3::ThresholdPreambleDetectionModel",
                                      "Threshold",
                                      DoubleValue(4),
                                      "MinimumRssi",
                                      DoubleValue(-82));
        WifiHelper wifi;
        wifi.SetStandard(WIFI_STANDARD_80211a);
        wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                     "DataMode",
                                     StringValue("OfdmRate6Mbps"),
                                     "ControlMode",
                                     StringValue("OfdmRate6Mbps"),
                                     "RtsCtsThreshold",
                                     UintegerValue(65535),
                                     "FragmentationThreshold",
                                     UintegerValue(65534));
        wifi.DisableFlowControl();
        ProbeMacHelper mac(config.dcf);
        m_devices = wifi.Install(phy, mac, m_nodes);
        m_result.streamsAssigned = wifi.AssignStreams(m_devices, 0);
        m_recorders.resize(config.nodeCount);
        for (uint32_t i = 0; i < config.nodeCount; ++i)
        {
            auto device = Device(i + 1);
            m_addresses.emplace(device->GetMac()->GetAddress(), i + 1);
            auto queue = device->GetMac()->GetTxop()->GetWifiMacQueue();
            queue->SetMaxSize(QueueSize(std::to_string(config.queueMaxPackets) + "p"));
            queue->SetAttribute("MaxDelay", TimeValue(NanoSeconds(config.queueLifetimeNs)));
            auto& recorder = m_recorders.at(i);
            recorder.owner = this;
            recorder.node = i + 1;
            Connect(device->GetPhy(),
                    "PhyTxPsduBegin",
                    MakeCallback(&Recorder::TxBegin, &recorder));
            Connect(device->GetPhy(), "PhyTxEnd", MakeCallback(&Recorder::TxEnd, &recorder));
            Connect(device->GetPhy(), "MonitorSnifferRx", MakeCallback(&Recorder::Rx, &recorder));
            Connect(device->GetPhy(), "PhyRxDrop", MakeCallback(&Recorder::RxDrop, &recorder));
            Connect(device->GetMac(), "AckedMpdu", MakeCallback(&Recorder::Acked, &recorder));
            Connect(device->GetMac(),
                    "MpduResponseTimeout",
                    MakeCallback(&Recorder::Timeout, &recorder));
            Connect(device->GetMac(), "DroppedMpdu", MakeCallback(&Recorder::Dropped, &recorder));
            Connect(queue, "Expired", MakeCallback(&Recorder::Expired, &recorder));
            if (!config.dcf)
            {
                Connect(device->GetMac()->GetTxop(),
                        "EnqueueRejected",
                        MakeCallback(&Recorder::Rejected, &recorder));
            }
            device->SetReceiveCallback(MakeCallback(&Recorder::UpperRx, &recorder));
            if (config.dropAckAtSenders && i % 2 == 0)
            {
                device->GetPhy()->SetPostReceptionErrorModel(CreateObject<ProbeAckErrorModel>());
            }
        }
        Simulator::Schedule(NanoSeconds(config.enqueueNs), &ProbeRun::Enqueue, this);
        for (const auto& window : config.windows)
        {
            Simulator::Schedule(NanoSeconds(window.startNs), &ProbeRun::Grant, this, window);
            Simulator::Schedule(NanoSeconds(window.endNs), &ProbeRun::Close, this);
        }
        Simulator::Stop(NanoSeconds(config.stopNs));
        Simulator::Run();
        Snapshot("final_queue");
        // Guard destroys nodes/events while raw callback targets are still alive.
        return m_result;
    }

  private:
    struct Recorder
    {
        ProbeRun* owner{};
        uint32_t node{};

        void TxBegin(WifiConstPsduMap psdus, WifiTxVector txVector, double powerW)
        {
            for (const auto& [staId, psdu] : psdus)
            {
                for (const auto& mpdu : *psdu)
                {
                    auto event = owner->MpduEvent("tx_begin", node, mpdu);
                    event.bytes = mpdu->GetSize();
                    event.durationNs =
                        WifiPhy::CalculateTxDuration(psdu->GetSize(),
                                                     txVector,
                                                     owner->Device(node)->GetPhy()->GetPhyBand())
                            .GetNanoSeconds();
                    std::ostringstream detail;
                    detail << txVector.GetMode() << ";power_w=" << std::setprecision(17) << powerW;
                    event.detail = detail.str();
                    owner->Add(std::move(event));
                }
            }
        }

        void TxEnd(Ptr<const Packet> packet)
        {
            owner->Add(owner->FrameEvent("tx_end", node, packet));
        }

        void Rx(Ptr<const Packet> packet,
                uint16_t frequency,
                WifiTxVector txVector,
                MpduInfo info,
                SignalNoiseDbm signal,
                uint16_t staId)
        {
            auto event = owner->FrameEvent("rx_ok", node, packet);
            event.signalDbm = signal.signal;
            event.noiseDbm = signal.noise;
            owner->Add(std::move(event));
        }

        void RxDrop(Ptr<const Packet> packet, WifiPhyRxfailureReason reason)
        {
            auto event = owner->FrameEvent("rx_drop", node, packet);
            std::ostringstream detail;
            detail << reason;
            event.detail = detail.str();
            owner->Add(std::move(event));
        }

        void Acked(Ptr<const WifiMpdu> mpdu)
        {
            owner->Add(owner->MpduEvent("acked_mpdu", node, mpdu));
        }

        void Timeout(uint8_t reason, Ptr<const WifiMpdu> mpdu, const WifiTxVector& txVector)
        {
            auto event = owner->MpduEvent("response_timeout", node, mpdu);
            event.detail = std::to_string(reason);
            owner->Add(std::move(event));
        }

        void Dropped(WifiMacDropReason reason, Ptr<const WifiMpdu> mpdu)
        {
            auto event = owner->MpduEvent("mac_drop", node, mpdu);
            event.detail = std::to_string(static_cast<int>(reason));
            owner->Add(std::move(event));
        }

        void Expired(Ptr<const WifiMpdu> mpdu)
        {
            owner->Add(owner->MpduEvent("queue_expired", node, mpdu));
        }

        void Rejected(Ptr<const WifiMpdu> mpdu)
        {
            auto event = owner->MpduEvent("queue_rejected", node, mpdu);
            event.detail = "enqueue_returned_false";
            owner->Add(std::move(event));
        }

        bool UpperRx(Ptr<NetDevice> device,
                     Ptr<const Packet> packet,
                     uint16_t protocol,
                     const Address& sender)
        {
            WifiProbeEvent event;
            event.kind = "upper_rx";
            event.node = node;
            event.peer = owner->Peer(Mac48Address::ConvertFrom(sender));
            event.packetId = PacketId(packet);
            event.bytes = packet->GetSize();
            event.frame = "data";
            owner->Add(std::move(event));
            return true;
        }
    };

    void Connect(Ptr<Object> object, const std::string& name, const CallbackBase& callback)
    {
        if (!object->TraceConnectWithoutContext(name, callback))
        {
            throw std::runtime_error("Missing probe trace: " + name);
        }
    }

    Ptr<WifiNetDevice> Device(uint32_t node) const
    {
        return StaticCast<WifiNetDevice>(m_devices.Get(node - 1));
    }

    uint32_t Peer(Mac48Address address) const
    {
        auto it = m_addresses.find(address);
        return it == m_addresses.end() ? 0 : it->second;
    }

    WifiProbeEvent FrameEvent(const std::string& kind,
                              uint32_t node,
                              Ptr<const Packet> packet) const
    {
        WifiMacHeader header;
        packet->PeekHeader(header);
        WifiProbeEvent event;
        event.kind = kind;
        event.node = node;
        event.peer = Peer(header.GetAddr1());
        event.packetId = PacketId(packet);
        event.frame = FrameKind(header);
        event.bytes = packet->GetSize();
        return event;
    }

    WifiProbeEvent MpduEvent(const std::string& kind,
                             uint32_t node,
                             Ptr<const WifiMpdu> mpdu) const
    {
        WifiProbeEvent event;
        event.kind = kind;
        event.node = node;
        event.peer = Peer(mpdu->GetHeader().GetAddr1());
        event.packetId = PacketId(mpdu->GetPacket());
        event.frame = FrameKind(mpdu->GetHeader());
        event.bytes = mpdu->GetSize();
        return event;
    }

    void Add(WifiProbeEvent event)
    {
        event.sequence = m_result.events.size();
        event.timeNs = Simulator::Now().GetNanoSeconds();
        if ((event.signalDbm && !std::isfinite(*event.signalDbm)) ||
            (event.noiseDbm && !std::isfinite(*event.noiseDbm)))
        {
            throw std::runtime_error("Nonfinite observed radio quantity");
        }
        m_result.events.push_back(std::move(event));
    }

    void Snapshot(const std::string& kind)
    {
        for (uint32_t node = 1; node <= m_result.config.nodeCount; ++node)
        {
            WifiProbeEvent event;
            event.kind = kind;
            event.node = node;
            event.queuePackets =
                Device(node)->GetMac()->GetTxop()->GetWifiMacQueue()->GetNPackets();
            Add(std::move(event));
        }
    }

    void Enqueue()
    {
        const auto& config = m_result.config;
        for (uint32_t node = 1; node <= config.nodeCount; node += 2)
        {
            for (uint32_t i = 0; i < config.packetsPerSender; ++i)
            {
                auto packet = Create<Packet>(config.payloadBytes);
                ProbePacketTag tag;
                tag.Set(static_cast<uint64_t>(node) * 1000 + i + 1);
                packet->AddPacketTag(tag);
                WifiProbeEvent event;
                event.kind = "injected";
                event.node = node;
                event.peer = node + 1;
                event.packetId = tag.Get();
                event.bytes = config.payloadBytes;
                event.frame = "data";
                Add(std::move(event));
                if (!Device(node)->Send(packet, Device(node + 1)->GetAddress(), 0x88b5))
                {
                    throw std::runtime_error("WifiNetDevice rejected injection");
                }
            }
        }
        Snapshot("after_enqueue");
    }

    void Grant(ProbeWindow window)
    {
        WifiProbeEvent open;
        open.kind = "window_open";
        open.durationNs = window.endNs - window.startNs;
        open.detail = m_result.config.dcf ? "dcf_observation_only" : "explicit_grants";
        Add(std::move(open));
        if (m_result.config.dcf)
        {
            return;
        }
        // The entire input was validated before any node was created.
        for (const auto& [tx, rx] : window.links)
        {
            auto txop = StaticCast<ProbeTxop>(Device(tx)->GetMac()->GetTxop());
            auto delay = m_delay->GetDelay(m_nodes.Get(tx - 1)->GetObject<MobilityModel>(),
                                          m_nodes.Get(rx - 1)->GetObject<MobilityModel>());
            WifiProbeEvent event;
            event.kind = "grant";
            event.node = tx;
            event.peer = rx;
            event.durationNs = txop->GetTransactionDuration(delay).GetNanoSeconds();
            Add(std::move(event));
            const auto result =
                txop->Grant(NanoSeconds(window.endNs), delay, m_result.config.allowOverrun);
            WifiProbeEvent outcome;
            outcome.kind = "grant_result";
            outcome.node = tx;
            outcome.peer = rx;
            switch (result)
            {
            case ProbeTxop::GrantResult::STARTED:
                outcome.detail = "started";
                break;
            case ProbeTxop::GrantResult::EMPTY:
                outcome.detail = "empty";
                break;
            case ProbeTxop::GrantResult::BUSY:
                outcome.detail = "busy";
                break;
            case ProbeTxop::GrantResult::NO_TIME:
                outcome.detail = "no_time";
                break;
            }
            Add(std::move(outcome));
        }
    }

    void Close()
    {
        Snapshot("window_close");
        // Diagnostic only: this is NOT claimed to be a general equal-time phase barrier.
        Simulator::ScheduleNow(&ProbeRun::Snapshot, this, std::string("close_followup"));
    }

    WifiProbeResult m_result;
    NodeContainer m_nodes;
    NetDeviceContainer m_devices;
    Ptr<ConstantSpeedPropagationDelayModel> m_delay;
    std::map<Mac48Address, uint32_t> m_addresses;
    std::vector<Recorder> m_recorders;
};

void
WriteString(std::ostream& out, const std::string& value)
{
    out << '"';
    for (unsigned char ch : value)
    {
        if (ch == '"' || ch == '\\')
        {
            out << '\\' << ch;
        }
        else if (ch < 0x20)
        {
            out << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<unsigned int>(ch) << std::dec;
        }
        else
        {
            out << ch;
        }
    }
    out << '"';
}

} // namespace

WifiProbeConfig
MakeWifiProbeConfig(const std::string& name)
{
    if (!CASES.contains(name))
    {
        throw std::invalid_argument("Unknown component case: " + name);
    }
    WifiProbeConfig config;
    config.scenario = name;
    if (name == "empty")
    {
        config.windows.front().links.clear();
    }
    else if (name == "gated")
    {
        config.nodeCount = 4;
        config.packetsPerSender = 3;
        config.windows = {{3000000, 4000000, {}},
                          {5000000, 7000000, {{1, 2}}},
                          {10000000, 12000000, {{1, 2}}}};
    }
    else if (name == "dcf" || name == "parallel")
    {
        config.nodeCount = 4;
        config.dcf = name == "dcf";
        config.windows.front().links = {{1, 2}, {3, 4}};
    }
    else if (name.starts_with("interference-"))
    {
        config.nodeCount = 6;
        config.desiredLossDb = 80;
        const auto count = static_cast<uint32_t>(name.back() - '0') + 1;
        config.windows.front().links.clear();
        for (uint32_t i = 0; i < count; ++i)
        {
            config.windows.front().links.emplace_back(2 * i + 1, 2 * i + 2);
        }
    }
    else if (name == "ack-loss")
    {
        config.dropAckAtSenders = true;
    }
    else if (name == "window-short")
    {
        config.windows.front().endNs = 6000000;
    }
    else if (name == "window-overrun")
    {
        config.windows.front().endNs = 6440000;
        config.allowOverrun = true;
    }
    else if (name == "boundary")
    {
        // Independent 802.11a calculation: DATA 1408us + SIFS 16us + ACK 44us + 2*100ns.
        config.windows.front().endNs = 6468200;
    }
    else if (name == "expiry")
    {
        config.queueLifetimeNs = 500000000;
        config.windows = {{502000000, 504000000, {{1, 2}}}};
        config.stopNs = 510000000;
    }
    else if (name == "overflow")
    {
        config.queueMaxPackets = 1;
        config.packetsPerSender = 2;
    }
    return config;
}

void
ValidateWifiProbeConfig(const WifiProbeConfig& config)
{
    if (!CASES.contains(config.scenario) || config.nodeCount < 2 || config.nodeCount > 6 ||
        config.nodeCount % 2 != 0 || config.packetsPerSender > 16 ||
        config.payloadBytes == 0 || config.payloadBytes > 1500 || config.queueMaxPackets == 0 ||
        config.queueMaxPackets > 64 || config.queueLifetimeNs <= 0 ||
        config.queueLifetimeNs > 10000000000 || config.enqueueNs < 1000000 ||
        config.stopNs <= config.enqueueNs || config.stopNs > 1000000000 ||
        config.seed == 0 || config.seed >= 4294944443U || config.run == 0 ||
        !std::isfinite(config.desiredLossDb) || !std::isfinite(config.crossLossDb) ||
        config.desiredLossDb < 0 || config.desiredLossDb > 200 ||
        config.crossLossDb < 0 || config.crossLossDb > 200 ||
        config.windows.empty() || config.windows.size() > 16)
    {
        throw std::invalid_argument("Invalid or unsupported bounded Wi-Fi probe configuration");
    }
    int64_t previousEnd = config.enqueueNs;
    for (const auto& window : config.windows)
    {
        if (window.startNs <= previousEnd || window.endNs <= window.startNs ||
            window.endNs >= config.stopNs - 3000000)
        {
            throw std::invalid_argument("Probe windows must be ordered with a post-window tail");
        }
        std::set<uint32_t> endpoints;
        for (const auto& [tx, rx] : window.links)
        {
            if (tx == 0 || tx > config.nodeCount || rx == 0 || rx > config.nodeCount ||
                !endpoints.insert(tx).second || !endpoints.insert(rx).second)
            {
                throw std::invalid_argument("Invalid/repeated endpoint in complete probe plan");
            }
            if (tx % 2 != 1 || rx != tx + 1)
            {
                throw std::invalid_argument("Probe supports only the injected fixed peer pairs");
            }
        }
        previousEnd = window.endNs;
    }
}

WifiProbeResult
RunWifiExecutionProbe(const WifiProbeConfig& config)
{
    ValidateWifiProbeConfig(config);
    ProbeRun run(config);
    return run.Execute();
}

void
WriteWifiProbeJson(std::ostream& out, const WifiProbeResult& result)
{
    const auto& c = result.config;
    out << std::setprecision(17) << std::boolalpha;
    out << "{\"schema_version\":1,\"scope\":\"wifi_component_diagnostic\",\"build\":{"
           "\"source_scope\":\"component-cpp-h-cmake-v1\",\"source_sha256\":\""
        << FANET_PROBE_SOURCE_SHA256 << "\",\"ns3_commit\":\"" << FANET_PROBE_NS3_COMMIT
        << "\",\"compiler\":\"" << FANET_PROBE_COMPILER << "\"},\"config\":{";
    out << "\"scenario\":";
    WriteString(out, c.scenario);
    out << ",\"node_count\":" << c.nodeCount << ",\"packets_per_sender\":" << c.packetsPerSender
        << ",\"payload_bytes\":" << c.payloadBytes << ",\"queue_max_packets\":" << c.queueMaxPackets
        << ",\"queue_lifetime_ns\":" << c.queueLifetimeNs << ",\"enqueue_ns\":" << c.enqueueNs
        << ",\"stop_ns\":" << c.stopNs << ",\"desired_loss_db\":" << c.desiredLossDb
        << ",\"cross_loss_db\":" << c.crossLossDb << ",\"dcf\":" << c.dcf
        << ",\"drop_ack_at_senders\":" << c.dropAckAtSenders
        << ",\"allow_overrun\":" << c.allowOverrun << ",\"seed\":" << c.seed
        << ",\"run\":" << c.run;
    out << ",\"radio\":{\"standard\":\"802.11a\",\"phy\":\"SpectrumWifiPhy\","
           "\"channel_number\":36,\"center_mhz\":5180,\"width_mhz\":20,"
           "\"data_mode\":\"OfdmRate6Mbps\",\"control_mode\":\"OfdmRate6Mbps\","
           "\"tx_power_dbm\":16,\"tx_gain_db\":0,\"rx_gain_db\":0,\"noise_figure_db\":7,"
           "\"rx_sensitivity_dbm\":-101,\"cca_ed_dbm\":-62,\"cca_sensitivity_dbm\":-82,"
           "\"error_model\":\"NistErrorRateModel\",\"preamble_snr_db\":4,"
           "\"preamble_min_rssi_dbm\":-82,\"frame_capture\":false,\"qos\":false,"
           "\"frame_retry_limit\":1,\"rts_threshold_bytes\":65535,"
           "\"fragment_threshold_bytes\":65534,\"propagation_speed_mps\":300000000,"
           "\"position_spacing_m\":30,\"fading\":false,\"control_delay_ns\":0},\"windows\":[";
    for (std::size_t i = 0; i < c.windows.size(); ++i)
    {
        const auto& w = c.windows[i];
        out << (i ? "," : "") << "{\"start_ns\":" << w.startNs << ",\"end_ns\":" << w.endNs
            << ",\"links\":[";
        for (std::size_t j = 0; j < w.links.size(); ++j)
        {
            out << (j ? "," : "") << "[" << w.links[j].first << "," << w.links[j].second << "]";
        }
        out << "]}";
    }
    out << "]},\"streams\":{\"first\":0,\"assigned\":" << result.streamsAssigned
        << "},\"events\":[";
    for (std::size_t i = 0; i < result.events.size(); ++i)
    {
        const auto& e = result.events[i];
        out << (i ? "," : "") << "{\"sequence\":" << e.sequence << ",\"time_ns\":" << e.timeNs
            << ",\"kind\":";
        WriteString(out, e.kind);
        out << ",\"node\":" << e.node << ",\"peer\":" << e.peer << ",\"packet_id\":" << e.packetId
            << ",\"frame\":";
        WriteString(out, e.frame);
        out << ",\"bytes\":" << e.bytes << ",\"duration_ns\":" << e.durationNs
            << ",\"queue_packets\":" << e.queuePackets << ",\"detail\":";
        WriteString(out, e.detail);
        out << ",\"signal_dbm\":";
        if (e.signalDbm)
        {
            out << *e.signalDbm;
        }
        else
        {
            out << "null";
        }
        out << ",\"noise_dbm\":";
        if (e.noiseDbm)
        {
            out << *e.noiseDbm;
        }
        else
        {
            out << "null";
        }
        out << "}";
    }
    out << "]}\n";
}

} // namespace ns3
