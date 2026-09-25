#include "scheduled-radio.h"

#include "ns3/constant-position-mobility-model.h"
#include "ns3/half-duplex-ideal-phy-signal-parameters.h"
#include "ns3/half-duplex-ideal-phy.h"
#include "ns3/header.h"
#include "ns3/propagation-delay-model.h"
#include "ns3/simple-net-device.h"
#include "ns3/simulator.h"
#include "ns3/single-model-spectrum-channel.h"
#include "ns3/spectrum-propagation-loss-model.h"
#include "ns3/spectrum-value.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <set>
#include <stdexcept>

namespace ns3::fanet
{
namespace
{
void
Require(bool ok, const char* message)
{
    if (!ok)
    {
        throw std::invalid_argument(message);
    }
}

class IdentityHeader : public Header
{
  public:
    Id packet{};
    Link link;

    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::fanet::IdentityHeader")
                                .SetParent<Header>()
                                .AddConstructor<IdentityHeader>();
        return tid;
    }

    TypeId GetInstanceTypeId() const override
    {
        return GetTypeId();
    }

    uint32_t GetSerializedSize() const override
    {
        return 24;
    }

    void Serialize(Buffer::Iterator i) const override
    {
        i.WriteHtonU64(packet);
        i.WriteHtonU64(link.first);
        i.WriteHtonU64(link.second);
    }

    uint32_t Deserialize(Buffer::Iterator i) override
    {
        packet = i.ReadNtohU64();
        link = {i.ReadNtohU64(), i.ReadNtohU64()};
        return 24;
    }

    void Print(std::ostream& out) const override
    {
        out << packet << ':' << link.first << "->" << link.second;
    }
};

IdentityHeader
Identity(Ptr<const ns3::Packet> packet)
{
    Require(packet && packet->GetSize() >= 24, "radio packet missing identity header");
    IdentityHeader header;
    packet->PeekHeader(header);
    return header;
}

class FixedDelay : public PropagationDelayModel
{
  public:
    Ns delay{};

    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::fanet::FixedRadioDelay")
                                .SetParent<PropagationDelayModel>()
                                .AddConstructor<FixedDelay>();
        return tid;
    }

    Time GetDelay(Ptr<MobilityModel>, Ptr<MobilityModel>) const override
    {
        return NanoSeconds(delay);
    }

    int64_t DoAssignStreams(int64_t) override
    {
        return 0;
    }
};

class MatrixLoss : public SpectrumPropagationLossModel
{
  public:
    std::map<Ptr<const MobilityModel>, Id> identities;
    std::map<Link, double> gains;
    std::shared_ptr<const MotionTrace> motion;
    std::function<void(const ReceptionSample&)> sample;

    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::fanet::MatrixSpectrumLoss")
                                .SetParent<SpectrumPropagationLossModel>()
                                .AddConstructor<MatrixLoss>();
        return tid;
    }

    int64_t DoAssignStreams(int64_t) override
    {
        return 0;
    }

    Ptr<SpectrumValue> DoCalcRxPowerSpectralDensity(Ptr<const SpectrumSignalParameters> params,
                                                    Ptr<const MobilityModel> a,
                                                    Ptr<const MobilityModel> b) const override
    {
        auto received = params->psd->Copy();
        const Link link{identities.at(a), identities.at(b)};
        const Ns at = Simulator::Now().GetNanoSeconds();
        const double gain = gains.at(link) * (motion ? motion->GainRatio(link, at) : 1.0);
        *received *= gain;
        if (motion)
        {
            const auto data = DynamicCast<const HalfDuplexIdealPhySignalParameters>(params);
            Require(data != nullptr, "moving channel received unsupported signal");
            const auto header = Identity(data->data);
            if (header.link == link)
            {
                sample({header.packet, link, at, gain, Integral(*received),
                        a->GetPosition(), b->GetPosition()});
            }
        }
        return received;
    }
};

class PlanPhy : public HalfDuplexIdealPhy
{
  public:
    Id node{};
    std::optional<Id> peer;
    std::function<void()> signalEnded;

    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::fanet::PlanSpectrumPhy")
                                .SetParent<HalfDuplexIdealPhy>()
                                .AddConstructor<PlanPhy>();
        return tid;
    }

    void StartRx(Ptr<SpectrumSignalParameters> params) override
    {
        auto data = DynamicCast<HalfDuplexIdealPhySignalParameters>(params);
        Require(data != nullptr, "unsupported signal in scheduled development radio");
        const auto identity = Identity(data->data);
        if (identity.link.second == node && peer == identity.link.first)
        {
            HalfDuplexIdealPhy::StartRx(params);
        }
        else
        {
            // Preserve PSD/duration and all interference, without synchronizing to another peer.
            HalfDuplexIdealPhy::StartRx(Create<SpectrumSignalParameters>(*params));
        }
        // Registered after this signal's upstream subtract/EndRx callbacks, at the same time.
        // Counting every signal closes the cycle only after all PHY work has actually drained.
        Simulator::Schedule(params->duration, signalEnded);
    }
};
} // namespace

struct ScheduledRadio::Impl
{
    Ptr<ControlledEnvironment> env;
    RadioSettings settings;
    std::vector<RadioFrame> frames;
    Ptr<SingleModelSpectrumChannel> channel;
    Ptr<MatrixLoss> loss;
    std::map<Id, Ptr<PlanPhy>> phys;
    std::map<Link, std::deque<Id>> pending;
    std::map<Link, std::pair<Id, Ns>> transmitting;
    struct Attempt
    {
        Id packet{};
        bool copied{false};
        bool committed{false};
    };
    std::map<Link, Attempt> attempts;
    std::vector<RadioEvent> events;
    std::vector<ReceptionSample> receptionSamples;
    ExecutionChannel executionChannel;
    std::shared_ptr<const MotionTrace> motion;
    Bytes payload{};
    Bytes frameBytes{};
    std::size_t epoch{};
    Ns end{};
    uint64_t starts{};
    uint64_t txActive{};
    uint64_t signals{};
    bool active{false};

    void SampleExecution(const std::vector<Link>& links, Ns at)
    {
        Require(at == Simulator::Now().GetNanoSeconds(),
                "execution audit must sample the actual current event");
        const auto& frame = frames.at(motion ? motion->Index(at) : epoch);
        executionChannel = {at, links, frame.noiseW, {}, {}};
        for (const auto& link : links)
        {
            executionChannel.powersW.emplace(link.first, frame.powersW.at(link.first));
            for (const auto& other : links)
            {
                const Link path{link.first, other.second};
                const double gain = frame.gains.at(path) *
                                    (motion ? motion->GainRatio(path, at) : 1.0);
                executionChannel.gains.emplace(path, gain);
            }
        }
    }

    void Record(const char* kind, const IdentityHeader& h)
    {
        const bool ack = settings.actualAck && !pending.contains(h.link);
        const Link original = ack ? Link{h.link.second, h.link.first} : h.link;
        events.push_back(
            {kind, h.packet, original, Simulator::Now().GetNanoSeconds(), payload,
             ack ? settings.ackBytes : frameBytes});
    }

    void CheckDrain()
    {
        if (active && starts == 0 && txActive == 0 && signals == 0)
        {
            Require(transmitting.empty(), "radio missing intended reception result");
            Require(attempts.empty(), "radio missing transaction deadline");
            active = false;
            env->FinishExternalExecution();
        }
    }

    void SignalEnded()
    {
        Require(active && signals > 0, "unexpected radio signal completion");
        --signals;
        CheckDrain();
    }

    void TxEnded(Ptr<const ns3::Packet> p)
    {
        Require(active && txActive > 0, "unexpected radio tx completion");
        const auto h = Identity(p);
        Record(settings.actualAck && !pending.contains(h.link) ? "ack_tx_end" : "tx_end", h);
        --txActive;
        CheckDrain();
    }

    void Reception(Ptr<const ns3::Packet> p, bool correct)
    {
        const auto h = Identity(p);
        const auto found = transmitting.find(h.link);
        Require(active && found != transmitting.end() && found->second.first == h.packet &&
                    found->second.second == Simulator::Now().GetNanoSeconds() &&
                    Simulator::Now().GetNanoSeconds() <= end,
                "unexpected, duplicate or out-of-window intended reception");
        const bool ack = settings.actualAck && !pending.contains(h.link);
        Record(ack ? (correct ? "ack_rx_ok" : "ack_rx_error")
                   : (correct ? "rx_ok" : "rx_error"), h);
        transmitting.erase(found);
        if (settings.actualAck)
        {
            const Link key = ack ? Link{h.link.second, h.link.first} : h.link;
            auto& attempt = attempts.at(key);
            Require(attempt.packet == h.packet && !attempt.committed,
                    "duplicate or stale radio transaction");
            if (ack)
            {
                Require(attempt.copied, "ACK without a DATA receive copy");
                if (correct)
                {
                    env->ConfirmExternalService(key, h.packet);
                    attempt.committed = true;
                    IdentityHeader original;
                    original.packet = h.packet;
                    original.link = key;
                    Record("transaction_commit", original);
                }
            }
            else if (correct)
            {
                // This copy is not an authoritative research packet or upper-layer delivery.
                Require(!attempt.copied, "duplicate receive copy");
                attempt.copied = true;
                ++starts;
                Simulator::Schedule(NanoSeconds(settings.turnaroundNs), [this, key, id = h.packet] {
                    --starts;
                    IdentityHeader ackHeader;
                    ackHeader.packet = id;
                    ackHeader.link = {key.second, key.first};
                    Transmit(ackHeader, settings.ackBytes, settings.ackRate, true);
                });
            }
            return;
        }
        if (correct)
        {
            env->ConfirmExternalService(h.link, h.packet);
            Record("ideal_confirmation", h);
            Require(pending.at(h.link).front() == h.packet, "non-FIFO radio completion");
            pending.at(h.link).pop_front();
            QueueStart(h.link, Simulator::Now().GetNanoSeconds());
        }
        else
        {
            // A failed head blocks this link for the remainder of the current cycle.
            pending.at(h.link).clear();
        }
    }

    void RxOk(Ptr<const ns3::Packet> p)
    {
        Reception(p, true);
    }

    void RxError(Ptr<const ns3::Packet> p)
    {
        Reception(p, false);
    }

    void Transmit(const IdentityHeader& identity, Bytes size, Bytes rate, bool ack)
    {
        const auto completion = AddTime(
            AddTime(Simulator::Now().GetNanoSeconds(), PacketDuration(size, rate)),
            settings.propagationNs);
        Require(completion <= end &&
                    transmitting.emplace(identity.link, std::make_pair(identity.packet, completion))
                        .second,
                "overlapping or late radio transmission");
        auto packet = Create<ns3::Packet>(static_cast<uint32_t>(size - 24));
        packet->AddHeader(identity);
        ++txActive;
        signals += phys.size() - 1;
        Record(ack ? "ack_tx_start" : "tx_start", identity);
        phys.at(identity.link.first)->SetRate(DataRate(rate * 8));
        Require(!phys.at(identity.link.first)->StartTx(packet), "PHY refused authorized transmission");
    }

    void FinishAttempt(const IdentityHeader& identity)
    {
        const auto found = attempts.find(identity.link);
        Require(found != attempts.end() && found->second.packet == identity.packet,
                "missing transaction at confirmation deadline");
        const bool committed = found->second.committed;
        attempts.erase(found); // Discard the non-authoritative copy on either outcome.
        --starts;
        if (committed)
        {
            Require(pending.at(identity.link).front() == identity.packet, "non-FIFO ACK commit");
            pending.at(identity.link).pop_front();
            QueueStart(identity.link, Simulator::Now().GetNanoSeconds());
        }
        else
        {
            Record("ack_timeout", identity);
            pending.at(identity.link).clear();
        }
        CheckDrain();
    }

    void QueueStart(Link key, Ns at)
    {
        if (pending.at(key).empty())
        {
            return;
        }
        ++starts;
        Simulator::Schedule(NanoSeconds(at) - Simulator::Now(), [this, key] {
            --starts;
            const auto rate = frames.at(epoch).rates.at(key);
            const auto duration = PacketDuration(frameBytes, rate);
            auto completion = AddTime(AddTime(Simulator::Now().GetNanoSeconds(), duration),
                                      settings.propagationNs);
            if (settings.actualAck)
            {
                completion = AddTime(AddTime(AddTime(completion, settings.turnaroundNs),
                                               PacketDuration(settings.ackBytes, settings.ackRate)),
                                     settings.propagationNs);
            }
            IdentityHeader identity;
            identity.packet = pending.at(key).front();
            identity.link = key;
            if (completion > end)
            {
                Record("window_insufficient", identity);
                pending.at(key).clear();
                CheckDrain();
                return;
            }
            if (settings.actualAck)
            {
                Require(attempts.emplace(key, Attempt{identity.packet}).second,
                        "duplicate active radio attempt");
                ++starts;
                Simulator::Schedule(NanoSeconds(completion) - Simulator::Now(), [this, identity] {
                    // The deadline was registered before the ACK. Settle after its same-time
                    // receive callback; the environment's external ticket stays held throughout.
                    Simulator::ScheduleNow([this, identity] { FinishAttempt(identity); });
                });
            }
            Transmit(identity, frameBytes, rate, false);
        });
    }
};

NS_OBJECT_ENSURE_REGISTERED(ScheduledRadio);

TypeId
ScheduledRadio::GetTypeId()
{
    static TypeId tid = TypeId("ns3::fanet::ScheduledRadio")
                            .SetParent<Object>()
                            .SetGroupName("FanetScheduler")
                            .AddConstructor<ScheduledRadio>();
    return tid;
}

ScheduledRadio::ScheduledRadio()
    : m_impl(std::make_unique<Impl>())
{
}

ScheduledRadio::~ScheduledRadio() = default;

void
ScheduledRadio::Configure(Ptr<ControlledEnvironment> env,
                          const RadioSettings& settings,
                          const std::vector<RadioFrame>& frames,
                          const std::optional<MotionSettings>& motion)
{
    auto& m = *m_impl;
    Require(!m.env && env, "radio already configured or missing environment");
    const auto observation = env->Observe();
    const auto& config = observation.config;
    const auto trajectory = motion ? std::make_shared<const MotionTrace>(
                                        *motion, config.start, config.period, config.nodes)
                                   : nullptr;
    Require(!motion || motion->boundaries.size() == frames.size(),
            "moving channel requires complete motion boundaries");
    Require(config.nodes.size() >= 2 && config.nodes.size() <= 64,
            "development radio supports 2 through 64 nodes; not a capacity limit");
    Require(std::isfinite(settings.centerHz) && std::isfinite(settings.bandwidthHz) &&
                settings.bandwidthHz >= 1 && settings.bandwidthHz <= 1e8 &&
                settings.centerHz > settings.bandwidthHz / 2 && settings.centerHz <= 1e11 &&
                settings.propagationNs >= 0 && settings.propagationNs <= config.period &&
                settings.overheadBytes >= 24 && settings.overheadBytes <= 4096 &&
                config.packetBytes <= 65536,
            "invalid development radio settings");
    if (settings.actualAck)
    {
        Require(settings.ackBytes >= 24 && settings.ackBytes <= 4096 && settings.ackRate > 0 &&
                    settings.ackRate <= UINT64_MAX / 8 && settings.turnaroundNs >= 0 &&
                    settings.turnaroundNs <= config.period,
                "invalid actual ACK settings");
        PacketDuration(settings.ackBytes, settings.ackRate);
    }
    Require(frames.size() == observation.remaining + 1, "incomplete private radio trajectory");
    for (std::size_t i = 0; i < frames.size(); ++i)
    {
        const auto& frame = frames[i];
        Require(frame.at == config.start + static_cast<Ns>(i) * config.period &&
                    std::isfinite(frame.noiseW) && frame.noiseW > 0 && frame.noiseW <= 1e3 &&
                    frame.powersW.size() == config.nodes.size() &&
                    frame.gains.size() == config.nodes.size() * (config.nodes.size() - 1),
                "invalid private radio frame");
        for (Id a : config.nodes)
        {
            const auto power = frame.powersW.at(a);
            Require(std::isfinite(power) && power > 0 && power <= 1e3, "invalid tx power");
            for (Id b : config.nodes)
            {
                if (a != b)
                {
                    const double gain = frame.gains.at({a, b});
                    Require(std::isfinite(gain) && gain >= 0 && gain <= 1e3,
                            "invalid complete channel gain");
                    // The pinned Shannon model accumulates uint32 bytes per chunk.
                    const double maxGain = gain * (trajectory ? trajectory->MaxGainRatio({a, b}, i)
                                                             : 1.0);
                    Require(std::isfinite(maxGain) && maxGain <= 1e3,
                            "moving channel gain bound exceeded");
                    const double bound = settings.bandwidthHz *
                                         std::log2(1 + power * maxGain / frame.noiseW) *
                                         (config.period / 1e9) / 8;
                    Require(std::isfinite(bound) && bound < UINT32_MAX / 2,
                            "Shannon accumulator bound exceeded");
                }
            }
        }
        for (const auto& [key, rate] : frame.rates)
        {
            Require(
                std::ranges::any_of(config.queues, [&](const auto& q) { return q.key == key; }) &&
                    rate > 0 && rate <= UINT64_MAX / 8,
                "invalid radio rate/queue");
            PacketDuration(config.packetBytes + settings.overheadBytes, rate);
        }
    }
    m.env = env;
    m.settings = settings;
    m.frames = frames;
    m.motion = trajectory;
    m.payload = config.packetBytes;
    m.frameBytes = m.payload + settings.overheadBytes;
    m.channel = CreateObject<SingleModelSpectrumChannel>();
    m.loss = CreateObject<MatrixLoss>();
    m.loss->motion = trajectory;
    m.loss->sample = [&m](const ReceptionSample& sample) { m.receptionSamples.push_back(sample); };
    m.channel->AddSpectrumPropagationLossModel(m.loss);
    auto delay = CreateObject<FixedDelay>();
    delay->delay = settings.propagationNs;
    m.channel->SetPropagationDelayModel(delay);
    Bands bands{{settings.centerHz - settings.bandwidthHz / 2,
                 settings.centerHz,
                 settings.centerHz + settings.bandwidthHz / 2}};
    auto spectrum = Create<SpectrumModel>(bands);
    for (Id id : config.nodes)
    {
        auto node = env->GetNode(id);
        Ptr<MobilityModel> mobility;
        if (trajectory)
        {
            auto moving = CreateObject<TraceMobility>();
            moving->Configure(trajectory, id);
            mobility = moving;
        }
        else
        {
            mobility = CreateObject<ConstantPositionMobilityModel>();
            mobility->SetPosition(Vector(static_cast<double>(m.phys.size()), 0, 0));
        }
        node->AggregateObject(mobility);
        auto device = CreateObject<SimpleNetDevice>();
        node->AddDevice(device);
        auto phy = CreateObject<PlanPhy>();
        phy->node = id;
        phy->SetDevice(device);
        phy->SetMobility(mobility);
        phy->SetChannel(m.channel);
        phy->SetTxPowerSpectralDensity(Create<SpectrumValue>(spectrum));
        phy->TraceConnectWithoutContext("TxEnd", MakeCallback(&Impl::TxEnded, &m));
        phy->TraceConnectWithoutContext("RxEndOk", MakeCallback(&Impl::RxOk, &m));
        phy->TraceConnectWithoutContext("RxEndError", MakeCallback(&Impl::RxError, &m));
        phy->signalEnded = [&m] { m.SignalEnded(); };
        m.channel->AddRx(phy);
        m.loss->identities.emplace(mobility, id);
        m.phys.emplace(id, phy);
    }
}

void
ScheduledRadio::StartCycle(const Action& action, bool boundaryFirst, Ns startOffset)
{
    Begin(action, boundaryFirst, startOffset, false);
}

void
ScheduledRadio::StartPreparedCycle(const Action& execution)
{
    Begin(execution, true, Simulator::Now().GetNanoSeconds() - execution.reference.sampledAt, true);
}

void
ScheduledRadio::Begin(const Action& action, bool boundaryFirst, Ns startOffset, bool prepared)
{
    auto& m = *m_impl;
    Require(m.env && !m.active, "radio requires a settled configured environment");
    const auto observation = prepared ? m.env->ExternalSample(action.reference)
                                      : m.env->Observe(action.reference);
    if (prepared)
    {
        m.env->ValidateExternalExecution(action);
    }
    else
    {
        m.env->ValidateAction(action);
    }
    Require(startOffset >= 0 && startOffset <= observation.config.period,
            "invalid radio start offset");
    const auto epoch = action.reference.epoch;
    const auto& frame = m.frames.at(epoch);
    std::set<Id> endpoints;
    for (auto [a, b] : action.links)
    {
        Require(frame.rates.contains({a, b}) && endpoints.insert(a).second &&
                    endpoints.insert(b).second,
                "radio plan rate or endpoint conflict");
    }
    m.epoch = epoch;
    m.end = action.reference.sampledAt + observation.config.period;
    m.events.clear();
    m.receptionSamples.clear();
    m.pending.clear();
    m.transmitting.clear();
    Require(m.attempts.empty(), "previous radio transaction survived its cycle");
    m.starts = m.txActive = m.signals = 0;
    m.loss->gains = frame.gains;
    for (auto& [id, phy] : m.phys)
    {
        phy->peer.reset();
        auto model = phy->GetRxSpectrumModel();
        auto tx = Create<SpectrumValue>(model);
        *tx = frame.powersW.at(id) / m.settings.bandwidthHz;
        auto noise = Create<SpectrumValue>(model);
        *noise = frame.noiseW / m.settings.bandwidthHz;
        phy->SetTxPowerSpectralDensity(tx);
        phy->SetNoisePowerSpectralDensity(noise);
    }
    for (const auto& key : action.links)
    {
        m.phys.at(key.second)->peer = key.first;
        if (m.settings.actualAck)
        {
            m.phys.at(key.first)->peer = key.second;
        }
        const auto q = std::ranges::find(observation.state.queues, key, &Queue::key);
        for (const auto& entry : q->entries)
        {
            m.pending[key].push_back(entry.packet);
        }
        m.pending.try_emplace(key);
    }
    m.active = true;
    const auto execute = [&] {
        const Ns at = AddTime(action.reference.sampledAt, startOffset);
        // Keep the cycle open until even an empty grant's audit event has completed.
        ++m.starts;
        Simulator::Schedule(NanoSeconds(at - Simulator::Now().GetNanoSeconds()),
            [&m, links = action.links, at] {
                m.SampleExecution(links, at);
                --m.starts;
                m.CheckDrain();
            });
        for (const auto& key : action.links)
        {
            m.QueueStart(key, AddTime(action.reference.sampledAt, startOffset));
        }
        m.CheckDrain();
    };
    if (prepared)
    {
        execute();
    }
    else
    {
        m.env->StartCycle(action, boundaryFirst, execute);
    }
}

const std::vector<RadioEvent>&
ScheduledRadio::Events() const
{
    Require(m_impl->env && !m_impl->active && !m_impl->env->IsBusy(),
            "radio events requested before settlement");
    m_impl->env->Result();
    return m_impl->events;
}

std::set<Link>
ScheduledRadio::ExecutionLinksAt(Ns at) const
{
    const auto& m = *m_impl;
    Require(m.env && m.motion && at == Simulator::Now().GetNanoSeconds(),
            "execution channel must be sampled at the actual current event");
    const auto& frame = m.frames.at(m.motion->Index(at));
    std::set<Link> available;
    for (const auto& [key, rate] : frame.rates)
    {
        const double gain = frame.gains.at(key) * m.motion->GainRatio(key, at);
        if (frame.powersW.at(key.first) * gain / frame.noiseW >= m.motion->Threshold())
        {
            available.insert(key);
        }
    }
    return available;
}

const std::vector<ReceptionSample>&
ScheduledRadio::ReceptionSamples() const
{
    Events(); // The same settled-cycle lifecycle guard applies to both logs.
    return m_impl->receptionSamples;
}

const ExecutionChannel&
ScheduledRadio::ExecutionSnapshot() const
{
    Events();
    return m_impl->executionChannel;
}

void
ScheduledRadio::DoDispose()
{
    if (m_impl)
    {
        for (auto& [id, phy] : m_impl->phys)
        {
            phy->Dispose();
        }
        if (m_impl->channel)
        {
            m_impl->channel->Dispose();
        }
        m_impl->phys.clear();
        m_impl->channel = nullptr;
        m_impl->loss = nullptr;
        m_impl->env = nullptr;
    }
    Object::DoDispose();
}
} // namespace ns3::fanet
