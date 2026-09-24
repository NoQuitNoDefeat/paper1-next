#include "moving-channel.h"

#include "ns3/simulator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace ns3::fanet
{
namespace
{
void
Require(bool value, const char* message)
{
    if (!value)
    {
        throw std::invalid_argument(message);
    }
}

std::pair<double, double>
Reflect(double position, double velocity, double low, double high, double seconds)
{
    const double width = high - low;
    // Preserve the Python reference operation boundaries rather than fusing multiply/add.
    volatile double displacement = velocity * seconds;
    const double unfolded = position - low + displacement;
    double phase = std::fmod(unfolded, 2 * width);
    if (phase < 0)
    {
        phase += 2 * width;
    }
    if (phase == 0)
    {
        return {low, std::abs(velocity)};
    }
    if (phase == width)
    {
        return {high, -std::abs(velocity)};
    }
    return phase < width ? std::pair{low + phase, velocity}
                         : std::pair{high - (phase - width), -velocity};
}

MotionState
Advance(const MotionState& state, const MotionSettings& settings, double seconds)
{
    MotionState result;
    std::tie(result.position.x, result.velocity.x) =
        Reflect(state.position.x, state.velocity.x, settings.low.x, settings.high.x, seconds);
    std::tie(result.position.y, result.velocity.y) =
        Reflect(state.position.y, state.velocity.y, settings.low.y, settings.high.y, seconds);
    std::tie(result.position.z, result.velocity.z) =
        Reflect(state.position.z, state.velocity.z, settings.low.z, settings.high.z, seconds);
    return result;
}

void
CheckAxis(double position, double velocity, double low, double high)
{
    Require(std::isfinite(low) && std::isfinite(high) && low < high &&
                high - low <= 1e9 && std::abs(low) <= 1e9 && std::abs(high) <= 1e9 &&
                std::isfinite(position) && low <= position && position <= high &&
                std::isfinite(velocity) && std::abs(velocity) <= 1000,
            "invalid bounded motion position, velocity or box");
}
} // namespace

MotionTrace::MotionTrace(const MotionSettings& settings, Ns start, Ns period,
                         const std::vector<Id>& nodes)
    : m_settings(settings), m_start(start), m_period(period)
{
    Require(start >= 0 && period > 0 && settings.boundaries.size() >= 2 &&
                settings.boundaries.size() <= 10001 && !nodes.empty() &&
                std::set<Id>(nodes.begin(), nodes.end()).size() == nodes.size() &&
                std::isfinite(settings.referenceDistance) && settings.referenceDistance > 0 &&
                std::isfinite(settings.pathLossExponent) && settings.pathLossExponent > 0 &&
                settings.pathLossExponent <= 10 && std::isfinite(settings.sinrThreshold) &&
                settings.sinrThreshold > 0 && settings.sinrThreshold <= 1e12,
            "invalid motion trajectory configuration");
    Require(settings.boundaries.size() - 1 <= static_cast<uint64_t>(INT64_MAX / period),
            "motion trajectory duration overflows");
    m_end = AddTime(start, static_cast<Ns>(settings.boundaries.size() - 1) * period);
    for (std::size_t i = 0; i < settings.boundaries.size(); ++i)
    {
        const auto& frame = settings.boundaries[i];
        Require(frame.size() == nodes.size(), "motion boundary node set mismatch");
        for (Id id : nodes)
        {
            Require(frame.contains(id), "motion boundary omits registered node");
            const auto& s = frame.at(id);
            CheckAxis(s.position.x, s.velocity.x, settings.low.x, settings.high.x);
            CheckAxis(s.position.y, s.velocity.y, settings.low.y, settings.high.y);
            CheckAxis(s.position.z, s.velocity.z, settings.low.z, settings.high.z);
            if (i)
            {
                const auto expected = Advance(settings.boundaries[i - 1].at(id), settings,
                                              period / 1e9);
                Require(CalculateDistance(expected.position, s.position) <= 1e-9 &&
                            CalculateDistance(expected.velocity, s.velocity) <= 1e-12,
                        "motion boundary teleports or changes unregistered velocity");
            }
        }
    }
}

std::size_t
MotionTrace::Index(Ns at) const
{
    Require(at >= m_start && at <= m_end, "motion query outside registered trajectory");
    return static_cast<std::size_t>((at - m_start) / m_period);
}

MotionState
MotionTrace::At(Id node, Ns at) const
{
    const auto index = Index(at);
    const auto& boundary = m_settings.boundaries.at(index);
    Require(boundary.contains(node), "motion query for unregistered node");
    const auto elapsed = (at - m_start) % m_period;
    // At a knot return the registered state exactly; do not normalize signed zero or velocity.
    return elapsed == 0 ? boundary.at(node)
                        : Advance(boundary.at(node), m_settings, elapsed / 1e9);
}

double
MotionTrace::GainRatio(Link link, Ns at) const
{
    const auto index = Index(at);
    const auto& boundary = m_settings.boundaries.at(index);
    Require(link.first != link.second && boundary.contains(link.first) &&
                boundary.contains(link.second), "invalid motion channel link");
    if ((at - m_start) % m_period == 0)
    {
        return 1.0;
    }
    const double base = std::max(m_settings.referenceDistance,
        CalculateDistance(boundary.at(link.first).position, boundary.at(link.second).position));
    const double now = std::max(m_settings.referenceDistance,
        CalculateDistance(At(link.first, at).position, At(link.second, at).position));
    const double ratio = std::pow(now / base, -m_settings.pathLossExponent);
    Require(std::isfinite(ratio) && ratio >= 0, "nonfinite moving path loss");
    return ratio;
}

double
MotionTrace::MaxGainRatio(Link link, std::size_t index) const
{
    const auto& boundary = m_settings.boundaries.at(index);
    const double base = std::max(m_settings.referenceDistance,
        CalculateDistance(boundary.at(link.first).position, boundary.at(link.second).position));
    const double ratio = std::pow(base / m_settings.referenceDistance, m_settings.pathLossExponent);
    Require(std::isfinite(ratio), "moving path loss upper bound overflows");
    return ratio;
}

double
MotionTrace::Threshold() const
{
    return m_settings.sinrThreshold;
}

Ns
MotionTrace::StartTime() const
{
    return m_start;
}

NS_OBJECT_ENSURE_REGISTERED(TraceMobility);

TypeId
TraceMobility::GetTypeId()
{
    static TypeId tid = TypeId("ns3::fanet::TraceMobility")
                            .SetParent<MobilityModel>()
                            .SetGroupName("FanetScheduler")
                            .AddConstructor<TraceMobility>();
    return tid;
}

void
TraceMobility::Configure(std::shared_ptr<const MotionTrace> trace, Id node)
{
    Require(!m_trace && trace, "mobility requires one immutable motion trace");
    trace->At(node, trace->StartTime());
    m_trace = std::move(trace);
    m_node = node;
    NotifyCourseChange();
}

Ptr<MobilityModel>
TraceMobility::Copy() const
{
    Require(m_trace != nullptr, "cannot copy unconfigured motion");
    auto copy = CreateObject<TraceMobility>();
    copy->Configure(m_trace, m_node);
    return copy;
}

Vector
TraceMobility::DoGetPosition() const
{
    Require(m_trace != nullptr, "unconfigured motion");
    return m_trace->At(m_node, Simulator::Now().GetNanoSeconds()).position;
}

Vector
TraceMobility::DoGetVelocity() const
{
    Require(m_trace != nullptr, "unconfigured motion");
    return m_trace->At(m_node, Simulator::Now().GetNanoSeconds()).velocity;
}

void
TraceMobility::DoSetPosition(const Vector&)
{
    throw std::logic_error("registered motion is immutable; configure a new episode");
}
} // namespace ns3::fanet
