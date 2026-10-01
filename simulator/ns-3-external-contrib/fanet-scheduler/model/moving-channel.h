#ifndef FANET_MOVING_CHANNEL_H
#define FANET_MOVING_CHANNEL_H

#include "controlled-state.h"

#include "ns3/mobility-model.h"

#include <memory>

namespace ns3::fanet
{
/** Boundary position and velocity in metres and metres per second. */
struct MotionState
{
    Vector position;
    Vector velocity;
};

/** Explicit non-random extension of the registered boundary channel facts. */
struct MotionSettings
{
    Vector low;
    Vector high;
    double referenceDistance{};
    double pathLossExponent{};
    double sinrThreshold{};
    std::vector<std::map<Id, MotionState>> boundaries;
};

/**
 * Immutable piecewise specular motion. Knots are the supplied boundary observations;
 * motion between knots uses their causal position/velocity, never future interpolation.
 * Boundary positions are checked for continuity to 1e-9 m (not a SINR tolerance); the
 * velocity may change at a knot, so motion is piecewise constant per cycle (paper1-next).
 */
class MotionTrace
{
  public:
    MotionTrace(const MotionSettings& settings, Ns start, Ns period,
                const std::vector<Id>& nodes);
    /** @return Position and velocity at an in-range absolute time, in integer nanoseconds. */
    MotionState At(Id node, Ns at) const;
    /** @return Index of the current left-closed boundary interval. */
    std::size_t Index(Ns at) const;
    /** @return Multiplicative path-loss change from the current boundary to at. */
    double GainRatio(Link link, Ns at) const;
    /** @return Maximum gain ratio in a given interval, bounded by reference distance. */
    double MaxGainRatio(Link link, std::size_t index) const;
    /** @return The explicitly configured single-MCS SINR threshold. */
    double Threshold() const;
    /** @return First registered absolute timestamp. */
    Ns StartTime() const;

  private:
    MotionSettings m_settings;
    Ns m_start{};
    Ns m_period{};
    Ns m_end{};
};

/** Query-driven mobility attached to the real UAV object; owns no simulator events. */
class TraceMobility : public MobilityModel
{
  public:
    static TypeId GetTypeId();
    /** Attach one immutable trace and stable node identity before querying or copying. */
    void Configure(std::shared_ptr<const MotionTrace> trace, Id node);
    Ptr<MobilityModel> Copy() const override;

  private:
    Vector DoGetPosition() const override;
    Vector DoGetVelocity() const override;
    void DoSetPosition(const Vector& position) override;
    std::shared_ptr<const MotionTrace> m_trace;
    Id m_node{};
};
} // namespace ns3::fanet
#endif
