#ifndef FANET_NONFROZEN_CONTROLLER_H
#define FANET_NONFROZEN_CONTROLLER_H

#include "control-channel.h"
#include "scheduled-radio.h"

namespace ns3::fanet
{
/** Explicit complete-and-fresh report envelope; unsupported reports fail, never become zeros. */
struct CollectionSettings
{
    ControlSettings channel;
    Ns maximumAge{};
    Ns reportPreparation{};
    Ns assembly{};
    Ns distributionWindow{};
};

/** Reconstructed values originate exclusively in the actually received report packets. */
struct CollectedState
{
    Observation sample;
    wire::Value observation;
    wire::Value physical;
    wire::Value reports;
    Ns cut{};
};

/**
 * Ground-controller event adapter for the complete-and-fresh report profile.
 * Full audit records are charged as control bytes in this first explicit encoding.
 * It contains no GNN, decoder, reward or packet-transition implementation.
 */
class NonfrozenController : public Object
{
  public:
    static TypeId GetTypeId();
    void Configure(Ptr<ControlledEnvironment> env, Ptr<ScheduledRadio> radio,
                   const CollectionSettings& settings);
    /** Collect the current boundary's reports through actual control-packet deliveries. */
    CollectedState Collect(const wire::Value& physical);
    /** Inject separately measured/registered encode and decode times; run to fixed cycle end. */
    ControlExecution Execute(const Action& action, Ns encodeTime, Ns decodeTime);
    /** Drain actual control arrivals since the last call, retaining original epoch identities. */
    wire::Value TakeArrivals();

  protected:
    void DoDispose() override;

  private:
    Ptr<ControlledEnvironment> m_env;
    Ptr<ScheduledRadio> m_radio;
    Ptr<Node> m_ground;
    Ptr<ControlChannel> m_channel;
    CollectionSettings m_settings;
    std::optional<CollectedState> m_current;
    std::vector<ControlDelivery> m_arrivals;
    EventId m_assemblyEvent;
    bool m_disposed{false};
};
} // namespace ns3::fanet
#endif
