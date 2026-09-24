#ifndef FANET_CONTROL_CHANNEL_H
#define FANET_CONTROL_CHANNEL_H

#include "bridge-wire.h"
#include "controlled-state.h"

#include "ns3/event-id.h"
#include "ns3/node.h"
#include "ns3/object.h"

#include <functional>
#include <map>
#include <set>

namespace ns3::fanet
{
/** Reliable ordered point-to-point control pipes, separate from the scheduled data channel. */
struct ControlSettings
{
    Bytes rate{};                  //!< Wire bytes per second in each direction of each pipe.
    std::map<Id, Ns> propagation;  //!< Explicit finite one-way delays to the ground controller.
};

/** Serialized control report or command, identified independently of ns3-ai transport messages. */
struct ControlMessage
{
    bool downlink{false}; //!< False is a state report; true is a scheduling command.
    Id node{};
    uint64_t epoch{};
    uint32_t sequence{};
    Ns sampledAt{};
    wire::Value payload;
};

/** Actual send/receive times and bytes from one control frame's event path. */
struct ControlDelivery
{
    ControlMessage message;
    Ns startedAt{};
    Ns endedAt{};
    Ns receivedAt{};
    Bytes payloadBytes{};
    Bytes wireBytes{};
};

/** Removal-only decision at one common execution event, distinct from DATA/ACK outcomes. */
struct ControlExecution
{
    Action execution;
    Ns intendedAt{};
    Ns evaluatedAt{};
    std::map<Link, std::string> reasons;
    std::map<Id, Ns> commandReceipts;
};

/**
 * Finite-rate serialized control packets over independent reliable ordered pipes.
 * This is an explicit dedicated-control-channel abstraction, not a data-channel MAC.
 * Configure once; the caller owns Run/Destroy and receives decoded packet bytes.
 */
class ControlChannel : public Object
{
  public:
    static TypeId GetTypeId();
    /** Register a separate ground-station object and the stable UAV identities. */
    void Configure(Ptr<Node> ground, const std::map<Id, Ptr<Node>>& nodes,
                   const ControlSettings& settings);
    /**
     * Queue a whole serialized frame; actual callbacks use simulation time, never wall time.
     * @param message Owned value, validated before scheduling any event.
     * @param at Earliest transmission time, integer ns, at or after Now().
     * @param receiver Callback after finite transmission and propagation, with decoded bytes.
     * @return Predicted deterministic arrival for this fixed reliable pipe, not a service fact.
     */
    Ns Send(const ControlMessage& message, Ns at,
            const std::function<void(const ControlDelivery&)>& receiver);
    /**
     * Send actual endpoint commands and classify the plan at a single execution event.
     * The independent physical callback is evaluated only at that event; it is not policy input.
     * Global deadline miss takes precedence over command miss, then physical invalidity.
     * Late receipts may be logged but can never re-authorize DATA after the common event.
     */
    void ScheduleCommandWindow(
        const Action& plan, Ns createdAt, Ns executeAt, Ns deadline,
        const std::function<std::set<Link>()>& physicalLinks,
        const std::function<void(const ControlExecution&)>& execute,
        const std::function<void(const ControlDelivery&)>& receipt = {});
    /** @return Number of queued/in-flight control frames owned by this object. */
    uint64_t Outstanding() const;

  protected:
    void DoDispose() override;

  private:
    ControlSettings m_settings;
    Ptr<Node> m_ground;
    std::map<Id, Ptr<Node>> m_nodes;
    std::map<std::pair<bool, Id>, Ns> m_freeAt;
    std::map<std::pair<bool, Id>, uint32_t> m_sequences;
    std::set<uint64_t> m_planEpochs;
    std::vector<EventId> m_events;
    uint64_t m_outstanding{};
    bool m_disposed{false};
};
} // namespace ns3::fanet
#endif
