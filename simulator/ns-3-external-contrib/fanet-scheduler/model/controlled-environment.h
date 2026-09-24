#ifndef FANET_CONTROLLED_ENVIRONMENT_H
#define FANET_CONTROLLED_ENVIRONMENT_H

#include "controlled-state.h"

#include "ns3/event-id.h"
#include "ns3/node.h"
#include "ns3/object.h"

#include <array>
#include <exception>
#include <functional>
#include <map>
#include <optional>
#include <set>

namespace ns3::fanet
{
/** @return Compiled module digest, locked ns-3 commit and compiler identity, respectively. */
std::array<std::string, 3> ControlledBuildInfo();

/** Independent deterministic research state; no WifiMac, Python state or shared memory. */
class ControlledEnvironment : public Object
{
  public:
    /** @return Registered ns-3 type. */
    static TypeId GetTypeId();
    /**
     * Replace a settled state only after validation succeeds; simulation time is not advanced.
     * @param config Explicit fixed configuration.
     * @param state Explicit settled state, including empty containers.
     * @param run Fresh printable ASCII token (1-128 bytes); never reused on this object.
     * @param trajectory Optional complete private exogenous input; no expected packet outcomes.
     */
    void Reset(const Config& config,
               const State& state,
               const std::string& run,
               const std::optional<std::vector<Frame>>& trajectory = std::nullopt);
    /** @return Independent value snapshot; expected, when supplied, must match exactly. */
    Observation Observe(const std::optional<Reference>& expected = std::nullopt) const;
    /** @return ns-3 Node for a stable research ID; no assumption about Node::GetId(). */
    Ptr<Node> GetNode(Id id) const;
    /** Preflight a complete action without consuming the private frame or changing state. */
    void ValidateAction(const Action& action) const;
    /**
     * Schedule one controlled cycle; caller owns Simulator::Run/Destroy.
     * @param action Complete queue plan bound to the current reference.
     * @param boundaryFirst Register closing/timer callbacks before fact callbacks for diagnostics.
     * Result() is unavailable until all owned work and the boundary request have completed.
     */
    void StartCycle(const Action& action,
                    bool boundaryFirst = true,
                    const std::function<void()>& externalExecutor = {},
                    const std::set<Link>& failedLinks = {});
    /** Begin a control-collection cycle with no authorized DATA; births still enter pending. */
    Observation BeginExternalCollection();
    /** Return only the immutable original sample while an external cycle is active. */
    Observation ExternalSample(const Reference& reference) const;
    /** Install one complete plan after control collection, before any external service. */
    void AcceptExternalPlan(const Action& action);
    /** Validate a removal-only execution subset of the accepted plan without exposing work state. */
    void ValidateExternalExecution(const Action& action) const;
    /** Apply one confirmed external FIFO service while the external execution ticket is held. */
    void ConfirmExternalService(Link key, Id packet);
    /** Release the external ticket only after all radio events and callbacks have drained. */
    void FinishExternalExecution();
    /** @return Complete value result; execution failure is rethrown, never zero-filled. */
    CycleResult Result() const;
    /** @return True only while this environment owns an unfinished cycle. */
    bool IsBusy() const;

  protected:
    void DoDispose() override;
    /** Apply an executor fact; guarded wrapper detects duplicates before a second mutation. */
    void CompleteService(std::size_t queueIndex, Id packet, std::size_t order);
    /** Execute an owned callback and convert unexpected exceptions to a failed episode. */
    void Guard(const std::function<void()>& work);

  private:
    void ValidateAgainstSample(const Action& action, const Observation& observation) const;
    void ScheduleWork(Ns at, const std::string& stage, const std::function<void()>& work);
    void FinishIfReady();
    void Settle();
    void RemoveEvents();
    void Record(const std::string& stage);
    std::optional<Observation> m_observation;
    std::map<Id, Ptr<Node>> m_nodes;
    std::vector<std::string> m_runs;
    bool m_disposed{false};
    bool m_busy{false};
    bool m_boundaryRequested{false};
    bool m_external{false};
    bool m_awaitingPlan{false};
    uint64_t m_outstanding{0};
    std::exception_ptr m_failure;
    std::optional<std::vector<Frame>> m_trajectory;
    std::optional<CycleResult> m_result;
    State m_work;
    std::vector<Arrival> m_pending;
    std::vector<EventId> m_events;
    std::vector<Route> m_admissionRoutes;
    std::map<std::pair<std::size_t, Id>, Ns> m_serviceTimes;
    std::size_t m_order{0};
};
} // namespace ns3::fanet
#endif
