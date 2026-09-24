#ifndef FANET_BRIDGE_RECORDS_H
#define FANET_BRIDGE_RECORDS_H
#include "bridge-wire.h"
#include "controlled-state.h"
#include "scheduled-radio.h"
#include "nonfrozen-controller.h"

namespace ns3::fanet::wire
{
/** INIT state and exogenous inputs, decoded completely before Reset. */
struct Initialization
{
    Config config;
    State state;
    std::vector<Frame> frames;
    List physical;
    Value metadata;
    std::optional<RadioSettings> radioSettings;
    std::vector<RadioFrame> radioFrames;
    std::optional<MotionSettings> radioMotion;
    std::optional<CollectionSettings> control;
    std::string executionProfile{"controlled-budget-v1"};
};

Initialization ReadInitialization(const Value& value);
/** Independent complete-interference execution predicate; never a policy decoder. */
std::set<Link> FailedSinrLinks(const Value& physical, const Action& action);
/** Conservative complete-result byte/count preflight, matching the documented Python bound. */
uint64_t RequiredResultCapacity(const Initialization& inputs, uint32_t capacity);
Action ReadPlan(const Value& value, const Reference& reference, Ns period, bool nonfrozen = false);
Value WriteObservation(const Observation& value);
Value WriteCycle(const CycleResult& value);
Value WritePhysical(const Value& source, const Reference& reference);
Value WriteReference(const Reference& reference);
Value WriteRadioEvents(const std::vector<RadioEvent>& events);
Value WriteReceptionSamples(const std::vector<ReceptionSample>& samples);
Value WriteExecutionChannel(const ExecutionChannel& sample);
} // namespace ns3::fanet::wire
#endif
