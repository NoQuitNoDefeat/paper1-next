#include "ns3/bridge-records.h"
#include "ns3/bridge-transport.h"
#include "ns3/controlled-environment.h"
#include "ns3/simulator.h"

#include <charconv>
#include <chrono>
#include <iostream>
#include <set>

using namespace ns3::fanet;
using namespace ns3::fanet::wire;

namespace
{
uint32_t
Number(const char* text)
{
    const std::string s(text);
    uint32_t value{};
    auto result = std::from_chars(s.data(), s.data() + s.size(), value);
    if (result.ec != std::errc{} || result.ptr != s.data() + s.size())
    {
        throw std::invalid_argument("invalid numeric option");
    }
    return value;
}

void
Require(bool ok, const char* message)
{
    if (!ok)
    {
        throw std::invalid_argument(message);
    }
}

Message
Receive(Interface& interface, uint32_t capacity)
{
    interface.CppRecvBegin();
    std::vector<uint8_t> bytes;
    try
    {
        bytes = CopyFrom(*interface.GetPy2CppVector());
    }
    catch (...)
    {
        interface.CppRecvEnd();
        throw;
    }
    interface.CppRecvEnd();
    return Decode(bytes, capacity);
}

void
Send(Interface& interface, const Message& message, uint32_t capacity)
{
    const auto bytes = Encode(message, capacity);
    interface.CppSendBegin();
    CopyTo(*interface.GetCpp2PyVector(), bytes);
    interface.CppSendEnd();
}
} // namespace

int
main(int argc, char* argv[])
{
    auto env = ns3::CreateObject<ControlledEnvironment>();
    ns3::Ptr<ScheduledRadio> radio;
    ns3::Ptr<NonfrozenController> controller;
    std::optional<CollectedState> collection;
    std::unique_ptr<Interface> interface;
    std::optional<Message> request;
    uint32_t rx = 0;
    try
    {
        Require(argc >= 4, "require mode, segment and run");
        const std::string mode = argv[1], name = argv[2], run = argv[3];
        ValidateName(name, run);
        if (mode == "--reserve")
        {
            Require(argc == 5, "reserve arguments");
            Reserve(name, run, Number(argv[4]));
            return 0;
        }
        if (mode == "--cleanup")
        {
            Require(argc == 4, "cleanup arguments");
            Cleanup(name, run);
            return 0;
        }
        if (mode == "--probe")
        {
            Require(argc == 4, "probe arguments");
            std::cout << (Exists(name) ? "present" : "absent") << '\n';
            return 0;
        }
        Require(mode == "--serve" && argc == 7, "serve arguments");
        const auto bytes = Number(argv[4]), tx = Number(argv[5]);
        rx = Number(argv[6]);
        VerifyReady(name, run, bytes, tx, rx);
        interface = std::make_unique<Interface>(false,
                                                true,
                                                false,
                                                bytes,
                                                name.c_str(),
                                                ENV_NAME,
                                                ACT_NAME,
                                                SYNC_NAME);
        std::optional<std::array<uint8_t, 32>> digest;
        std::optional<Initialization> inputs;
        uint64_t sequence = 0;
        bool hello = false, final = false;
        while (true)
        {
            request.reset();
            request = Receive(*interface, tx);
            const auto& message = *request;
            Require(RunString(message.run) == run && message.sequence == sequence,
                    "wrong run or duplicate/out-of-order sequence");
            Require(!digest || message.digest == *digest, "configuration identity changed");
            digest = message.digest;
            Require(sequence != UINT64_MAX, "message sequence exhausted");
            ++sequence;
            Message response = message;
            if (!hello)
            {
                Require(message.kind == HELLO && message.epoch == 0, "expected HELLO");
                message.payload.Object({"shm_bytes", "tx_capacity", "rx_capacity"});
                Require(message.payload.At("shm_bytes").U() == bytes &&
                            message.payload.At("tx_capacity").U() == tx &&
                            message.payload.At("rx_capacity").U() == rx,
                        "HELLO capacity mismatch");
                const auto build = ControlledBuildInfo();
                response.kind = READY;
                response.payload = Map{{"source_sha256", build[0]},
                                       {"ns3_commit", build[1]},
                                       {"compiler", build[2]},
                                       {"ns3_ai_commit", FANET_BRIDGE_AI_COMMIT},
                                       {"shm_bytes", uint64_t{bytes}},
                                       {"tx_capacity", uint64_t{tx}},
                                       {"rx_capacity", uint64_t{rx}}};
                hello = true;
            }
            else if (!inputs)
            {
                Require(message.kind == INIT && message.epoch == 0, "expected INIT");
                auto decoded = ReadInitialization(message.payload);
                Require(message.sampledAt == decoded.config.start, "INIT time mismatch");
                Require(RequiredResultCapacity(decoded, rx) <= rx,
                        "rx capacity below conservative result bound");
                env->Reset(decoded.config, decoded.state, run, decoded.frames);
                if (decoded.radioSettings)
                {
                    radio = ns3::CreateObject<ScheduledRadio>();
                    radio->Configure(env, *decoded.radioSettings, decoded.radioFrames,
                                     decoded.radioMotion);
                }
                inputs = std::move(decoded);
                response.kind = STATE;
                const auto state = env->Observe();
                response.payload =
                    Map{{"observation", WriteObservation(state)},
                        {"physical", WritePhysical(inputs->physical.at(0), state.reference)},
                        {"metadata", inputs->metadata}};
                if (inputs->control)
                {
                    controller = ns3::CreateObject<NonfrozenController>();
                    controller->Configure(env, radio, *inputs->control);
                    collection = controller->Collect(inputs->physical.at(0));
                    auto& payload = std::get<Map>(response.payload.data);
                    payload["observation"] = collection->observation;
                    payload["physical"] = collection->physical;
                    payload.emplace("collection", collection->reports);
                }
                if (radio)
                {
                    std::get<Map>(response.payload.data)
                        .emplace("radio_profile", inputs->radioSettings->actualAck
                                                      ? "transaction-ack-v1" : "ideal-spectrum-v1");
                    if (inputs->radioMotion)
                    {
                        std::get<Map>(response.payload.data).emplace(
                            "motion_profile", "continuous-motion-frame-quasistatic-v1");
                    }
                }
            }
            else
            {
                const auto state = collection ? collection->sample : env->Observe();
                Require(message.epoch == state.reference.epoch &&
                            message.sampledAt == state.reference.sampledAt,
                        "stale epoch/time");
                if (final)
                {
                    Require(message.kind == ACK, "expected final acknowledgement");
                    message.payload.Object({});
                    response.kind = CLOSED;
                    response.payload = Map{};
                    Send(*interface, response, rx);
                    break;
                }
                if (message.kind == STOP)
                {
                    message.payload.Object({"reason"});
                    const auto& reason = message.payload.At("reason").S();
                    Require(reason == "external_stop" ||
                                (reason == "range_end" && !state.remaining),
                            "invalid external end reason");
                    response.kind = FINAL;
                    response.payload =
                        Map{{"observation", WriteObservation(state)},
                            {"physical",
                             WritePhysical(inputs->physical.at(state.reference.epoch),
                                           state.reference)},
                            {"terminated", false},
                            {"truncated", true},
                            {"rollout_cut", false},
                            {"reason", reason}};
                    if (controller)
                    {
                        std::get<Map>(response.payload.data).emplace("abandoned_collection",
                                                                    collection.has_value());
                    }
                    final = true;
                }
                else
                {
                    Require(message.kind == PLAN && state.remaining > 0, "expected active PLAN");
                    const auto action =
                        ReadPlan(message.payload, state.reference, inputs->config.period,
                                 controller != nullptr);
                    // Resource and physical-edge checks protect execution, not a second decoder.
                    std::set<Id> endpoints;
                    std::set<Link> edges;
                    const auto& links = inputs->physical.at(state.reference.epoch).At("links");
                    for (const auto& edge : links.Array())
                    {
                        const auto& k = edge.At("key").Array();
                        edges.emplace(k[0].U(), k[1].U());
                    }
                    for (auto [a, b] : action.links)
                    {
                        Require(endpoints.insert(a).second && endpoints.insert(b).second &&
                                    edges.contains({a, b}),
                                "plan resource or physical-edge violation");
                    }
                    if (!controller)
                    {
                        env->ValidateAction(action);
                    }
                    const auto prepareStart = std::chrono::steady_clock::now();
                    auto runStart = prepareStart;
                    std::optional<ControlExecution> execution;
                    if (controller)
                    {
                        const auto& timing = message.payload.At("timing");
                        Require(collection && timing.At("collection_completed_at_ns").I() ==
                                                  collection->cut &&
                                    timing.At("execute_at_ns").I() == AddTime(
                                        timing.At("plan_created_at_ns").I(),
                                        inputs->control->distributionWindow),
                                "plan timing does not match actual collection or common window");
                        execution = controller->Execute(action, timing.At("encode_ns").I(),
                                                        timing.At("decode_ns").I());
                        collection.reset();
                    }
                    else if (radio)
                    {
                        radio->StartCycle(action);
                    }
                    else
                    {
                        const auto failed = inputs->executionProfile == "full-sinr-v1"
                                                ? FailedSinrLinks(
                                                      inputs->physical.at(state.reference.epoch),
                                                      action)
                                                : std::set<Link>{};
                        env->StartCycle(action, true, {}, failed);
                    }
                    if (!controller)
                    {
                        runStart = std::chrono::steady_clock::now();
                        ns3::Simulator::Run();
                    }
                    const auto runEnd = std::chrono::steady_clock::now();
                    const auto result = env->Result();
                    response.kind = RESULT;
                    response.epoch = result.next.reference.epoch;
                    response.sampledAt = result.endedAt;
                    response.payload = Map{
                        {"cycle", WriteCycle(result)},
                        {"physical",
                         WritePhysical(inputs->physical.at(response.epoch), result.next.reference)},
                        {"timing", message.payload.At("timing")}};
                    if (radio)
                    {
                        const auto prepareNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                   runStart - prepareStart)
                                                   .count();
                        const auto runNs =
                            std::chrono::duration_cast<std::chrono::nanoseconds>(runEnd - runStart)
                                .count();
                        std::get<Map>(response.payload.data)
                            .emplace("radio",
                                     Map{{"profile", inputs->radioSettings->actualAck
                                                          ? "transaction-ack-v1" : "ideal-spectrum-v1"},
                                         {"events", WriteRadioEvents(radio->Events())},
                                         {"execution_channel",
                                          WriteExecutionChannel(radio->ExecutionSnapshot())},
                                         {"prepare_wall_ns", static_cast<int64_t>(prepareNs)},
                                         {"simulator_run_wall_ns", static_cast<int64_t>(runNs)}});
                        if (inputs->radioMotion)
                        {
                            auto& report = std::get<Map>(
                                std::get<Map>(response.payload.data).at("radio").data);
                            report.emplace("motion_profile", "continuous-motion-frame-quasistatic-v1");
                            report.emplace("reception_samples",
                                           WriteReceptionSamples(radio->ReceptionSamples()));
                        }
                    }
                    if (controller)
                    {
                        List reasons, receipts, executed;
                        for (const auto& [key, reason] : execution->reasons)
                        {
                            reasons.push_back(Map{{"link", List{key.first, key.second}},
                                                  {"reason", reason}});
                        }
                        for (const auto& key : execution->execution.links)
                        {
                            executed.push_back(List{key.first, key.second});
                        }
                        for (const auto& [id, at] : execution->commandReceipts)
                        {
                            receipts.push_back(Map{{"node_id", id}, {"received_at_ns", at}});
                        }
                        auto& body = std::get<Map>(response.payload.data);
                        body.emplace("control", Map{{"profile", "fresh-complete-control-v1"},
                            {"evaluated_at_ns", execution->evaluatedAt}, {"reasons", reasons},
                            {"executed_links", executed}, {"command_receipts", receipts},
                            {"arrivals", controller->TakeArrivals()}});
                        body.emplace("next_collection", Value{});
                        if (result.next.remaining)
                        {
                            collection = controller->Collect(inputs->physical.at(response.epoch));
                            body["physical"] = collection->physical;
                            std::get<Map>(body.at("cycle").data)["next_observation"] =
                                collection->observation;
                            body["next_collection"] = collection->reports;
                        }
                    }
                }
            }
            Send(*interface, response, rx);
        }
        if (controller)
        {
            controller->Dispose();
        }
        if (radio)
        {
            radio->Dispose();
        }
        env->Dispose();
        ns3::Simulator::Destroy();
        return 0;
    }
    catch (const std::exception& error)
    {
        if (interface && request)
        {
            try
            {
                auto response = *request;
                response.kind = ERROR;
                std::string reason = error.what();
                if (reason.size() > 512)
                {
                    reason.resize(512);
                }
                response.payload = Map{{"code", "invalid_or_failed_exchange"},
                                       {"detail", reason},
                                       {"fatal", true}};
                Send(*interface, response, rx);
            }
            catch (...)
            {
            }
        }
        if (controller)
        {
            controller->Dispose();
        }
        if (radio)
        {
            radio->Dispose();
        }
        env->Dispose();
        ns3::Simulator::Destroy();
        std::cerr << "FANET bridge failed: " << error.what() << '\n';
        return 1;
    }
}
