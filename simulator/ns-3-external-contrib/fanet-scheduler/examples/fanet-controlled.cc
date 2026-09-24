#include "ns3/controlled-environment.h"
#include "ns3/simulator.h"

#include <charconv>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>

using ns3::CreateObject;
using ns3::Simulator;
using namespace ns3::fanet;

namespace
{
// This bounded text fixture transport is not the future ns3-ai wire protocol.
class Reader
{
  public:
    explicit Reader(std::istream& in)
        : m_in(in)
    {
    }
    std::string Token()
    {
        std::string token;
        if (!(m_in >> token) || token.size() > 128)
        {
            throw std::invalid_argument("missing or oversized fixture token");
        }
        return token;
    }
    void Tag(const std::string& expected)
    {
        if (Token() != expected)
        {
            throw std::invalid_argument("expected fixture tag " + expected);
        }
    }
    template <typename T> T Number()
    {
        return Parse<T>(Token());
    }
    template <typename T> std::optional<T> Optional()
    {
        auto token = Token();
        return token == "none" ? std::nullopt : std::optional<T>{Parse<T>(token)};
    }
    bool Flag()
    {
        const auto value = Number<unsigned>();
        if (value > 1)
        {
            throw std::invalid_argument("boolean token must be 0 or 1");
        }
        return value == 1;
    }
    std::size_t Count()
    {
        const auto count = Number<uint64_t>();
        if (count > 100000)
        {
            throw std::invalid_argument("fixture record count exceeds development limit");
        }
        return count;
    }
    Link Key()
    {
        const auto tx = Number<Id>();
        return {tx, Number<Id>()};
    }
    std::vector<Packet> Packets(const std::string& tag)
    {
        Tag(tag);
        std::vector<Packet> values;
        for (auto n = Count(); n > 0; --n)
        {
            values.push_back(
                {Number<Id>(), Number<Id>(), Number<Id>(), Number<Ns>(), Optional<Ns>()});
        }
        return values;
    }
    std::vector<Entry> Entries()
    {
        std::vector<Entry> values;
        for (auto n = Count(); n > 0; --n)
        {
            values.push_back({Number<Id>(), Number<Ns>(), Number<Ns>()});
        }
        return values;
    }
    std::vector<Route> Routes(const std::string& tag)
    {
        Tag(tag);
        std::vector<Route> values;
        for (auto n = Count(); n > 0; --n)
        {
            values.push_back({Number<Id>(), Number<Id>(), Optional<Id>()});
        }
        return values;
    }
    void End()
    {
        std::string extra;
        if (m_in >> extra)
        {
            throw std::invalid_argument("unexpected trailing fixture data");
        }
    }

  private:
    template <typename T> T Parse(const std::string& token)
    {
        T value{};
        const auto result = std::from_chars(token.data(), token.data() + token.size(), value);
        if (result.ec != std::errc{} || result.ptr != token.data() + token.size())
        {
            throw std::invalid_argument("invalid or overflowing integer token");
        }
        return value;
    }
    std::istream& m_in;
};

void
String(std::ostream& out, const std::string& text)
{
    out << '"';
    for (unsigned char ch : text)
    {
        if (ch == '"' || ch == '\\')
        {
            out << '\\' << ch;
        }
        else if (ch < 32 || ch >= 127)
        {
            out << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << unsigned(ch)
                << std::dec << std::setfill(' ');
        }
        else
        {
            out << ch;
        }
    }
    out << '"';
}

template <typename T, typename Writer>
void
Array(std::ostream& out, const std::vector<T>& values, Writer writer)
{
    out << '[';
    bool first = true;
    for (const auto& value : values)
    {
        if (!first)
        {
            out << ',';
        }
        first = false;
        writer(out, value);
    }
    out << ']';
}

void
Key(std::ostream& out, Link key)
{
    out << '[' << key.first << ',' << key.second << ']';
}

void
Integer(std::ostream& out, Id id)
{
    out << id;
}

template <typename T, typename Writer>
void
Optional(std::ostream& out, const std::optional<T>& value, Writer writer)
{
    if (value)
    {
        writer(out, *value);
    }
    else
    {
        out << "null";
    }
}

void
Ref(std::ostream& out, const Reference& ref)
{
    out << "{\"episode_id\":";
    String(out, ref.run);
    out << ",\"period_index\":" << ref.epoch << ",\"sampled_at_ns\":" << ref.sampledAt
        << ",\"stage\":\"decision\"}";
}

void
Packets(std::ostream& out, const std::vector<Packet>& packets)
{
    Array(out, packets, [](auto& stream, const auto& packet) {
        stream << "{\"packet_id\":" << packet.id << ",\"source\":" << packet.source
               << ",\"destination\":" << packet.destination << ",\"born_at_ns\":" << packet.born
               << ",\"deadline_ns\":null}";
    });
}

void
Containers(std::ostream& out, const State& state, const Config& config, Ns sampledAt)
{
    out << "\"queues\":";
    std::size_t i = 0;
    Array(out, state.queues, [&](auto& stream, const auto& queue) {
        stream << "{\"key\":";
        Key(stream, queue.key);
        stream << ",\"capacity_bytes\":" << config.queues.at(i++).capacity << ",\"entries\":";
        Array(stream, queue.entries, [](auto& s, const auto& e) {
            s << "{\"packet_id\":" << e.packet << ",\"node_arrived_at_ns\":" << e.arrived
              << ",\"enqueued_at_ns\":" << e.entered << '}';
        });
        stream << ",\"occupancy_bytes\":" << Occupancy(queue.entries.size(), config.packetBytes)
               << ",\"hol_ns\":"
               << (queue.entries.empty() ? 0 : sampledAt - queue.entries[0].entered)
               << '}';
    });
    out << ",\"waiting_areas\":";
    i = 0;
    Array(out, state.waiting, [&](auto& stream, const auto& area) {
        const auto& limits = config.waiting.at(i++);
        stream << "{\"node_id\":" << area.node << ",\"capacity_bytes\":" << limits.capacity
               << ",\"max_wait_ns\":" << limits.maxWait << ",\"entries\":";
        Array(stream, area.entries, [](auto& s, const auto& e) {
            s << "{\"packet_id\":" << e.packet << ",\"node_arrived_at_ns\":" << e.arrived
              << ",\"waiting_since_ns\":" << e.entered << '}';
        });
        stream << ",\"occupancy_bytes\":"
               << Occupancy(area.entries.size(), config.packetBytes) << '}';
    });
    out << ",\"packets\":";
    Packets(out, state.packets);
}

void
Observe(std::ostream& out, const Observation& observation)
{
    out << "{\"reference\":";
    Ref(out, observation.reference);
    out << ",\"kind\":\"queue_state\",\"node_ids\":";
    Array(out, observation.config.nodes, Integer);
    out << ",\"packet_size_bytes\":" << observation.config.packetBytes << ',';
    Containers(out, observation.state, observation.config, observation.reference.sampledAt);
    out << ",\"routes\":";
    Array(out, observation.state.routes, [](auto& s, const auto& route) {
        s << "{\"node_id\":" << route.node << ",\"destination\":" << route.destination
          << ",\"next_hop\":";
        Optional(s, route.nextHop, Integer);
        s << '}';
    });
    out << ",\"pending_packet_ids\":[],\"periods_remaining\":" << observation.remaining << '}';
}

void
Cycle(std::ostream& out, const CycleResult& result)
{
    out << "{\"action\":{\"reference\":";
    Ref(out, result.action.reference);
    out << ",\"links\":";
    Array(out, result.action.links, Key);
    out << "},\"ended_at_ns\":" << result.endedAt << ",\"services\":";
    Array(out, result.services, [](auto& s, const auto& service) {
        s << "{\"key\":";
        Key(s, service.key);
        s << ",\"packet_ids\":";
        Array(s, service.packets, Integer);
        s << ",\"bytes_served\":" << service.bytes << '}';
    });
    out << ",\"events\":";
    Array(out, result.events, [](auto& s, const auto& event) {
        s << "{\"kind\":";
        String(s, event.kind);
        s << ",\"packet_id\":" << event.packet << ",\"at_ns\":" << event.at
          << ",\"node_id\":" << event.node << ",\"link\":";
        Optional(s, event.link, Key);
        s << ",\"origin\":";
        Optional(s, event.origin, String);
        s << ",\"terminal_class\":";
        Optional(s, event.terminalClass, String);
        s << ",\"reason\":";
        Optional(s, event.reason, String);
        s << ",\"end_to_end_ns\":";
        Optional(s, event.endToEnd, [](auto& stream, Ns value) { stream << value; });
        s << '}';
    });
    out << ",\"post_service\":{\"reference\":";
    Ref(out, result.post.reference);
    out << ",\"sampled_at_ns\":" << result.post.sampledAt << ",\"stage\":\"post_service\",";
    Containers(out, result.post.state, result.next.config, result.post.sampledAt);
    out << ",\"pending\":";
    Array(out, result.post.pending, [](auto& s, const auto& arrival) {
        s << "{\"packet_id\":" << arrival.packet << ",\"node_id\":" << arrival.node
          << ",\"node_arrived_at_ns\":" << arrival.arrived << ",\"origin\":";
        String(s, arrival.origin);
        s << '}';
    });
    out << "},\"initial_ids\":";
    Array(out, result.initialIds, Integer);
    out << ",\"new_source_ids\":";
    Array(out, result.newIds, Integer);
    out << ",\"risk_ids\":";
    Array(out, result.riskIds, Integer);
    out << ",\"next_observation\":";
    Observe(out, result.next);
    out << ",\"reward_status\":\"not_computed\"}";
}
} // namespace

int
main(int argc, char* argv[])
{
    auto env = CreateObject<ControlledEnvironment>();
    try
    {
        bool boundaryFirst = true;
        if (argc == 2 && std::string(argv[1]) == "--boundary-last")
        {
            boundaryFirst = false;
        }
        else if (argc != 1)
        {
            throw std::invalid_argument("unknown controlled fixture option");
        }
        std::string input;
        for (char ch; std::cin.get(ch);)
        {
            if (input.size() >= 8 * 1024 * 1024)
            {
                throw std::invalid_argument("development fixture exceeds 8 MiB");
            }
            input.push_back(ch);
        }
        std::istringstream source(input);
        Reader read(source);
        read.Tag("FANET_CONTROLLED_V1");
        read.Tag("RUN");
        const auto run = read.Token();
        read.Tag("CONFIG");
        Config config;
        config.packetBytes = read.Number<Bytes>();
        config.start = read.Number<Ns>();
        config.end = read.Number<Ns>();
        config.period = read.Number<Ns>();
        config.packetDeadlines = read.Flag();
        config.retransmissions = read.Flag();
        read.Tag("NODES");
        for (auto n = read.Count(); n > 0; --n)
        {
            config.nodes.push_back(read.Number<Id>());
        }
        read.Tag("QUEUES");
        for (auto n = read.Count(); n > 0; --n)
        {
            config.queues.push_back({read.Key(), read.Number<Bytes>()});
        }
        read.Tag("WAITS");
        for (auto n = read.Count(); n > 0; --n)
        {
            config.waiting.push_back({read.Number<Id>(), read.Number<Bytes>(), read.Number<Ns>()});
        }
        State initial;
        initial.packets = read.Packets("PACKETS");
        read.Tag("QUEUE_STATE");
        for (auto n = read.Count(); n > 0; --n)
        {
            initial.queues.push_back({read.Key(), read.Entries()});
        }
        read.Tag("WAIT_STATE");
        for (auto n = read.Count(); n > 0; --n)
        {
            initial.waiting.push_back({read.Number<Id>(), read.Entries()});
        }
        initial.routes = read.Routes("ROUTES");
        read.Tag("TRAJECTORY");
        const bool hasTrajectory = read.Flag();
        std::optional<std::vector<Frame>> frames;
        if (hasTrajectory)
        {
            frames.emplace();
            for (auto n = read.Count(); n > 0; --n)
            {
                read.Tag("FRAME");
                Frame frame;
                frame.start = read.Number<Ns>();
                read.Tag("SERVICES");
                for (auto count = read.Count(); count > 0; --count)
                {
                    frame.services.push_back({read.Key(), read.Number<Bytes>(), read.Number<Ns>(),
                                              read.Flag(), read.Optional<Bytes>()});
                }
                frame.births = read.Packets("BIRTHS");
                frame.routeUpdates = read.Routes("UPDATES");
                frames->push_back(std::move(frame));
            }
        }
        read.Tag("PLANS");
        std::vector<Action> plans;
        for (auto n = read.Count(); n > 0; --n)
        {
            read.Tag("PLAN");
            Action action{{read.Token(), read.Number<uint64_t>(), read.Number<Ns>()}, {}};
            for (auto count = read.Count(); count > 0; --count)
            {
                action.links.push_back(read.Key());
            }
            plans.push_back(std::move(action));
        }
        read.End();
        env->Reset(config, initial, run, frames);
        const auto first = env->Observe();
        std::vector<CycleResult> results;
        for (const auto& action : plans)
        {
            env->StartCycle(action, boundaryFirst);
            Simulator::Run();
            results.push_back(env->Result());
        }
        const auto build = ControlledBuildInfo();
        std::ostringstream out;
        out << "{\"schema_version\":1,\"scope\":\"controlled_research_alignment\",\"build\":{";
        out << "\"source_scope\":\"component-cpp-h-cmake-v1\",\"source_sha256\":";
        String(out, build[0]);
        out << ",\"ns3_commit\":";
        String(out, build[1]);
        out << ",\"compiler\":";
        String(out, build[2]);
        out << "},\"initial_observation\":";
        Observe(out, first);
        out << ",\"cycles\":";
        Array(out, results, Cycle);
        out << ",\"callbacks\":";
        Array(out, results, [](auto& s, const auto& result) {
            Array(s, result.callbacks, [](auto& stream, const auto& record) {
                stream << "{\"time_ns\":" << record.at << ",\"stage\":";
                String(stream, record.stage);
                stream << ",\"outstanding\":" << record.outstanding << '}';
            });
        });
        out << ",\"final_observation\":";
        Observe(out, env->Observe());
        out << ",\"simulator_time_ns\":" << Simulator::Now().GetNanoSeconds() << "}\n";
        env->Dispose();
        Simulator::Destroy();
        // Publish only after the entire requested run, including final observation, succeeded.
        std::cout << out.str();
        return 0;
    }
    catch (const std::exception& error)
    {
        env->Dispose();
        Simulator::Destroy();
        std::cerr << "controlled environment failed: " << error.what() << '\n';
        return 1;
    }
}
