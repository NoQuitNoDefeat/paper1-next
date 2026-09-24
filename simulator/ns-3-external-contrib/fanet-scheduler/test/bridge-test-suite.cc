#include "ns3/bridge-records.h"
#include "ns3/bridge-wire.h"
#include "ns3/bridge-transport.h"
#include "ns3/test.h"

#include <functional>
#include <iomanip>
#include <sstream>
#include <unistd.h>

using namespace ns3;
using namespace ns3::fanet::wire;
namespace
{
bool Rejects(const std::function<void()>& fn)
{
    try { fn(); }
    catch (const std::exception&)
    {
        return true;
    }
    return false;
}
class WireCase : public TestCase
{
  public:
    explicit WireCase(unsigned which) : TestCase("bridge wire case " + std::to_string(which)),
                                        m_which(which)
                                        {
                                        }
  private:
    void DoRun() override
    {
        if (m_which == 0)
        {
            const std::vector<uint8_t> expected{7, 2, 0, 0, 0, 1, 3, 2, 0, 0, 0, 0, 0, 0, 0};
            const auto actual = EncodeValue(List{false, uint64_t{2}}, 1024);
            NS_TEST_ASSERT_MSG_EQ(actual == expected, true, "Independent canonical byte answer");
            const auto list = DecodeValue(expected).Array();
            NS_TEST_ASSERT_MSG_EQ(list[0].B(), false, "Boolean is distinct from ID zero");
            NS_TEST_ASSERT_MSG_EQ(list[1].U(), 2, "Little endian integer");
        }
        if (m_which == 1)
        {
            Message message;
            message.kind = PLAN; message.sequence = UINT64_MAX; message.epoch = UINT64_MAX;
            message.sampledAt = INT64_MAX; message.payload = Map{};
            const auto bytes = Encode(message, 101);
            NS_TEST_ASSERT_MSG_EQ(bytes.size(), 101, "Exact capacity accepted");
            NS_TEST_ASSERT_MSG_EQ(bytes[8], 2, "Explicit extended protocol version");
            auto legacy = bytes;
            legacy[8] = 1;
            NS_TEST_ASSERT_MSG_EQ(Rejects([&] { Decode(legacy, 101); }), true,
                                  "Old protocol cannot silently consume new payload semantics");
            NS_TEST_ASSERT_MSG_EQ(bytes[16], 5, "Payload length offset");
            NS_TEST_ASSERT_MSG_EQ(bytes[20], 96, "Header length offset");
            NS_TEST_ASSERT_MSG_EQ(bytes[63], 127, "Signed time byte order");
            const auto restored = Decode(bytes, 101);
            NS_TEST_ASSERT_MSG_EQ(restored.sequence, UINT64_MAX, "Exact unsigned sequence");
            NS_TEST_ASSERT_MSG_EQ(restored.sampledAt, INT64_MAX, "Exact time");
            NS_TEST_ASSERT_MSG_EQ(Rejects([&] { Encode(message, 100); }), true,
                                  "One byte below message capacity rejected");
        }
        if (m_which == 2)
        {
            for (const auto& value : std::vector<std::vector<uint8_t>>{
                {}, {9}, {0, 0}, {3, 0}, {7, 255, 255, 255, 255},
                {4, 255, 255, 255, 255, 255, 255, 255, 255},
                {5, 0, 0, 0, 0, 0, 0, 248, 127}})
            {
                NS_TEST_ASSERT_MSG_EQ(Rejects([&] { DecodeValue(value); }), true,
                                      "Malformed/overflow/nonfinite value rejected");
            }
        }
        if (m_which == 3)
        {
            Message message;
            message.kind = HELLO; message.payload = Map{};
            const auto original = Encode(message, 1024);
            for (auto offset : {0, 8, 10, 12, 16, 20, 63})
            {
                auto bytes = original; bytes[offset] = 255;
                NS_TEST_ASSERT_MSG_EQ(Rejects([&] { Decode(bytes, 1024); }), true,
                                      "Invalid header rejected");
            }
        }
        if (m_which == 4)
        {
            Value value = Map{{"n", uint64_t{0}}};
            NS_TEST_ASSERT_MSG_EQ(Rejects([&] { value.Object({}); }), true, "Unknown field");
            NS_TEST_ASSERT_MSG_EQ(Rejects([&] { value.Object({"n", "missing"}); }), true,
                                  "Missing field");
            NS_TEST_ASSERT_MSG_EQ(Rejects([&] { value.At("n").I(); }), true, "Signedness is typed");
        }
        if (m_which == 5)
        {
            NS_TEST_ASSERT_MSG_EQ(Rejects([] { ValidateAllocation(16384, 8192, 8192); }), true,
                                  "Both vectors and overhead must fit");
            NS_TEST_ASSERT_MSG_EQ(Rejects([] { ValidateAllocation(4096, 4096, 4096); }), true,
                                  "Too small segment");
        }
    }
    unsigned m_which;
};
class ResourceCase : public TestCase
{
  public:
    ResourceCase() : TestCase("bridge resource ownership and preallocated byte vectors")
    {
    }
  private:
    void DoRun() override
    {
        std::ostringstream token;
        token << std::hex << std::setfill('0') << std::setw(16) << getpid()
              << "0000000000000001";
        const auto run = token.str();
        const auto name = "p1ai_" + run.substr(0, 20);
        Reserve(name, run, 65536);
        try
        {
            NS_TEST_ASSERT_MSG_EQ(Rejects([&] { Reserve(name, run, 65536); }), true,
                                  "Existing resource is not replaced");
            const auto other = run.substr(0, 31) + "2";
            NS_TEST_ASSERT_MSG_EQ(Rejects([&] { Cleanup(name, other); }), true,
                                  "Wrong owner cannot remove segment");
            Prepare(name, run, 65536, 16384, 16384);
            VerifyReady(name, run, 65536, 16384, 16384);
            bip::managed_shared_memory segment(bip::open_only, name.c_str());
            const auto free = segment.get_free_memory();
            auto* vector = segment.find<Interface::Cpp2PyMsgVector>(ENV_NAME).first;
            Message message; message.kind = HELLO; message.payload = Map{};
            const auto small = Encode(message, 16384);
            CopyTo(*vector, small);
            NS_TEST_ASSERT_MSG_EQ(CopyFrom(*vector) == small, true, "Owned exact byte copy");
            List full(15, Value(std::string(1024, 'x')));
            full.emplace_back(std::string(843, 'y'));
            message.payload = full;
            const auto maximum = Encode(message, 16384);
            NS_TEST_ASSERT_MSG_EQ(maximum.size(), 16384, "Exact allocated capacity");
            CopyTo(*vector, maximum);
            NS_TEST_ASSERT_MSG_EQ(CopyFrom(*vector) == maximum, true, "Full byte vector copied");
            auto tooLarge = maximum;
            tooLarge.push_back(0);
            NS_TEST_ASSERT_MSG_EQ(Rejects([&] { CopyTo(*vector, tooLarge); }), true,
                                  "One byte over allocated vector rejected");
            NS_TEST_ASSERT_MSG_EQ(CopyFrom(*vector) == maximum, true,
                                  "Failed send leaves previous vector unchanged");
            auto shortened = small; shortened.pop_back(); CopyTo(*vector, shortened);
            NS_TEST_ASSERT_MSG_EQ(Rejects([&] { CopyFrom(*vector); }), true,
                                  "Truncated transfer cannot read a stale tail");
            NS_TEST_ASSERT_MSG_EQ(segment.get_free_memory(), free,
                                  "Logical vector resize never reallocates reserved storage");
        }
        catch (...)
        {
            Cleanup(name, run); throw;
        }
        Cleanup(name, run); Cleanup(name, run);
        NS_TEST_ASSERT_MSG_EQ(Exists(name), false, "Owned resource removed idempotently");
    }
};
class ResultPopulationCase : public TestCase
{
  public:
    ResultPopulationCase()
        : TestCase("result capacity bounds a cycle rather than cumulative traffic")
    {
    }

  private:
    void DoRun() override
    {
        Initialization inputs;
        inputs.config.packetBytes = 1000;
        inputs.config.nodes = {1, 2};
        inputs.config.queues = {{{1, 2}, 1999}};
        inputs.config.waiting = {{1, 1999, 50000000}, {2, 1999, 50000000}};
        inputs.state.packets.resize(1);
        inputs.state.routes = {{1, 2, 2}};
        inputs.frames.resize(120);
        for (auto& frame : inputs.frames)
        {
            frame.births.resize(175);
            frame.routeUpdates = {{1, 2, 2}};
        }
        inputs.physical = {Map{}};
        // Three whole-packet slots + 175 peak births, one distinct route,
        // one five-byte placeholder physical object for this arithmetic test.
        NS_TEST_ASSERT_MSG_EQ(RequiredResultCapacity(inputs, 2097152),
                              821079,
                              "Whole-packet and unique-route bound");
        inputs.radioSettings = ns3::fanet::RadioSettings{};
        // paper1-next: the radio term also reserves 4096 + 256 * (nodes / 2)^2 bytes
        // (added upstream in adc0be5 without updating these expectations): +4352 here.
        NS_TEST_ASSERT_MSG_EQ(RequiredResultCapacity(inputs, 2097152),
                              1181943,
                              "Radio events use the same cycle population");
        inputs.frames.clear();
        inputs.config.packetBytes = 1;
        inputs.config.queues[0].capacity = UINT64_MAX;
        inputs.config.waiting[0].capacity = UINT64_MAX;
        inputs.config.waiting[1].capacity = UINT64_MAX;
        NS_TEST_ASSERT_MSG_EQ(RequiredResultCapacity(inputs, 2097152),
                              31443,
                              "Huge capacities cannot overflow or exceed one lifetime packet");
    }
};

class SinrExecutionCase : public TestCase
{
  public:
    SinrExecutionCase() : TestCase("independent full interference and inclusive SINR boundary")
    {
    }

  private:
    void DoRun() override
    {
        using namespace ns3::fanet;
        List nodes, links, gains;
        for (Id n = 1; n <= 6; ++n)
        {
            nodes.push_back(Map{{"node_id", n}, {"transmit_power_w", 1.0}});
        }
        for (Link key : {Link{1, 2}, Link{3, 4}, Link{5, 6}})
        {
            links.push_back(Map{{"key", List{key.first, key.second}}, {"sinr_threshold", 3.0}});
            for (Id sender : {Id{1}, Id{3}, Id{5}})
            {
                gains.push_back(Map{{"sender", sender}, {"receiver", key.second},
                                    {"gain", sender == key.first ? (sender == 5 ? 3.0 : 10.0)
                                                               : (key.second == 2 ? 2.0 : 0.0)}});
            }
        }
        const Value physical = Map{{"nodes", nodes}, {"links", links}, {"gains", gains},
                                   {"noise_w", 1.0}};
        NS_TEST_ASSERT_MSG_EQ(FailedSinrLinks(physical, {{}, {{1, 2}, {3, 4}}}).empty(),
                              true, "A pair alone is feasible");
        const std::set<Link> expected{{1, 2}};
        NS_TEST_ASSERT_MSG_EQ(
            FailedSinrLinks(physical, {{}, {{1, 2}, {3, 4}, {5, 6}}}) == expected,
            true, "Cumulative interference fails only victim; exact SINR three succeeds");
    }
};

class BridgeSuite : public TestSuite
{
  public:
    BridgeSuite() : TestSuite("fanet-bridge", Type::UNIT)
    {
        for (unsigned i = 0; i < 6; ++i)
        {
            AddTestCase(new WireCase(i), TestCase::Duration::QUICK);
        }
        AddTestCase(new ResourceCase(), TestCase::Duration::QUICK);
        AddTestCase(new ResultPopulationCase(), TestCase::Duration::QUICK);
        AddTestCase(new SinrExecutionCase(), TestCase::Duration::QUICK);
    }
};
static BridgeSuite suite;
} // namespace
