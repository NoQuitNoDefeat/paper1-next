#ifndef FANET_BRIDGE_WIRE_H
#define FANET_BRIDGE_WIRE_H

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace ns3::fanet::wire
{
/** Owned canonical tree; no native structure layout is used on the wire. */
struct Value
{
    using List = std::vector<Value>;
    using Map = std::map<std::string, Value>;
    std::variant<std::monostate, bool, uint64_t, int64_t, double, std::string, List, Map> data;
    Value() = default;
    template <typename T> Value(T value) : data(std::move(value))
    {
    }
    Value(const char* value) : data(std::string(value))
    {
    }
    const Map& Object(std::initializer_list<const char*> fields) const;
    const List& Array() const;
    const Value& At(const std::string& key) const;
    uint64_t U() const;
    int64_t I() const;
    double D() const;
    bool B() const;
    const std::string& S() const;
    bool Null() const;
};
using Map = Value::Map;
using List = Value::List;
constexpr std::size_t HEADER_BYTES = 96;
constexpr uint16_t WIRE_VERSION = 2;
enum Kind : uint16_t
{
    HELLO = 1, READY, INIT, STATE, PLAN, RESULT, STOP, FINAL, ACK, CLOSED, ERROR
};
/** Fixed 96-byte header followed by exactly one canonical value. */
struct Message
{
    Kind kind{};
    std::array<uint8_t, 16> run{};
    uint64_t sequence{};
    uint64_t epoch{};
    int64_t sampledAt{};
    std::array<uint8_t, 32> digest{};
    Value payload;
};
/** Encode a tree into an independently owned bounded byte buffer. */
std::vector<uint8_t> EncodeValue(const Value& value, std::size_t limit);
/** Decode exact canonical bytes; trailing data, unknown tags and nonfinite values fail. */
Value DecodeValue(const std::vector<uint8_t>& bytes);
std::vector<uint8_t> Encode(const Message& message, std::size_t capacity);
Message Decode(const std::vector<uint8_t>& bytes, std::size_t capacity);
std::string RunString(const std::array<uint8_t, 16>& run);
} // namespace ns3::fanet::wire
#endif
