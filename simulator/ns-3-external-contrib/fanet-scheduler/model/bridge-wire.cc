#include "bridge-wire.h"

#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ns3::fanet::wire
{
namespace
{
void Require(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::invalid_argument(message);
    }
}
class Writer
{
  public:
    std::vector<uint8_t> bytes;
    explicit Writer(std::size_t limit) : m_limit(limit)
    {
    }
    void U(uint64_t value, std::size_t n)
    {
        Require(n <= m_limit - bytes.size(), "wire capacity exceeded");
        for (std::size_t i = 0; i < n; ++i)
        {
            bytes.push_back(static_cast<uint8_t>(value >> (8 * i)));
        }
    }
    void Item(const Value& value, unsigned depth = 0)
    {
        Require(depth <= 24, "wire nesting limit");
        const auto& d = value.data;
        if (value.Null())
        {
            U(0, 1);
        }
        else if (auto p = std::get_if<bool>(&d))
        {
            U(*p ? 2 : 1, 1);
        }
        else if (auto p = std::get_if<uint64_t>(&d))
        {
            U(3, 1); U(*p, 8);
        }
        else if (auto p = std::get_if<int64_t>(&d))
        {
            Require(*p >= 0, "negative wire time");
            U(4, 1); U(static_cast<uint64_t>(*p), 8);
        }
        else if (auto p = std::get_if<double>(&d))
        {
            Require(std::isfinite(*p), "nonfinite wire real");
            U(5, 1); U(std::bit_cast<uint64_t>(*p), 8);
        }
        else if (auto p = std::get_if<std::string>(&d))
        {
            Require(p->size() <= 1024, "wire string limit");
            U(6, 1); U(p->size(), 4);
            for (unsigned char ch : *p)
            {
                Require(ch >= 32 && ch <= 126, "wire string must be printable ASCII");
                U(ch, 1);
            }
        }
        else if (auto p = std::get_if<List>(&d))
        {
            Require(p->size() <= 100000, "wire array limit");
            U(7, 1); U(p->size(), 4);
            for (const auto& child : *p)
            {
                Item(child, depth + 1);
            }
        }
        else
        {
            const auto& object = std::get<Map>(d);
            Require(object.size() <= 256, "wire object limit");
            U(8, 1); U(object.size(), 4);
            for (const auto& [key, child] : object)
            {
                Item(Value(key), depth + 1); Item(child, depth + 1);
            }
        }
    }
  private:
    std::size_t m_limit;
};
class Reader
{
  public:
    explicit Reader(const std::vector<uint8_t>& bytes) : m_bytes(bytes)
    {
    }
    uint64_t U(std::size_t n)
    {
        Require(n <= m_bytes.size() - m_pos, "truncated wire bytes");
        uint64_t value = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            value |= static_cast<uint64_t>(m_bytes[m_pos++]) << (8 * i);
        }
        return value;
    }
    std::size_t Count(std::size_t maximum)
    {
        const auto n = U(4);
        Require(n <= maximum && n <= m_bytes.size() - m_pos, "wire length/count limit");
        return n;
    }
    Value Item(unsigned depth = 0)
    {
        Require(depth <= 24, "wire nesting limit");
        switch (U(1))
        {
        case 0: return {};
        case 1: return false;
        case 2: return true;
        case 3: return U(8);
        case 4:
        {
            auto value = U(8);
            Require(value <= INT64_MAX, "negative wire time");
            return static_cast<int64_t>(value);
        }
        case 5:
        {
            auto value = std::bit_cast<double>(U(8));
            Require(std::isfinite(value), "nonfinite wire real");
            return value;
        }
        case 6:
        {
            std::string text;
            auto n = Count(1024);
            for (std::size_t i = 0; i < n; ++i)
            {
                auto ch = U(1);
                Require(ch >= 32 && ch <= 126, "wire ASCII string");
                text.push_back(static_cast<char>(ch));
            }
            return text;
        }
        case 7:
        {
            List list;
            auto n = Count(100000);
            for (std::size_t i = 0; i < n; ++i)
            {
                list.push_back(Item(depth + 1));
            }
            return list;
        }
        case 8:
        {
            Map object;
            auto n = Count(256);
            for (std::size_t i = 0; i < n; ++i)
            {
                auto key = Item(depth + 1).S();
                Require(object.empty() || key > object.rbegin()->first,
                        "wire keys must be unique and sorted");
                object.emplace(key, Item(depth + 1));
            }
            return object;
        }
        default: throw std::invalid_argument("unknown wire tag");
        }
    }
    void End()
    {
        Require(m_pos == m_bytes.size(), "trailing wire bytes");
    }
  private:
    const std::vector<uint8_t>& m_bytes;
    std::size_t m_pos{};
};
} // namespace

const Map& Value::Object(std::initializer_list<const char*> fields) const
{
    const auto& object = std::get<Map>(data);
    Require(object.size() == fields.size(), "missing/unknown record fields");
    for (const auto* field : fields)
    {
        Require(object.contains(field), "missing field");
    }
    return object;
}
const List& Value::Array() const
{
    return std::get<List>(data);
}
const Value& Value::At(const std::string& key) const
{
    return std::get<Map>(data).at(key);
}
uint64_t Value::U() const
{
    return std::get<uint64_t>(data);
}
int64_t Value::I() const
{
    return std::get<int64_t>(data);
}
double Value::D() const
{
    return std::get<double>(data);
}
bool Value::B() const
{
    return std::get<bool>(data);
}
const std::string& Value::S() const
{
    return std::get<std::string>(data);
}
bool Value::Null() const
{
    return std::holds_alternative<std::monostate>(data);
}

std::vector<uint8_t> EncodeValue(const Value& value, std::size_t limit)
{
    Writer out(limit); out.Item(value); return std::move(out.bytes);
}
Value DecodeValue(const std::vector<uint8_t>& bytes)
{
    Reader in(bytes); auto value = in.Item(); in.End(); return value;
}
std::vector<uint8_t> Encode(const Message& message, std::size_t capacity)
{
    Require(capacity >= HEADER_BYTES && message.sampledAt >= 0, "wire header range");
    auto body = EncodeValue(message.payload, capacity - HEADER_BYTES);
    Writer out(capacity);
    for (char ch : std::string("FANETAI1"))
    {
        out.U(ch, 1);
    }
    out.U(WIRE_VERSION, 2); out.U(message.kind, 2); out.U(0, 4); out.U(body.size(), 4);
    out.U(HEADER_BYTES, 4);
    for (auto ch : message.run)
    {
        out.U(ch, 1);
    }
    out.U(message.sequence, 8); out.U(message.epoch, 8); out.U(message.sampledAt, 8);
    for (auto ch : message.digest)
    {
        out.U(ch, 1);
    }
    for (auto ch : body)
    {
        out.U(ch, 1);
    }
    return std::move(out.bytes);
}
Message Decode(const std::vector<uint8_t>& bytes, std::size_t capacity)
{
    Require(bytes.size() >= HEADER_BYTES && bytes.size() <= capacity, "wire size limit");
    Reader in(bytes);
    for (char ch : std::string("FANETAI1"))
    {
        Require(in.U(1) == ch, "wire magic");
    }
    Require(in.U(2) == WIRE_VERSION, "wire version");
    Message message;
    auto kind = in.U(2);
    Require(kind >= HELLO && kind <= ERROR, "wire kind");
    message.kind = static_cast<Kind>(kind);
    Require(in.U(4) == 0, "wire flags");
    Require(in.U(4) == bytes.size() - HEADER_BYTES, "wire payload length");
    Require(in.U(4) == HEADER_BYTES, "wire header length");
    for (auto& ch : message.run)
    {
        ch = in.U(1);
    }
    message.sequence = in.U(8); message.epoch = in.U(8);
    auto time = in.U(8);
    Require(time <= INT64_MAX, "wire time overflow");
    message.sampledAt = static_cast<int64_t>(time);
    for (auto& ch : message.digest)
    {
        ch = in.U(1);
    }
    message.payload = in.Item(); in.End(); return message;
}
std::string RunString(const std::array<uint8_t, 16>& run)
{
    std::string text;
    for (auto ch : run)
    {
        text.push_back("0123456789abcdef"[ch >> 4]);
        text.push_back("0123456789abcdef"[ch & 15]);
    }
    return text;
}
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
} // namespace ns3::fanet::wire
