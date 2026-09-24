#ifndef FANET_BRIDGE_TRANSPORT_H
#define FANET_BRIDGE_TRANSPORT_H
#include "bridge-wire.h"
#include "ns3/ns3-ai-msg-interface.h"

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace ns3::fanet::wire
{
using Interface = Ns3AiMsgInterfaceImpl<uint8_t, uint8_t>;
namespace bip = boost::interprocess;
constexpr const char* ENV_NAME = "fanet_env";
constexpr const char* ACT_NAME = "fanet_act";
constexpr const char* SYNC_NAME = "fanet_sync";
constexpr const char* OWNER_NAME = "fanet_owner";

/** Limits are development allocation bounds, not maximum supported network sizes. */
inline void ValidateAllocation(uint64_t bytes, uint64_t tx, uint64_t rx)
{
    if (bytes < 16384 || bytes > 256 * 1024 * 1024 || tx < 4096 || rx < 4096 ||
        tx > UINT32_MAX || rx > UINT32_MAX || tx + rx + 8192 > bytes)
    {
        throw std::invalid_argument("shared segment/capacity preflight failed");
    }
}
inline void ValidateName(const std::string& name, const std::string& run)
{
    if (run.size() != 32 || run.find_first_not_of("0123456789abcdef") != std::string::npos ||
        name != "p1ai_" + run.substr(0, 20))
    {
        throw std::invalid_argument("invalid owned segment identity");
    }
}
inline bool Exists(const std::string& name)
{
    try
    {
        bip::shared_memory_object object(bip::open_only, name.c_str(), bip::read_only);
        return true;
    }
    catch (const bip::interprocess_exception& error)
    {
        if (error.get_error_code() == bip::not_found_error)
        {
            return false;
        }
        throw;
    }
}
inline void VerifyOwner(bip::managed_shared_memory& segment, const std::string& run)
{
    const auto owner = segment.find<char>(OWNER_NAME);
    if (owner.second != run.size() || !owner.first ||
        !std::equal(run.begin(), run.end(), owner.first))
    {
        throw std::invalid_argument("segment belongs to another owner");
    }
}
/** Atomically reserve without removing an existing segment, unlike upstream creator mode. */
inline void Reserve(const std::string& name, const std::string& run, uint32_t bytes)
{
    ValidateName(name, run);
    if (bytes < 16384 || bytes > 256 * 1024 * 1024)
    {
        throw std::invalid_argument("segment size out of range");
    }
    bool created = false;
    try
    {
        bip::managed_shared_memory segment(bip::create_only, name.c_str(), bytes);
        created = true;
        auto* owner = segment.construct<char>(OWNER_NAME)[run.size()]();
        std::copy(run.begin(), run.end(), owner);
    }
    catch (...)
    {
        if (created)
        {
            bip::shared_memory_object::remove(name.c_str());
        }
        throw;
    }
}
inline void Cleanup(const std::string& name, const std::string& run)
{
    ValidateName(name, run);
    if (!Exists(name))
    {
        return;
    }
    bip::managed_shared_memory segment(bip::open_only, name.c_str());
    VerifyOwner(segment, run);
    if (!bip::shared_memory_object::remove(name.c_str()))
    {
        throw std::runtime_error("owned segment removal failed");
    }
}
/** Reserve fixed vectors once; no shared allocator activity occurs during exchanges. */
inline void Prepare(const std::string& name, const std::string& run,
                    uint32_t bytes, uint32_t tx, uint32_t rx)
{
    ValidateName(name, run); ValidateAllocation(bytes, tx, rx);
    bip::managed_shared_memory segment(bip::open_only, name.c_str());
    VerifyOwner(segment, run);
    if (segment.get_size() != bytes)
    {
        throw std::invalid_argument("segment size mismatch");
    }
    const Interface::Cpp2PyMsgAllocator allocator(segment.get_segment_manager());
    auto* env = segment.construct<Interface::Cpp2PyMsgVector>(ENV_NAME)(allocator);
    auto* act = segment.construct<Interface::Py2CppMsgVector>(ACT_NAME)(allocator);
    env->resize(rx); act->resize(tx);
    segment.construct<Ns3AiMsgSync>(SYNC_NAME)();
    if (env->size() != rx || act->size() != tx || env->capacity() != rx || act->capacity() != tx)
    {
        throw std::runtime_error("allocated vectors disagree with declared capacity");
    }
}
inline void VerifyReady(const std::string& name, const std::string& run,
                        uint32_t bytes, uint32_t tx, uint32_t rx)
{
    ValidateName(name, run); ValidateAllocation(bytes, tx, rx);
    bip::managed_shared_memory segment(bip::open_only, name.c_str());
    VerifyOwner(segment, run);
    auto env = segment.find<Interface::Cpp2PyMsgVector>(ENV_NAME);
    auto act = segment.find<Interface::Py2CppMsgVector>(ACT_NAME);
    auto sync = segment.find<Ns3AiMsgSync>(SYNC_NAME);
    if (segment.get_size() != bytes || !env.first || !act.first || !sync.first ||
        env.second != 1 || act.second != 1 || sync.second != 1 ||
        env.first->capacity() != rx || act.first->capacity() != tx)
    {
        throw std::invalid_argument("shared vector configuration mismatch");
    }
}
template <typename Vector> void CopyTo(Vector& vector, const std::vector<uint8_t>& bytes)
{
    if (bytes.size() < HEADER_BYTES || bytes.size() > vector.capacity())
    {
        throw std::invalid_argument("message exceeds allocated vector");
    }
    vector.resize(bytes.size());
    std::copy(bytes.begin(), bytes.end(), vector.begin());
}
template <typename Vector> std::vector<uint8_t> CopyFrom(const Vector& vector)
{
    if (vector.size() < HEADER_BYTES)
    {
        throw std::invalid_argument("short vector");
    }
    uint64_t length = 0;
    for (unsigned i = 0; i < 4; ++i) { length |= uint64_t{vector[16 + i]} << (8 * i); }
    if (length != vector.size() - HEADER_BYTES)
    {
        throw std::invalid_argument("oversized message");
    }
    return {vector.begin(), vector.begin() + HEADER_BYTES + length};
}
} // namespace ns3::fanet::wire
#endif
