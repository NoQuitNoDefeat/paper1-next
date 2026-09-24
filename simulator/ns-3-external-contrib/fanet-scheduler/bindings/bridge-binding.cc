#include "../model/bridge-transport.h"
#include "../model/controlled-environment.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace ns3::fanet::wire
{
/** Exactly one instance in each isolated worker; caller supervises all blocking waits. */
class WorkerChannel
{
  public:
    WorkerChannel(const std::string& name, const std::string& run,
                  uint32_t bytes, uint32_t tx, uint32_t rx)
    {
        static bool used = false;
        if (used)
        {
            throw std::runtime_error("native channel requires a fresh process");
        }
        used = true;
        Prepare(name, run, bytes, tx, rx);
        m_interface = std::make_unique<Interface>(false, true, false, bytes, name.c_str(),
                                                 ENV_NAME, ACT_NAME, SYNC_NAME);
    }
    py::bytes Exchange(py::bytes input)
    {
        const std::string data = input;
        std::vector<uint8_t> output;
        {
            py::gil_scoped_release release;
            std::vector<uint8_t> bytes(data.begin(), data.end());
            if (bytes.size() < HEADER_BYTES ||
                bytes.size() > m_interface->GetPy2CppVector()->capacity())
            {
                throw std::invalid_argument("outgoing message size");
            }
            m_interface->PySendBegin();
            CopyTo(*m_interface->GetPy2CppVector(), bytes);
            m_interface->PySendEnd();
            m_interface->PyRecvBegin();
            try
            {
                output = CopyFrom(*m_interface->GetCpp2PyVector());
            }
            catch (...)
            {
                m_interface->PyRecvEnd(); throw;
            }
            m_interface->PyRecvEnd();
        }
        return py::bytes(reinterpret_cast<const char*>(output.data()), output.size());
    }
  private:
    std::unique_ptr<Interface> m_interface;
};
} // namespace ns3::fanet::wire

PYBIND11_MODULE(fanet_bridge_native, module)
{
    using namespace ns3::fanet::wire;
    py::class_<WorkerChannel>(module, "WorkerChannel")
        .def(py::init<const std::string&, const std::string&, uint32_t, uint32_t, uint32_t>())
        .def("exchange", &WorkerChannel::Exchange);
    module.def("build_info", [] {
        const auto info = ns3::fanet::ControlledBuildInfo();
        return py::make_tuple(info[0], info[1], info[2], FANET_BRIDGE_AI_COMMIT);
    });
}
