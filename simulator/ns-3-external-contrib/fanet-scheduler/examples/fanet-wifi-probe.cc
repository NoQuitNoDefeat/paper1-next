#include "ns3/command-line.h"
#include "ns3/wifi-execution-probe.h"

#include <iostream>
#include <stdexcept>

using namespace ns3;

int
main(int argc, char* argv[])
{
    std::string name = "single";
    uint32_t seed = 41;
    uint64_t run = 1;
    CommandLine cmd(__FILE__);
    cmd.AddValue("case", "Documented Wi-Fi component case name", name);
    cmd.AddValue("seed", "ns-3 RNG seed", seed);
    cmd.AddValue("run", "ns-3 RNG run number", run);
    cmd.Parse(argc, argv);
    try
    {
        auto config = MakeWifiProbeConfig(name);
        config.seed = seed;
        config.run = run;
        auto result = RunWifiExecutionProbe(config);
        WriteWifiProbeJson(std::cout, result);
        if (!std::cout)
        {
            throw std::runtime_error("Could not write complete component report");
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "Wi-Fi component probe failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
