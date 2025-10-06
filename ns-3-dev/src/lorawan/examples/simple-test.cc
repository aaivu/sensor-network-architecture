/*
 * simple-test.cc - Minimal test file to verify compilation
 */

#include "ns3/core-module.h"
#include "ns3/lorawan-module.h"

using namespace ns3;
using namespace lorawan;

NS_LOG_COMPONENT_DEFINE("SimpleTest");

int main(int argc, char* argv[])
{
    std::cout << "Hello ns-3 LoRaWAN!" << std::endl;
    return 0;
}
