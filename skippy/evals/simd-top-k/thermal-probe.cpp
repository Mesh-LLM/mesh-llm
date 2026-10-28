#include <iostream>
extern "C" int thermal_state();
int main() {
    const auto state = thermal_state();
    std::cout << "{\"thermal_state\":" << state << ",\"scope\":\"instrumentation prerequisite only\"}\n";
    return state >= 0 && state <= 1 ? 0 : 1;
}
