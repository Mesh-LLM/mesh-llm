#include <nvml.h>

namespace {
struct nvml_guard {
    nvmlDevice_t device{};
    bool ready = false;
    nvml_guard() {
        if (nvmlInit_v2() == NVML_SUCCESS) {
            ready = nvmlDeviceGetHandleByIndex_v2(0, &device) == NVML_SUCCESS;
        }
    }
};
}

// This reports GPU thermal/hardware slowdown only. Linux CPU temperatures and
// available counters remain separate observations; unknown is never healthy.
extern "C" int thermal_state() {
    static nvml_guard guard;
    if (!guard.ready) { return -1; }
    const unsigned long long thermal = nvmlClocksThrottleReasonHwThermalSlowdown |
                                      nvmlClocksEventReasonSwThermalSlowdown;
    unsigned long long supported = 0;
    unsigned long long active = 0;
    if (nvmlDeviceGetSupportedClocksEventReasons(guard.device, &supported) != NVML_SUCCESS ||
            (supported & thermal) != thermal ||
            nvmlDeviceGetCurrentClocksEventReasons(guard.device, &active) != NVML_SUCCESS) {
        return -1;
    }
    const auto hardware = nvmlClocksThrottleReasonHwSlowdown |
                          nvmlClocksThrottleReasonHwPowerBrakeSlowdown;
    return (active & (thermal | hardware)) != 0 ? 2 : 0;
}
