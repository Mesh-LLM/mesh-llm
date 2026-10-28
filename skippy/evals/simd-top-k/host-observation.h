#pragma once

#include "json.hpp"
#include <filesystem>

nlohmann::ordered_json temperature_sensor_readings(const std::filesystem::path & root);
nlohmann::ordered_json sampler_host_observation(bool include_processes);
