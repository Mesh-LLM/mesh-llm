#include "host-observation.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>
#ifdef __APPLE__
#include <libproc.h>
#endif

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;

static std::string read_line(const fs::path & path) {
    std::ifstream in(path);
    std::string text;
    std::getline(in, text);
    return text;
}

json temperature_sensor_readings(const fs::path & root) {
    json readings = json::array();
    std::error_code error;
    for (fs::directory_iterator sensors(root, error), end; !error && sensors != end; sensors.increment(error)) {
        const auto name = read_line(sensors->path() / "name");
        std::error_code files_error;
        for (fs::directory_iterator files(sensors->path(), files_error); !files_error && files != end; files.increment(files_error)) {
            const auto file = files->path().filename().string();
            if (file.rfind("temp", 0) != 0 || file.size() < 11 || file.substr(file.size() - 6) != "_input") { continue; }
            std::ifstream in(files->path());
            int value;
            std::string extra;
            const bool valid = static_cast<bool>(in >> value) && !(in >> extra) && value >= -273150 && value <= 250000;
            readings.push_back({{"sensor", name}, {"input", file},
                {"millidegrees_celsius", valid ? json(value) : json(nullptr)}});
        }
    }
    return {{"scope", "available hwmon sensors; no CPU/GPU identity inferred"},
        {"readings", readings}, {"directory_error", error ? json(error.message()) : json(nullptr)},
        {"healthy", nullptr}};
}

static void append_workspace(json & rows, int pid, const std::string & cwd) {
    const auto start = cwd.find("/jianyang/work/run_");
    if (start == std::string::npos) { return; }
    const auto run_start = start + std::string("/jianyang/work/").size();
    const auto end = cwd.find('/', run_start);
    rows.push_back({{"pid", pid}, {"run_directory", cwd.substr(run_start, end - run_start)}});
}

static json job_processes() {
    json rows = json::array();
    size_t unreadable = 0;
#ifdef __linux__
    std::error_code error;
    for (fs::directory_iterator pids("/proc", error), end; !error && pids != end; pids.increment(error)) {
        const auto name = pids->path().filename().string();
        if (name.find_first_not_of("0123456789") != std::string::npos) { continue; }
        std::error_code link_error;
        const auto cwd = fs::read_symlink(pids->path() / "cwd", link_error);
        if (link_error) { ++unreadable; continue; }
        append_workspace(rows, std::stoi(name), cwd.string());
    }
#elif defined(__APPLE__)
    std::vector<int> pids(65536);
    const int bytes = proc_listpids(PROC_ALL_PIDS, 0, pids.data(), static_cast<int>(pids.size() * sizeof(int)));
    if (bytes <= 0) { ++unreadable; }
    for (int i = 0; i < bytes / static_cast<int>(sizeof(int)); ++i) {
        if (pids[i] <= 0) { continue; }
        proc_vnodepathinfo paths{};
        const int read = proc_pidinfo(pids[i], PROC_PIDVNODEPATHINFO, 0, &paths, sizeof(paths));
        if (read != static_cast<int>(sizeof(paths))) { ++unreadable; continue; }
        append_workspace(rows, pids[i], paths.pvi_cdir.vip_path);
    }
#endif
    return {{"scope", "readable process working directories; not proof of complete OS exclusivity"},
        {"jianyang_workspaces", rows}, {"unreadable_processes", unreadable}, {"healthy", nullptr}};
}

json sampler_host_observation(bool include_processes) {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const char * run = std::getenv("JIANYANG_RUN_ID");
    json value{{"unix_ms", std::chrono::duration_cast<std::chrono::milliseconds>(now).count()},
        {"controller_pid", getpid()}, {"jy_run_id", run ? json(run) : json(nullptr)}};
#ifdef __linux__
    value["temperatures"] = temperature_sensor_readings("/sys/class/hwmon");
#else
    value["temperatures"] = {{"readings", json::array()}, {"healthy", nullptr},
        {"scope", "not collected by this adapter; worker thermal guard remains mandatory"}};
#endif
    if (include_processes) { value["processes"] = job_processes(); }
    return value;
}
