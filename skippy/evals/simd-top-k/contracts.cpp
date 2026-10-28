#include "json.hpp"
#include "host-observation.h"

#include <cerrno>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;

static void check(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

static json fixtures() {
    json result = json::array();
    for (const char * name : {"lazy-before-tool-trigger", "large-HTML-write-argument", "Unicode-and-partial-UTF8",
                             "restricted-values-and-rejection-fallback", "closing-delimiters-and-EOS"}) {
        result.push_back({{"name", name}, {"teacher_tokens", {1, 2}}, {"synthetic_contract_fixture", true}});
    }
    return result;
}

static void save(const fs::path & path, const json & value) {
    std::ofstream out(path);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << value.dump() << '\n';
}

static int run(const std::vector<std::string> & arguments) {
    const auto pid = fork();
    check(pid >= 0, "fork failed");
    if (pid == 0) {
        std::vector<char *> raw;
        for (const auto & arg : arguments) { raw.push_back(const_cast<char *>(arg.c_str())); }
        raw.push_back(nullptr);
        execv(raw[0], raw.data());
        _exit(127);
    }
    int status = 0;
    pid_t exited;
    do { exited = waitpid(pid, &status, 0); } while (exited < 0 && errno == EINTR);
    check(exited == pid && WIFEXITED(status), "child crashed");
    return WEXITSTATUS(status);
}

static int synthetic_worker(const char * executable) {
    const char arm = fs::path(executable).filename().string().at(0);
    std::cout << json({{"ready", true}, {"fixtures", fixtures()}}) << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
        const auto request = json::parse(line);
        check(request.at("action") == "measure", "unexpected mock action");
        check(request.at("arm") == std::string(1, arm), "controller chose wrong worker");
        const bool decode = request.at("decode");
        const int repeats = request.at("repetitions");
        check(repeats == (decode ? 1 : 4), "controller chose wrong repetition count");
        std::vector<int64_t> times(static_cast<size_t>(repeats) * 2, !decode && arm == 'D' ? 900 : 1000);
        std::vector<int> tokens;
        for (int i = 0; i < repeats; ++i) { tokens.push_back(1); tokens.push_back(2); }
        std::cout << json({{"arm", std::string(1, arm)}, {"case", request.at("case")},
            {"scope", decode ? "teacher-forced-native-decode" : "fixed-input-native-sampler"},
            {"times_ns", times}, {"selected_tokens", tokens}, {"teacher_tokens", 2},
            {"repetitions", repeats}, {"block", request.at("block")}, {"warmup", request.at("warmup")},
            {"thermal_before", 0}, {"thermal_after", 0}, {"peak_rss_bytes", 1048576}}) << std::endl;
    }
    return 0;
}

static json read_rows(const fs::path & path) {
    std::ifstream in(path);
    check(in.good(), "missing synthetic observations");
    json rows = json::array();
    std::string line;
    while (std::getline(in, line)) { rows.push_back(json::parse(line)); }
    return rows;
}

static void save_rows(const fs::path & path, const json & rows) {
    std::ofstream out(path);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    for (const auto & row : rows) { out << row.dump() << '\n'; }
}

static void audit_contracts(const std::string & auditor, const fs::path & root, const json & original) {
    const auto rows_path = root / "synthetic-observations.jsonl";
    const auto output_path = root / "synthetic-audit.json";
    const auto audit = [&](const json & rows) {
        save_rows(rows_path, rows);
        fs::remove(output_path);
        return run({auditor, rows_path.string(), (root / "fixtures.json").string(), "5", output_path.string()});
    };
    check(audit(original) == 0, "valid synthetic rows rejected");
    json accepted;
    std::ifstream(output_path) >> accepted;
    check(accepted.at("screen_pass") == true && accepted.at("qualified_speed") == false, "screen gate misclassified");
    for (const auto & row : accepted.at("aggregate")) {
        const double expected = row.at("comparison") == "DC" || row.at("comparison") == "DB"
            ? (row.at("scope") == "fixed-input-native-sampler" ? 0.9 : 1.0) : 1.0;
        for (const char * field : {"ratio", "lower_95", "upper_95"}) {
            const double actual = row.at("confidence").at(field);
            check(actual > expected - 1e-12 && actual < expected + 1e-12, "paired bootstrap changed a constant ratio");
        }
    }
    auto varied = original;
    for (auto & row : varied) {
        if (row.at("arm") != "D") { continue; }
        int factor = 1;
        for (const auto & input : fixtures()) {
            if (input.at("name") == row.at("case")) { break; }
            ++factor;
        }
        const int multiplier = row.at("block").get<int>() >= 3 ? 2 : 1;
        for (auto & time : row["times_ns"]) { time = 1000 * factor * multiplier; }
    }
    check(audit(varied) == 0, "valid variable-ratio rows rejected");
    std::ifstream(output_path) >> accepted;
    for (const auto & row : accepted.at("aggregate")) {
        if (row.at("comparison") != "DC" && row.at("comparison") != "DB") { continue; }
        // Independent binomial result: two of five blocks double each case's
        // ratio. For five bootstrap draws, the 5/95% quantiles have 0/4 doubles.
        const double base = std::pow(120.0, 0.2);
        const double point = base * std::pow(2.0, 0.4);
        const double upper = base * std::pow(2.0, 0.8);
        const bool decode = row.at("scope") == "teacher-forced-native-decode";
        for (const auto & expected : std::vector<std::pair<std::string, double>>{
                 {"ratio", decode ? 1.0 / point : point},
                 {"lower_95", decode ? 1.0 / upper : base},
                 {"upper_95", decode ? 1.0 / base : upper}}) {
            const double actual = row.at("confidence").at(expected.first);
            check(std::abs(actual - expected.second) < 1e-12, "variable-ratio case-balanced bootstrap differs from binomial result");
        }
    }
    for (int mutation = 0; mutation < 8; ++mutation) {
        auto rows = original;
        if (mutation == 0) { rows.erase(rows.begin()); }
        if (mutation == 1) { rows.push_back(rows.at(0)); }
        if (mutation == 2) { rows.at(0)["thermal_before"] = -1; }
        if (mutation == 3) { rows.at(0)["worker_iteration"] = -4; }
        if (mutation == 4) { rows.at(0)["arm_order"] = "BCDR"; }
        if (mutation == 5) { rows.at(0)["times_ns"] = {0}; }
        if (mutation == 6) { rows.at(0)["teacher_tokens"] = 3; }
        if (mutation == 7) {
            for (auto & row : rows) {
                if (row.at("arm") == "D" && row.at("warmup") == false) {
                    row["selected_tokens"][0] = 3;
                    break;
                }
            }
        }
        check(audit(rows) != 0, "malformed synthetic rows accepted");
    }
    auto slow = original;
    for (auto & row : slow) {
        if (row.at("arm") == "D") {
            for (auto & time : row["times_ns"]) { time = 1100; }
        }
    }
    check(audit(slow) == 0, "complete slower candidate was discarded");
    std::ifstream(output_path) >> accepted;
    check(accepted.at("screen_pass") == false && accepted.at("within_one_percent") == false &&
          accepted.at("sampler_improves_at_least_five_percent") == false, "slower candidate passed gates");
}

static void host_contracts(const fs::path & root) {
    const auto sensor = root / "sensors/hwmon0";
    fs::create_directories(sensor);
    { std::ofstream out(sensor / "name"); out << "synthetic-sensor"; }
    { std::ofstream out(sensor / "temp1_input"); out << "52000\n"; }
    { std::ofstream out(sensor / "temp2_input"); out << "unavailable\n"; }
    { std::ofstream out(sensor / "temp3_input"); out << "999999999\n"; }
    const auto readings = temperature_sensor_readings(root / "sensors");
    check(readings.at("healthy").is_null() && readings.at("readings").size() == 3, "sensor availability became a health claim");
    int valid = 0, unknown = 0;
    for (const auto & item : readings.at("readings")) {
        if (item.at("millidegrees_celsius").is_null()) { ++unknown; }
        else { check(item.at("millidegrees_celsius") == 52000, "wrong sensor units"); ++valid; }
    }
    check(valid == 1 && unknown == 2, "unavailable temperature became healthy");
    const auto missing = temperature_sensor_readings(root / "missing-sensors");
    check(!missing.at("directory_error").is_null() && missing.at("healthy").is_null(), "missing sensors silently accepted");
}

int main(int argc, char ** argv) {
    try {
        if (argc == 4 && std::string(argv[2]) == "serve") { return synthetic_worker(argv[0]); }
        check(argc == 3, "usage: simd-contracts CONTROLLER AUDITOR");
        const auto root = fs::temp_directory_path() / ("simd-synthetic-contracts-" + std::to_string(getpid()));
        check(fs::create_directory(root), "contract directory already exists");
        host_contracts(root);
        save(root / "fixtures.json", fixtures());
        std::vector<std::string> command{fs::absolute(argv[1]).string()};
        for (char arm : std::string("BCDR")) {
            const auto link = root / std::string(1, arm);
            fs::create_symlink(fs::canonical(argv[0]), link);
            command.push_back(link.string());
        }
        command.insert(command.end(), {"synthetic-model-never-opened", root.string(), (root / "screen").string(), "screen", "0", "5"});
        check(run(command) == 0, "four-arm synthetic controller failed");
        const auto rows = read_rows(root / "screen/observations.jsonl");
        check(rows.size() == 800, "controller produced incomplete screen");
        const auto hosts = read_rows(root / "screen/host-observations.jsonl");
        check(hosts.size() == 1640, "missing durable call brackets or arm observations");
        int before = 0, after = 0;
        for (const auto & host : hosts) {
            check(host.at("host").at("unix_ms").get<int64_t>() > 0, "missing host observation timestamp");
            check(host.at("host").at("temperatures").at("healthy").is_null(), "unknown sensors became healthy");
            before += host.at("event") == "before-call";
            after += host.at("event") == "after-call";
        }
        check(before == 800 && after == 800, "worker calls lack durable host observations");
        check(run(command) != 0, "controller overwrote existing evidence");
        auto invalid = command;
        invalid[7] = (root / "wrong-range").string();
        invalid[10] = "4";
        check(run(invalid) != 0, "controller accepted changed screen size");
        audit_contracts(fs::absolute(argv[2]).string(), root, rows);
        fs::remove_all(root);
        std::cout << "Synthetic controller/completeness/bootstrap/gate contracts passed; no model or performance execution\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "simd-contracts: " << error.what() << '\n';
        return 1;
    }
}
