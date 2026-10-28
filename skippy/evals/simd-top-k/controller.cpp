#include "../grammar-sampling/worker.h"
#include "host-observation.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

static void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

static void measure(worker & target, const json & fixtures, std::ofstream & output,
                    std::ofstream & host_output,
                    const std::string & phase, int block, char arm, const std::string & order) {
    for (int iteration = -3; iteration <= 0; ++iteration) {
        for (const bool decode : {false, true}) {
            for (const auto & input : fixtures) {
                // These durable observations bracket the entire worker call,
                // outside all per-token timing, and survive a client outage.
                host_output << json({{"event", "before-call"}, {"block", block}, {"arm", std::string(1, arm)},
                    {"case", input.at("name")}, {"iteration", iteration}, {"decode", decode},
                    {"host", sampler_host_observation(false)}}) << std::endl;
                auto row = target.call({{"action", "measure"}, {"arm", std::string(1, arm)},
                    {"case", input.at("name")}, {"decode", decode},
                    {"repetitions", decode ? 1 : 4}, {"block", block}, {"warmup", iteration < 0}});
                host_output << json({{"event", "after-call"}, {"block", block}, {"arm", std::string(1, arm)},
                    {"case", input.at("name")}, {"iteration", iteration}, {"decode", decode},
                    {"host", sampler_host_observation(false)}}) << std::endl;
                require(row.at("arm") == std::string(1, arm), "worker arm mismatch");
                require(row.at("case") == input.at("name"), "worker case mismatch");
                require(row.at("block") == block && row.at("warmup") == (iteration < 0), "worker block mismatch");
                row["protocol"] = "simd-top-k-v1";
                row["phase"] = phase;
                row["worker_pid"] = target.pid;
                row["worker_iteration"] = iteration;
                row["arm_order"] = order;
                output << row << '\n';
                output.flush();
            }
        }
    }
}

int main(int argc, char ** argv) {
    try {
        require(argc == 11, "usage: simd-controller B C D R MODEL INPUT_DIR OUTPUT_DIR screen|qualification FIRST_BLOCK BLOCKS");
        const std::string phase = argv[8];
        const int first = std::stoi(argv[9]);
        const int blocks = std::stoi(argv[10]);
        require((phase == "screen" && first == 0 && blocks == 5) ||
                (phase == "qualification" && (first == 0 || first == 10 || first == 20) && blocks == 10),
                "range differs from frozen screen/qualification plan");
        const std::filesystem::path out = argv[7];
        std::filesystem::create_directories(out);
        require(!std::filesystem::exists(out / "observations.jsonl"), "refusing to overwrite observations");
        std::ifstream manifest_file(std::filesystem::path(argv[6]) / "fixtures.json");
        const auto fixtures = json::parse(manifest_file);
        require(fixtures.is_array() && fixtures.size() == 5, "expected all five admitted fixtures");
        require(!std::filesystem::exists(out / "host-observations.jsonl"), "refusing to overwrite host observations");
        std::ofstream results(out / "observations.jsonl");
        results.exceptions(std::ios::badbit | std::ios::failbit);
        std::ofstream host_output(out / "host-observations.jsonl");
        host_output.exceptions(std::ios::badbit | std::ios::failbit);
        std::ofstream admitted(out / "admitted-fixtures.json");
        admitted.exceptions(std::ios::badbit | std::ios::failbit);
        admitted << fixtures.dump(2) << '\n';
        admitted.close();
        const std::array<std::string, 4> orders{"BCRD", "CDBR", "DRCB", "RBDC"};
        std::set<pid_t> processes;
        for (int block = first; block < first + blocks; ++block) {
            const auto & order = orders[static_cast<size_t>(block) % orders.size()];
            for (const char arm : order) {
                // finish() precedes the next construction: one resident model,
                // with independent process history and complete warmup per arm.
                const int executable = arm == 'B' ? 1 : arm == 'C' ? 2 : arm == 'D' ? 3 : 4;
                worker target(argv[executable], argv[5], argv[6]);
                require(processes.insert(target.pid).second, "worker PID reused");
                require(target.ready.at("fixtures") == fixtures, "worker fixture identity mismatch");
                host_output << json({{"event", "arm-start"}, {"block", block}, {"arm", std::string(1, arm)},
                    {"host", sampler_host_observation(true)}}) << std::endl;
                measure(target, fixtures, results, host_output, phase, block, arm, order);
                std::cerr << "completed phase=" << phase << " block=" << block << " arm=" << arm
                          << " pid=" << target.pid << std::endl;
                target.finish();
                host_output << json({{"event", "arm-end"}, {"block", block}, {"arm", std::string(1, arm)},
                    {"host", sampler_host_observation(true)}}) << std::endl;
            }
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "simd-controller: " << error.what() << '\n';
        return 1;
    }
}
