#include "worker.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

int main(int argc, char ** argv) {
    try {
        if (argc != 8) { throw std::runtime_error("usage: fresh-controller BASELINE CANDIDATE MODEL INPUT_DIR OUTPUT_DIR FIRST_BLOCK BLOCKS"); }
        const int first = std::stoi(argv[6]);
        const int blocks = std::stoi(argv[7]);
        if (first < 0 || blocks < 1 || first + blocks > 30) { throw std::runtime_error("invalid block range"); }
        const std::filesystem::path out = argv[5];
        std::filesystem::create_directories(out);
        std::ofstream results(out / "observations.jsonl");
        results.exceptions(std::ios::badbit | std::ios::failbit);
        const std::array<std::string, 6> orders{"ABC", "BCA", "CAB", "CBA", "ACB", "BAC"};
        json fixtures;
        std::set<pid_t> processes;
        for (int block = first; block < first + blocks; ++block) {
            const auto & order = orders[static_cast<size_t>(block) % orders.size()];
            for (const char arm : order) {
                worker target(arm == 'C' ? argv[2] : argv[1], argv[3], argv[4]);
                if (!processes.insert(target.pid).second) { throw std::runtime_error("worker PID reused"); }
                if (fixtures.is_null()) {
                    fixtures = target.ready.at("fixtures");
                    std::ofstream(out / "admitted-fixtures.json") << fixtures.dump(2) << '\n';
                }
                if (target.ready.at("fixtures") != fixtures) { throw std::runtime_error("fixture identity mismatch"); }
                // Every arm gets an independent lifetime and identical warmup.
                // Loading, prefill and IPC remain outside per-token timing.
                for (int iteration = -3; iteration <= 0; ++iteration) {
                    for (const bool decode : {false, true}) {
                        for (const auto & input : fixtures) {
                            auto row = target.call({{"action", "measure"}, {"arm", std::string(1, arm)},
                                {"case", input.at("name")}, {"decode", decode},
                                {"repetitions", decode ? 1 : 4}, {"block", block}, {"warmup", iteration < 0}});
                            row["protocol_revision"] = 8;
                            row["worker_pid"] = target.pid;
                            row["worker_iteration"] = iteration;
                            row["arm_order"] = order;
                            results << row << '\n';
                            results.flush();
                        }
                    }
                }
                std::cerr << "completed block=" << block << " arm=" << arm << " pid=" << target.pid << std::endl;
                target.finish();
            }
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "grammar-fresh-controller: " << error.what() << '\n';
        return 1;
    }
}
