#include "worker.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>

static void measure(worker & target, const json & fixtures, std::ofstream & output,
                    int pair, int block, char arm, const json & condition, bool precondition) {
    const std::string implementation(1, arm == 'R' ? 'B' : arm);
    for (const auto & input : fixtures) {
        auto result = target.call({{"action", "measure"}, {"arm", implementation},
                                   {"case", input.at("name")}, {"decode", false},
                                   {"repetitions", 20}, {"block", block}, {"warmup", block < 0}});
        result["arm"] = std::string(1, arm);
        result["implementation"] = implementation;
        result["worker"] = arm == 'B' || arm == 'A' ? "baseline" : "candidate";
        result["pair"] = pair;
        result["worker_pid"] = target.pid;
        result["condition"] = condition;
        result["preconditioning"] = precondition;
        output << result << '\n';
        output.flush();
        if (!output) { throw std::runtime_error("observation write failed"); }
    }
}

int main(int argc, char ** argv) {
    try {
        if (argc != 7) {
            throw std::runtime_error("usage: process-control BASELINE CANDIDATE MODEL INPUT_DIR OUTPUT_DIR PLAN");
        }
        std::ifstream plan_file(argv[6]);
        const auto plan = json::parse(plan_file);
        const auto & pairs = plan.at("pairs");
        if (pairs.size() != 24 || plan.at("warmup_blocks") != 3 || plan.at("repetitions_per_cell") != 20) {
            throw std::runtime_error("unexpected control design");
        }
        std::set<std::string> conditions;
        for (const auto & condition : pairs) {
            const auto startup = condition.at("startup").get<std::string>();
            const auto order = condition.at("order").get<std::string>();
            if ((startup != "baseline-first" && startup != "candidate-first") ||
                std::set<char>(order.begin(), order.end()) != std::set<char>{'B', 'R', 'C'} ||
                order.size() != 3 || !condition.at("A_preconditioning").is_boolean() ||
                !conditions.insert(condition.dump()).second) {
                throw std::runtime_error("incomplete or duplicate factorial condition");
            }
        }
        const std::filesystem::path out(argv[5]);
        std::filesystem::create_directories(out);
        std::ofstream results(out / "observations.jsonl");
        json admitted;
        int pair = 0;
        for (const auto & condition : pairs) {
            std::unique_ptr<worker> baseline, candidate;
            if (condition.at("startup") == "baseline-first") {
                baseline = std::make_unique<worker>(argv[1], argv[3], argv[4]);
                candidate = std::make_unique<worker>(argv[2], argv[3], argv[4]);
            } else {
                candidate = std::make_unique<worker>(argv[2], argv[3], argv[4]);
                baseline = std::make_unique<worker>(argv[1], argv[3], argv[4]);
            }
            const auto fixtures = baseline->ready.at("fixtures");
            if (fixtures.size() != 5 || fixtures != candidate->ready.at("fixtures") ||
                (!admitted.is_null() && admitted != fixtures)) {
                throw std::runtime_error("worker fixture identity mismatch");
            }
            admitted = fixtures;
            std::ofstream(out / "admitted-fixtures.json") << admitted.dump(2) << '\n';
            if (condition.at("A_preconditioning").get<bool>()) {
                measure(*baseline, fixtures, results, pair, -4, 'A', condition, true);
            }
            for (int block = -3; block <= 0; ++block) {
                // Preserve R5's per-case arm interleaving, including warmup.
                for (const auto & input : fixtures) {
                    for (char arm : condition.at("order").get<std::string>()) {
                        measure(arm == 'B' ? *baseline : *candidate, json::array({input}),
                                results, pair, block, arm, condition, false);
                    }
                }
            }
            std::cerr << "completed pair=" << pair << " condition=" << condition << std::endl;
            ++pair;
        }
        std::ofstream(out / "control-summary.json") << json({{"pairs", pair}, {"qualification", false}}).dump(2) << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "grammar-process-control: " << error.what() << '\n';
        return 1;
    }
}
