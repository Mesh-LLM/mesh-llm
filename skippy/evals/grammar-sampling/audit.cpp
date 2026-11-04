#include "json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;
using cell_key = std::tuple<std::string, std::string, int>;

static void check(bool condition, const std::string & reason) {
    if (!condition) { throw std::runtime_error(reason); }
}

static json confidence(const std::vector<double> & logs) {
    check(!logs.empty(), "missing paired observations");
    const double point = std::exp(std::accumulate(logs.begin(), logs.end(), 0.0) / logs.size());
    std::mt19937 random(7301);
    std::uniform_int_distribution<size_t> pick(0, logs.size() - 1);
    std::vector<double> bootstrap;
    bootstrap.reserve(10000);
    for (int trial = 0; trial < 10000; ++trial) {
        double sum = 0.0;
        for (size_t i = 0; i < logs.size(); ++i) { sum += logs[pick(random)]; }
        bootstrap.push_back(std::exp(sum / logs.size()));
    }
    std::sort(bootstrap.begin(), bootstrap.end());
    return {{"ratio", point}, {"lower_95", bootstrap[499]}, {"upper_95", bootstrap[9499]}, {"paired_blocks", logs.size()}};
}

int main(int argc, char ** argv) {
    try {
        check(argc == 4, "usage: audit OBSERVATIONS EXPECTED_BLOCKS OUTPUT");
        const int blocks = std::stoi(argv[2]);
        check(blocks == 5 || blocks == 30, "unexpected block count");
        std::ifstream input(argv[1]);
        check(input.good(), "missing observations");
        std::map<cell_key, std::map<std::string, json>> cells;
        std::set<std::string> cases;
        const std::set<std::string> expected_cases{
            "lazy-before-tool-trigger", "large-HTML-write-argument", "Unicode-and-partial-UTF8",
            "restricted-values-and-rejection-fallback", "closing-delimiters-and-EOS"};
        std::set<std::tuple<std::string, std::string, int, std::string, int>> warmups;
        std::map<std::pair<int, std::string>, int> worker_processes;
        std::map<int, std::pair<int, std::string>> process_owners;
        const std::array<std::string, 6> orders{"ABC", "BCA", "CAB", "CBA", "ACB", "BAC"};
        std::string line;
        size_t warmup_rows = 0;
        while (std::getline(input, line)) {
            auto row = json::parse(line);
            const auto arm = row.at("arm").get<std::string>();
            check(arm == "A" || arm == "B" || arm == "C", "unknown arm");
            const int block = row.at("block");
            check(row.at("protocol_revision").get<int>() == 8, "wrong process protocol");
            check(block >= 0 && block < blocks, "invalid block identity");
            check(row.at("arm_order").get<std::string>() == orders[block % orders.size()], "wrong arm order");
            const int pid = row.at("worker_pid");
            check(pid > 0, "invalid worker PID");
            const auto owner = std::make_pair(block, arm);
            const auto [owned, inserted] = worker_processes.emplace(owner, pid);
            check(inserted || owned->second == pid, "worker changed inside arm/block");
            const auto [process, fresh] = process_owners.emplace(pid, owner);
            check(fresh || process->second == owner, "worker shared across arms or blocks");
            const auto name = row.at("case").get<std::string>();
            const auto scope = row.at("scope").get<std::string>();
            check(expected_cases.count(name), "unknown case identity");
            check(scope == "fixed-input-native-sampler" || scope == "teacher-forced-native-decode", "unknown scope");
            check(row.at("repetitions").get<int>() == (scope == "fixed-input-native-sampler" ? 4 : 1), "wrong repetitions");
            check(row.at("thermal_before").get<int>() <= 1 && row.at("thermal_after").get<int>() <= 1, "thermal pressure");
            if (row.at("warmup").get<bool>()) {
                const int iteration = row.at("worker_iteration");
                check(iteration >= -3 && iteration < 0, "invalid warmup identity");
                check(warmups.emplace(name, scope, block, arm, iteration).second, "duplicate warmup");
                ++warmup_rows;
                continue;
            }
            check(row.at("worker_iteration").get<int>() == 0, "invalid measured iteration");
            cases.insert(name);
            auto & arms = cells[{name, scope, block}];
            check(arms.emplace(arm, row).second, "duplicate arm/case/block");
        }
        check(cases.size() == 5 && warmup_rows == static_cast<size_t>(blocks * 3 * 5 * 3 * 2), "missing cases or warmups");
        check(worker_processes.size() == static_cast<size_t>(blocks * 3), "missing fresh workers");
        check(cells.size() == static_cast<size_t>(5 * 2 * blocks), "missing case/scope/block");
        std::map<std::pair<std::string, std::string>, std::vector<double>> cb;
        std::map<std::pair<std::string, std::string>, std::vector<double>> ca;
        std::map<std::pair<std::string, std::string>, std::vector<int64_t>> token_times;
        std::map<std::pair<std::string, std::string>, std::vector<double>> arm_rates;
        std::map<std::string, int64_t> peak_rss;
        for (const auto & [key, arms] : cells) {
            const auto & [name, scope, block] = key;
            (void)block;
            check(arms.size() == 3, "missing comparison arm");
            check(arms.at("B").at("selected_tokens") == arms.at("C").at("selected_tokens"), "B/C replay decision mismatch");
            std::map<std::string, double> means;
            for (const auto & [arm, row] : arms) {
                const auto times = row.at("times_ns").get<std::vector<int64_t>>();
                const size_t expected = row.at("teacher_tokens").get<size_t>() * row.at("repetitions").get<size_t>();
                check(times.size() == expected && !times.empty(), "incomplete token timings");
                check(row.at("selected_tokens").size() == expected, "incomplete sampled decisions");
                check(std::all_of(times.begin(), times.end(), [](auto ns) { return ns > 0; }), "invalid timing");
                means[arm] = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
                auto & all = token_times[{arm, scope}];
                all.insert(all.end(), times.begin(), times.end());
                arm_rates[{arm, scope}].push_back(1e9 / means[arm]);
                peak_rss[arm] = std::max(peak_rss[arm], row.at("peak_rss_bytes").get<int64_t>());
            }
            const bool decode = scope == "teacher-forced-native-decode";
            cb[{name, scope}].push_back(std::log(decode ? means["B"] / means["C"] : means["C"] / means["B"]));
            ca[{name, scope}].push_back(std::log(decode ? means["A"] / means["C"] : means["C"] / means["A"]));
        }
        json per_case = json::array();
        std::map<std::string, std::vector<double>> overall_cb, overall_ca;
        bool llama_gate = true;
        bool existing_win = true;
        for (const auto & [key, values] : cb) {
            const auto & [name, scope] = key;
            check(values.size() == static_cast<size_t>(blocks), "incomplete paired blocks");
            const auto b = confidence(values);
            const auto a = confidence(ca.at(key));
            const bool decode = scope == "teacher-forced-native-decode";
            const bool within = decode ? b.at("lower_95").get<double>() >= 0.99 : b.at("upper_95").get<double>() <= 1.01;
            llama_gate = llama_gate && within;
            // The win gate is aggregated; preserve restrictive/fallback cases
            // even where extra rejection sampling is expected to cost more.
            per_case.push_back({{"case", name}, {"scope", scope}, {"C_vs_B", b}, {"C_vs_A", a}, {"within_one_percent", within}});
            auto & bc = overall_cb[scope];
            auto & ac = overall_ca[scope];
            if (bc.empty()) { bc.resize(blocks, 0.0); ac.resize(blocks, 0.0); }
            for (int block = 0; block < blocks; ++block) {
                bc[block] += values[block] / 5.0;
                ac[block] += ca.at(key)[block] / 5.0;
            }
        }
        json aggregate = json::array();
        for (const auto & [scope, values] : overall_cb) {
            const auto b = confidence(values);
            const auto a = confidence(overall_ca.at(scope));
            const bool decode = scope == "teacher-forced-native-decode";
            const bool within = decode ? b.at("lower_95").get<double>() >= 0.99 : b.at("upper_95").get<double>() <= 1.01;
            llama_gate = llama_gate && within;
            const bool improves = decode ? a.at("lower_95").get<double>() > 1.0 : a.at("upper_95").get<double>() < 1.0;
            existing_win = existing_win && improves;
            aggregate.push_back({{"scope", scope}, {"C_vs_B", b}, {"C_vs_A", a}, {"within_one_percent", within}, {"improves_existing", improves}});
        }
        json arms = json::array();
        for (auto & [key, times] : token_times) {
            const auto & [arm, scope] = key;
            std::sort(times.begin(), times.end());
            const auto & rates = arm_rates.at(key);
            double lograte = 0;
            for (auto rate : rates) { lograte += std::log(rate); }
            arms.push_back({{"arm", arm}, {"scope", scope}, {"p50_ns", times[times.size() / 2]},
                            {"p95_ns", times[(times.size() - 1) * 95 / 100]},
                            {"geomean_tokens_s", std::exp(lograte / rates.size())}, {"peak_rss_bytes", peak_rss[arm]}});
        }
        const bool qualified = blocks == 30 && llama_gate && existing_win;
        json result = {{"schema", 1}, {"blocks", blocks}, {"complete", true}, {"B_C_decisions_match", true},
                       {"within_one_percent", llama_gate}, {"improves_existing", existing_win}, {"qualified_speed", qualified},
                       {"per_case", per_case}, {"aggregate", aggregate}, {"arms", arms},
                       {"remaining_gates", {"external source/model/binary identity audit", "native and MTP correctness", "co-tenancy audit", "shared MLflow artifact readback"}}};
        std::ofstream(argv[3]) << result.dump(2) << '\n';
        std::cout << result.dump(2) << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "grammar-audit: " << error.what() << '\n';
        return 1;
    }
}
