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
#include <tuple>
#include <vector>

using json = nlohmann::ordered_json;
using cell = std::tuple<std::string, std::string, int>;
using comparison = std::tuple<std::string, std::string, std::string>;

static void check(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

static json confidence(const std::vector<double> & logs) {
    check(!logs.empty(), "missing paired blocks");
    std::mt19937 rng(7301);
    std::uniform_int_distribution<size_t> pick(0, logs.size() - 1);
    std::vector<double> bootstrap;
    bootstrap.reserve(10000);
    for (int trial = 0; trial < 10000; ++trial) {
        double sum = 0.0;
        for (size_t i = 0; i < logs.size(); ++i) { sum += logs[pick(rng)]; }
        bootstrap.push_back(std::exp(sum / logs.size()));
    }
    std::sort(bootstrap.begin(), bootstrap.end());
    return {{"ratio", std::exp(std::accumulate(logs.begin(), logs.end(), 0.0) / logs.size())},
            {"lower_95", bootstrap[499]}, {"upper_95", bootstrap[9499]}, {"paired_blocks", logs.size()}};
}

static bool within(const std::string & scope, const json & interval) {
    return scope == "fixed-input-native-sampler" ? interval.at("upper_95").get<double>() <= 1.01
                                                 : interval.at("lower_95").get<double>() >= 0.99;
}

struct observations {
    std::map<cell, std::map<std::string, json>> cells;
    std::map<std::pair<int, std::string>, int> workers;
    size_t rows = 0;
};

static observations read(const char * path, const json & fixtures, int blocks) {
    const std::set<std::string> expected_cases{
        "lazy-before-tool-trigger", "large-HTML-write-argument", "Unicode-and-partial-UTF8",
        "restricted-values-and-rejection-fallback", "closing-delimiters-and-EOS"};
    std::map<std::string, size_t> counts;
    check(fixtures.is_array() && fixtures.size() == 5, "invalid frozen fixtures");
    for (const auto & input : fixtures) {
        const auto name = input.at("name").get<std::string>();
        check(expected_cases.count(name) == 1, "unknown frozen case");
        check(counts.emplace(name, input.at("teacher_tokens").size()).second, "duplicate frozen case");
        check(counts.at(name) > 0, "empty frozen continuation");
    }
    const std::array<std::string, 4> orders{"BCRD", "CDBR", "DRCB", "RBDC"};
    std::set<std::tuple<int, std::string, std::string, std::string, int>> identities;
    std::map<int, std::pair<int, std::string>> owners;
    observations result;
    std::ifstream source(path);
    check(source.good(), "missing observations");
    std::string line;
    while (std::getline(source, line)) {
        const auto row = json::parse(line);
        const auto arm = row.at("arm").get<std::string>();
        const auto name = row.at("case").get<std::string>();
        const auto scope = row.at("scope").get<std::string>();
        const int block = row.at("block");
        const int iteration = row.at("worker_iteration");
        check(arm == "B" || arm == "C" || arm == "D" || arm == "R", "unknown arm");
        check(expected_cases.count(name) == 1, "unknown case");
        check(scope == "fixed-input-native-sampler" || scope == "teacher-forced-native-decode", "unknown scope");
        check(row.at("protocol") == "simd-top-k-v1", "wrong protocol");
        check(row.at("phase") == (blocks == 5 ? "screen" : "qualification"), "wrong phase");
        check(block >= 0 && block < blocks, "invalid global block");
        check(row.at("arm_order") == orders[static_cast<size_t>(block) % orders.size()], "wrong arm order");
        check(iteration >= -3 && iteration <= 0 && row.at("warmup") == (iteration < 0), "wrong warmup identity");
        check(identities.emplace(block, arm, name, scope, iteration).second, "duplicate row");
        const int pid = row.at("worker_pid");
        check(pid > 0, "invalid PID");
        const auto owner = std::make_pair(block, arm);
        const auto [worker, inserted] = result.workers.emplace(owner, pid);
        check(inserted || worker->second == pid, "worker changed within an arm/block");
        const auto [process, fresh] = owners.emplace(pid, owner);
        check(fresh || process->second == owner, "worker reused across arms/blocks");
        const int repetitions = scope == "fixed-input-native-sampler" ? 4 : 1;
        check(row.at("repetitions") == repetitions, "wrong repetition count");
        check(row.at("teacher_tokens") == counts.at(name), "wrong frozen continuation length");
        for (const char * field : {"thermal_before", "thermal_after"}) {
            const int thermal = row.at(field);
            check(thermal >= 0 && thermal <= 1, "thermal pressure or unknown thermal state");
        }
        const auto times = row.at("times_ns").get<std::vector<int64_t>>();
        const auto expected = counts.at(name) * repetitions;
        check(times.size() == expected && row.at("selected_tokens").size() == expected, "incomplete token observations");
        check(std::all_of(times.begin(), times.end(), [](auto ns) { return ns > 0; }), "nonpositive token time");
        for (const auto & token : row.at("selected_tokens")) {
            check(token.is_number_integer() && token.get<int64_t>() >= 0 && token.get<int64_t>() < 248320, "invalid token decision");
        }
        check(row.at("peak_rss_bytes").get<int64_t>() > 0, "invalid RSS");
        if (iteration == 0) {
            check(result.cells[{name, scope, block}].emplace(arm, row).second, "duplicate measured cell");
        }
        ++result.rows;
    }
    check(result.rows == static_cast<size_t>(blocks * 4 * 4 * 5 * 2), "missing observations or warmups");
    check(result.workers.size() == static_cast<size_t>(blocks * 4), "missing fresh workers");
    check(result.cells.size() == static_cast<size_t>(blocks * 5 * 2), "missing measured cells");
    return result;
}

static json analyze(const observations & data, int blocks) {
    std::map<comparison, std::vector<double>> ratios;
    std::map<std::pair<std::string, std::string>, std::vector<int64_t>> token_times;
    std::map<std::pair<std::string, std::string>, std::vector<double>> rates;
    std::map<std::string, int64_t> rss;
    for (const auto & [key, arms] : data.cells) {
        const auto & [name, scope, block] = key;
        (void)block;
        check(arms.size() == 4, "missing paired arm");
        std::map<std::string, double> means;
        for (const auto & [arm, row] : arms) {
            check(row.at("selected_tokens") == arms.at("B").at("selected_tokens"), "frozen B decision mismatch");
            const auto times = row.at("times_ns").get<std::vector<int64_t>>();
            means[arm] = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
            auto & all = token_times[{arm, scope}];
            all.insert(all.end(), times.begin(), times.end());
            rates[{arm, scope}].push_back(1e9 / means.at(arm));
            rss[arm] = std::max(rss[arm], row.at("peak_rss_bytes").get<int64_t>());
        }
        for (const auto & pair : {std::string("DC"), std::string("DB"), std::string("RC"), std::string("CB")}) {
            const double numerator = means.at(pair.substr(0, 1));
            const double denominator = means.at(pair.substr(1, 1));
            const double ratio = scope == "fixed-input-native-sampler" ? numerator / denominator : denominator / numerator;
            ratios[{name, scope, pair}].push_back(std::log(ratio));
        }
    }
    json per_case = json::array();
    std::map<std::pair<std::string, std::string>, std::vector<double>> overall;
    bool no_regression = true;
    for (const auto & [key, logs] : ratios) {
        const auto & [name, scope, pair] = key;
        check(logs.size() == static_cast<size_t>(blocks), "missing paired blocks");
        const auto interval = confidence(logs);
        const bool gate = pair == "DC" || pair == "DB";
        if (gate) { no_regression = no_regression && within(scope, interval); }
        per_case.push_back({{"case", name}, {"scope", scope}, {"comparison", pair}, {"confidence", interval},
                            {"qualification_comparison", gate}});
        auto & aggregate = overall[{scope, pair}];
        if (aggregate.empty()) { aggregate.resize(blocks, 0.0); }
        for (int i = 0; i < blocks; ++i) { aggregate[i] += logs[i] / 5.0; }
    }
    json aggregates = json::array();
    bool improves = false;
    for (const auto & [key, logs] : overall) {
        const auto & [scope, pair] = key;
        const auto interval = confidence(logs);
        if (pair == "DC" || pair == "DB") { no_regression = no_regression && within(scope, interval); }
        if (scope == "fixed-input-native-sampler" && pair == "DC") { improves = interval.at("upper_95").get<double>() <= 0.95; }
        aggregates.push_back({{"scope", scope}, {"comparison", pair}, {"confidence", interval}});
    }
    json summaries = json::array();
    for (auto & [key, times] : token_times) {
        const auto & [arm, scope] = key;
        std::sort(times.begin(), times.end());
        double lograte = 0;
        for (double rate : rates.at(key)) { lograte += std::log(rate); }
        summaries.push_back({{"arm", arm}, {"scope", scope}, {"p50_ns", times[times.size() / 2]},
            {"p95_ns", times[(times.size() - 1) * 95 / 100]},
            {"geomean_tokens_s", std::exp(lograte / rates.at(key).size())}, {"peak_rss_bytes", rss.at(arm)}});
    }
    return {{"schema", 1}, {"blocks", blocks}, {"rows", data.rows}, {"complete", true}, {"all_decisions_match_B", true},
            {"within_one_percent", no_regression}, {"sampler_improves_at_least_five_percent", improves},
            {"screen_pass", blocks == 5 && no_regression && improves},
            {"qualified_speed", blocks == 30 && no_regression && improves},
            {"per_case", per_case}, {"aggregate", aggregates}, {"arms", summaries},
            {"remaining_gates", {"external source/model/input/binary audit", "full correctness", "selected ISA evidence",
                                  "placement and thermal audit", "independent statistics check", "shared MLflow readback"}}};
}

int main(int argc, char ** argv) {
    try {
        check(argc == 5, "usage: simd-audit OBSERVATIONS FROZEN_FIXTURES EXPECTED_BLOCKS OUTPUT");
        const int blocks = std::stoi(argv[3]);
        check(blocks == 5 || blocks == 30, "unexpected block count");
        std::ifstream manifest(argv[2]);
        const auto fixtures = json::parse(manifest);
        const auto result = analyze(read(argv[1], fixtures, blocks), blocks);
        std::ofstream out(argv[4]);
        out.exceptions(std::ios::badbit | std::ios::failbit);
        out << result.dump(2) << '\n';
        std::cout << result.dump(2) << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "simd-audit: " << error.what() << '\n';
        return 1;
    }
}
