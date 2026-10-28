#include "worker.h"

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

using json = nlohmann::ordered_json;


int main(int argc, char ** argv) {
    try {
        if (argc != 7) { throw std::runtime_error("usage: controller BASELINE CANDIDATE MODEL INPUT_DIR OUTPUT_DIR prerequisite|correctness|diagnostic|development|qualification"); }
        const std::string phase = argv[6];
        const bool diagnostic = phase == "diagnostic";
        if (phase != "prerequisite" && phase != "correctness" && !diagnostic && phase != "development" && phase != "qualification") { throw std::runtime_error("unknown phase"); }
        const std::filesystem::path out = argv[5];
        std::filesystem::create_directories(out);
        worker baseline(argv[1], argv[3], argv[4]);
        worker candidate(argv[2], argv[3], argv[4]);
        if (baseline.ready.at("fixtures") != candidate.ready.at("fixtures")) { throw std::runtime_error("A/B/C fixture identity mismatch"); }
        std::ofstream(out / "admitted-fixtures.json") << baseline.ready.at("fixtures").dump(2) << '\n';
        if (phase == "prerequisite" || phase == "correctness") {
            if (phase == "prerequisite") { baseline.call({{"action", "capture"}}); }
            std::ofstream(out / "correctness.json") << candidate.call({{"action", "correctness"}}).dump(2) << '\n';
            candidate.finish();
            baseline.finish();
            return 0;
        }
        const int blocks = diagnostic ? 6 : (phase == "development" ? 5 : 30);
        const std::array<std::string, 6> orders = diagnostic
            ? std::array<std::string, 6>{"BRC", "RCB", "CBR", "CRB", "BCR", "RBC"}
            : std::array<std::string, 6>{"ABC", "BCA", "CAB", "CBA", "ACB", "BAC"};
        std::ofstream results(out / "observations.jsonl");
        for (int block = -3; block < blocks; ++block) {
            const auto & order = orders[static_cast<size_t>(block + 6) % orders.size()];
            for (bool decode : {false, true}) {
                if (diagnostic && decode) { continue; }
                for (const auto & input : baseline.ready.at("fixtures")) {
                    for (char arm : order) {
                        auto & target = arm == 'C' || arm == 'R' ? candidate : baseline;
                        const std::string implementation(1, arm == 'R' ? 'B' : arm);
                        auto result = target.call({{"action", "measure"}, {"arm", implementation},
                                                   {"case", input.at("name")}, {"decode", decode},
                                                   {"repetitions", diagnostic ? 20 : (decode ? 1 : 4)}, {"block", block}, {"warmup", block < 0}});
                        if (diagnostic) {
                            result["arm"] = std::string(1, arm);
                            result["implementation"] = implementation;
                            result["worker"] = arm == 'B' ? "baseline" : "candidate";
                        }
                        results << result << '\n';
                        results.flush();
                        std::cerr << phase << " block=" << block << " arm=" << arm << " case=" << input.at("name")
                                  << " scope=" << (decode ? "decode" : "sampler") << std::endl;
                    }
                }
            }
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "grammar-controller: " << error.what() << '\n';
        return 1;
    }
}
