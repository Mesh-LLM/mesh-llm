#include "json.hpp"

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

struct worker {
    pid_t pid = -1;
    FILE * input = nullptr;
    FILE * output = nullptr;
    json ready;

    worker(const char * executable, const char * model, const char * directory) {
        int to_child[2], from_child[2];
        if (pipe(to_child) || pipe(from_child)) { throw std::runtime_error("pipe"); }
        pid = fork();
        if (pid < 0) { throw std::runtime_error("fork"); }
        if (pid == 0) {
            dup2(to_child[0], STDIN_FILENO);
            dup2(from_child[1], STDOUT_FILENO);
            close(to_child[0]); close(to_child[1]);
            close(from_child[0]); close(from_child[1]);
            execl(executable, executable, model, "serve", directory, static_cast<char *>(nullptr));
            _exit(127);
        }
        close(to_child[0]); close(from_child[1]);
        input = fdopen(to_child[1], "w");
        output = fdopen(from_child[0], "r");
        ready = read();
        if (!ready.value("ready", false)) { throw std::runtime_error("worker not ready"); }
    }

    json read() {
        char * line = nullptr;
        size_t capacity = 0;
        auto bytes = getline(&line, &capacity, output);
        if (bytes < 0) { free(line); throw std::runtime_error("worker exited without result"); }
        auto value = json::parse(line, line + bytes);
        free(line);
        return value;
    }

    json call(const json & request) {
        auto text = request.dump();
        if (fprintf(input, "%s\n", text.c_str()) < 0 || fflush(input)) {
            throw std::runtime_error("worker command failed");
        }
        return read();
    }

    ~worker() {
        if (input) { fclose(input); }
        if (output) { fclose(output); }
        if (pid > 0) {
            int status = 0;
            waitpid(pid, &status, 0);
        }
    }
};

int main(int argc, char ** argv) {
    try {
        if (argc != 7) { throw std::runtime_error("usage: controller BASELINE CANDIDATE MODEL INPUT_DIR OUTPUT_DIR prerequisite|diagnostic|development|qualification"); }
        const std::string phase = argv[6];
        const bool diagnostic = phase == "diagnostic";
        if (phase != "prerequisite" && !diagnostic && phase != "development" && phase != "qualification") { throw std::runtime_error("unknown phase"); }
        const std::filesystem::path out = argv[5];
        std::filesystem::create_directories(out);
        worker baseline(argv[1], argv[3], argv[4]);
        worker candidate(argv[2], argv[3], argv[4]);
        if (baseline.ready.at("fixtures") != candidate.ready.at("fixtures")) { throw std::runtime_error("A/B/C fixture identity mismatch"); }
        std::ofstream(out / "admitted-fixtures.json") << baseline.ready.at("fixtures").dump(2) << '\n';
        if (phase == "prerequisite") {
            baseline.call({{"action", "capture"}});
            std::ofstream(out / "correctness.json") << candidate.call({{"action", "correctness"}}).dump(2) << '\n';
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
