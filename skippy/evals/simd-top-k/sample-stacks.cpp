#include "json.hpp"

#include <chrono>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

using json = nlohmann::ordered_json;
using timer = std::chrono::steady_clock;

static void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

class debugger {
    pid_t pid = -1;
    int input = -1;
    int output = -1;
    std::string pending;
    std::ofstream & transcript;
    int token = 0;

public:
    bool stopped = false;
    bool exited = false;

    debugger(char ** argv, std::ofstream & log) : transcript(log) {
        int commands[2], replies[2];
        require(pipe(commands) == 0 && pipe(replies) == 0, "debugger pipe");
        pid = fork();
        require(pid >= 0, "debugger fork");
        if (pid == 0) {
            setpgid(0, 0);
            dup2(commands[0], STDIN_FILENO);
            dup2(replies[1], STDOUT_FILENO);
            dup2(replies[1], STDERR_FILENO);
            close(commands[0]); close(commands[1]); close(replies[0]); close(replies[1]);
            std::vector<char *> args{argv[0], const_cast<char *>("--quiet"), const_cast<char *>("--nx"),
                const_cast<char *>("--interpreter=mi2"), const_cast<char *>("--args")};
            for (auto ** p = argv + 1; *p; ++p) { args.push_back(*p); }
            args.push_back(nullptr);
            execvp(args[0], args.data());
            _exit(127);
        }
        close(commands[0]); close(replies[1]);
        input = commands[1]; output = replies[0];
    }

    ~debugger() {
        if (input >= 0) { close(input); }
        if (output >= 0) { close(output); }
        if (pid > 0) {
            // Only this debugger and its owned inferior are in this group.
            kill(-pid, SIGKILL);
            waitpid(pid, nullptr, 0);
        }
    }

    std::string line(timer::time_point deadline) {
        while (pending.find('\n') == std::string::npos) {
            require(timer::now() < deadline, "debugger response deadline");
            pollfd fd{output, POLLIN, 0};
            const int ready = poll(&fd, 1, 100);
            require(ready >= 0, "debugger poll");
            if (ready == 0) { continue; }
            char buffer[8192];
            const auto count = read(output, buffer, sizeof(buffer));
            require(count > 0, "debugger closed before reply");
            pending.append(buffer, static_cast<size_t>(count));
            require(pending.size() < 8 * 1024 * 1024, "oversized debugger reply");
        }
        const auto end = pending.find('\n');
        auto result = pending.substr(0, end);
        pending.erase(0, end + 1);
        transcript << result << '\n';
        transcript.flush();
        if (result.rfind("*stopped", 0) == 0) {
            stopped = true;
            exited = result.find("reason=\"exited") != std::string::npos;
        }
        return result;
    }

    std::string command(const std::string & value, timer::time_point deadline) {
        const auto id = std::to_string(++token);
        const auto request = id + value + '\n';
        require(write(input, request.data(), request.size()) == static_cast<ssize_t>(request.size()), "debugger write");
        for (;;) {
            auto reply = line(deadline);
            if (reply.rfind(id + '^', 0) == 0) {
                if (exited && value == "-exec-interrupt --all") { return reply; }
                require(reply.rfind(id + "^error", 0) != 0, "debugger rejected command; see transcript");
                return reply;
            }
        }
    }

    void await_stop(timer::time_point deadline) {
        while (!stopped) { line(deadline); }
    }
};

__attribute__((noinline)) static void calibration_child() {
    volatile uint64_t state = 73;
    const auto end = timer::now() + std::chrono::seconds(3);
    while (timer::now() < end) {
        for (int i = 0; i < 10000; ++i) { state = state * 1664525 + 1013904223; }
    }
}

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--calibration-child") {
            calibration_child();
            return 0;
        }
        require(argc >= 5, "usage: simd-sample-stacks OUTPUT_JSON TRANSCRIPT GDB PROGRAM [ARG...]");
        signal(SIGPIPE, SIG_IGN);
        std::ofstream log(argv[2]);
        require(log.good(), "cannot create debugger transcript");
        debugger child(argv + 3, log);
        const auto deadline = timer::now() + std::chrono::seconds(180);
        child.command("-gdb-set pagination off", deadline);
        child.command("-gdb-set confirm off", deadline);
        child.command("-gdb-set disable-randomization off", deadline);
        child.command("-gdb-set mi-async on", deadline);
        child.command("-exec-run", deadline);
        json stacks = json::array();
        for (int sample = 0; sample < 3000 && timer::now() < deadline; ++sample) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            if (!child.stopped) { child.command("-exec-interrupt --all", deadline); }
            child.await_stop(deadline);
            if (child.exited) { break; }
            const auto frames = child.command("-stack-list-frames --thread 1 0 31", deadline);
            stacks.push_back({{"sample", sample}, {"mi_frames", frames}});
            child.stopped = false;
            child.command("-exec-continue", deadline);
        }
        if (!child.exited) {
            if (!child.stopped) { child.command("-exec-interrupt --all", deadline); }
            child.await_stop(deadline);
        }
        child.command("-gdb-exit", deadline);
        std::ofstream out(argv[1]);
        require(out.good(), "cannot create stack samples");
        out << json({{"scope", "diagnostic interrupted execution; never qualification timing"},
            {"interval_ms", 20}, {"sample_limit", 3000}, {"inferior_exited", child.exited},
            {"stacks", stacks}}).dump(2) << '\n';
        require(!stacks.empty(), "no stack samples collected");
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "simd-sample-stacks: " << error.what() << '\n';
        return 1;
    }
}
