#pragma once

#include "json.hpp"

#include <cstdio>
#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <stdexcept>
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
        // A later worker must not inherit another worker's pipe endpoints:
        // otherwise reversed startup/destruction order can prevent EOF forever.
        for (int fd : {to_child[0], to_child[1], from_child[0], from_child[1]}) {
            if (fcntl(fd, F_SETFD, FD_CLOEXEC) < 0) { throw std::runtime_error("pipe close-on-exec"); }
        }
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

    void finish() {
        if (input) { fclose(input); input = nullptr; }
        if (output) { fclose(output); output = nullptr; }
        if (pid > 0) {
            int status = 0;
            pid_t exited;
            do { exited = waitpid(pid, &status, 0); } while (exited < 0 && errno == EINTR);
            pid = -1;
            if (exited < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                throw std::runtime_error("worker failed during shutdown");
            }
        }
    }

    ~worker() {
        try { finish(); } catch (...) { }
    }
};

