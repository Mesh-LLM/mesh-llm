#include "llvm/Support/JSON.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

class FixtureDirectory {
public:
  FixtureDirectory() {
    llvm::SmallString<128> created;
    auto error = llvm::sys::fs::createUniqueDirectory("skippy-rewriter-fixture", created);
    if (error) throw std::runtime_error("cannot create temporary fixture directory: " + error.message());
    path = fs::path(created.str().str());
  }

  ~FixtureDirectory() {
    std::error_code error;
    fs::remove_all(path, error);
  }

  FixtureDirectory(const FixtureDirectory &) = delete;
  FixtureDirectory &operator=(const FixtureDirectory &) = delete;

  fs::path path;
};

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

std::string read(const fs::path &path) {
  std::ifstream input(path);
  require(input.good(), "cannot read " + path.string());
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

const llvm::json::Object &object(const llvm::json::Value &value) {
  auto *result = value.getAsObject();
  require(result != nullptr, "expected JSON object");
  return *result;
}

const llvm::json::Value &field(const llvm::json::Object &record, llvm::StringRef key) {
  auto *value = record.get(key);
  require(value != nullptr, "missing field " + key.str());
  return *value;
}

std::string text(const llvm::json::Object &record, llvm::StringRef key) {
  auto value = field(record, key).getAsString();
  require(value.has_value(), "invalid string " + key.str());
  return value->str();
}

const llvm::json::Array &array(const llvm::json::Object &record, llvm::StringRef key) {
  auto *value = field(record, key).getAsArray();
  require(value != nullptr, "invalid array " + key.str());
  return *value;
}

bool contains(const llvm::json::Array &values, llvm::StringRef expected) {
  for (const auto &value : values) {
    auto string = value.getAsString();
    if (string && *string == expected) return true;
  }
  return false;
}

std::vector<std::string> editKinds(const llvm::json::Object &record) {
  std::vector<std::string> kinds;
  for (const auto &edit : array(record, "edits")) kinds.push_back(text(object(edit), "kind"));
  return kinds;
}

llvm::json::Value invoke(const fs::path &tool, const fs::path &sourceRoot,
                         const fs::path &report, const std::string &sourceName,
                         bool apply = false) {
  std::string executable = tool.string();
  std::string source = (sourceRoot / "src/models" / sourceName).string();
  std::string root = sourceRoot.string();
  std::string output = report.string();
  std::vector<llvm::StringRef> arguments{executable, "--source-root", root,
                                         "--llama-commit", "fixture", "--report", output};
  if (apply) arguments.push_back("--apply");
  arguments.insert(arguments.end(), {source, "--", "-std=c++17"});
  require(llvm::sys::ExecuteAndWait(executable, arguments) == 0,
          "rewriter failed for " + sourceName);
  auto parsed = llvm::json::parse(read(report));
  require(static_cast<bool>(parsed), "invalid report for " + sourceName);
  return std::move(*parsed);
}

const llvm::json::Object &builder(const llvm::json::Value &report) {
  const auto &builders = array(object(report), "builders");
  require(builders.size() == 1, "expected exactly one builder");
  return object(builders[0]);
}

void check(const fs::path &tool, const fs::path &root, const fs::path &reports,
           const std::string &name, const std::string &verdict,
           const std::vector<std::string> &edits = {}, const std::string &reason = "") {
  auto report = invoke(tool, root, reports / (name + ".json"), name);
  const auto &record = builder(report);
  require(text(record, "verdict") == verdict, name + ": wrong verdict");
  require(editKinds(record) == edits, name + ": wrong edits");
  if (!reason.empty()) require(text(record, "unsupported_reason") == reason, name + ": wrong refusal");
  std::cout << "PASS " << name << '\n';
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: skippy-stage-rewriter-fixture TOOL FIXTURE_ROOT\n";
    return 2;
  }
  try {
    FixtureDirectory fixture;
    const auto &temp = fixture.path;
    auto root = temp / "source";
    fs::copy(argv[2], root, fs::copy_options::recursive);
    auto tool = fs::absolute(argv[1]);

    auto conventionalReport = invoke(tool, root, temp / "conventional.json", "conventional.cpp");
    const auto &conventional = builder(conventionalReport);
    require(text(conventional, "verdict") == "transformable", "conventional verdict");
    const auto &proof = object(field(conventional, "proof"));
    const auto &loop = object(field(proof, "loop"));
    require(text(loop, "end") == "n_layer" && text(loop, "start") == "0" &&
                text(loop, "var") == "il", "conventional loop");
    require(text(proof, "activation_in") == "inpL" &&
                text(proof, "activation_out") == "inpL", "conventional activation");
    require(editKinds(conventional) == std::vector<std::string>{"insert_begin_block", "insert_end_block"}, "conventional edits");
    invoke(tool, root, temp / "conventional-applied.json", "conventional.cpp", true);
    auto transformed = read(root / "src/models/conventional.cpp");
    for (auto expected : {"begin_block(inpL, il);", "end_block(inpL, il);", "for (int il = 0; il < n_layer; ++il)"})
      require(transformed.find(expected) != std::string::npos, "missing conventional transformation");
    auto second = invoke(tool, root, temp / "conventional-second.json", "conventional.cpp");
    require(text(builder(second), "verdict") == "already_transformed" &&
                editKinds(builder(second)).empty(), "second pass must have zero edits");
    std::cout << "PASS conventional apply and idempotence\n";

    for (const auto &entry : {std::pair{"continue-path.cpp", "insert_end_block_before_continue"},
                              {"continue-unbraced.cpp", "wrap_end_block_before_continue"}}) {
      auto report = invoke(tool, root, temp / (std::string(entry.first) + ".json"), entry.first);
      const auto &record = builder(report);
      require(text(record, "verdict") == "transformable", "continue verdict");
      auto kinds = editKinds(record);
      require(std::count(kinds.begin(), kinds.end(), entry.second) == 1, "continue edit count");
      invoke(tool, root, temp / (std::string(entry.first) + "-applied.json"), entry.first, true);
    }
    require(read(root / "src/models/continue-path.cpp").find("end_block(inpL, il);\n      continue;") != std::string::npos, "continued source");
    require(read(root / "src/models/continue-unbraced.cpp").find("if (il == 2) {\n        end_block(inpL, il);\n        continue;\n    }") != std::string::npos, "unbraced source");
    std::cout << "PASS braced and unbraced continue\n";

    for (const auto &entry : {std::pair{"glm-dsa.cpp", "glm_dsa_top_k_sideband"},
                              {"kimi-k3.cpp", "kimi_k3_residual_sideband"},
                              {"hyperconnection.cpp", "hyperconnection_activation_frontier"},
                              {"rwkv-first-value.cpp", "rwkv_first_value_sideband"}}) {
      auto report = invoke(tool, root, temp / (std::string(entry.first) + ".json"), entry.first);
      const auto &record = builder(report);
      require(text(record, "verdict") == "transformable" &&
                  contains(array(object(field(record, "proof")), "scope_evidence"), entry.second) &&
                  editKinds(record) == std::vector<std::string>{"insert_begin_block", "insert_end_block"},
              std::string(entry.first) + ": scope/edit mismatch");
      std::cout << "PASS " << entry.first << '\n';
    }
    auto auxiliary = invoke(tool, root, temp / "auxiliary.json", "auxiliary.cpp");
    require(text(builder(auxiliary), "verdict") == "supported_auxiliary" &&
                text(object(field(builder(auxiliary), "proof")), "execution_scope") == "final_stage_sidecar" &&
                editKinds(builder(auxiliary)).empty(), "auxiliary scope");
    std::cout << "PASS auxiliary.cpp\n";
    check(tool, root, temp, "multiple-domains.cpp", "supported_whole_model");
    auto delegated = invoke(tool, root, temp / "delegated-stacks.json", "delegated-stacks.cpp");
    const auto &delegatedRecord = builder(delegated);
    const auto &delegatedProof = object(field(delegatedRecord, "proof"));
    require(text(delegatedRecord, "verdict") == "supported_whole_model" &&
                text(delegatedProof, "execution_scope") == "multiple_sequential_layer_domains" &&
                array(delegatedProof, "scope_evidence").size() == 1 &&
                contains(array(delegatedProof, "scope_evidence"), "model.n_layers_per_stack") &&
                editKinds(delegatedRecord).empty(), "delegated scope");
    std::cout << "PASS delegated-stacks.cpp\n";
    check(tool, root, temp, "delegated-opaque.cpp", "unsupported_shape", {}, "no layer block loop");
    check(tool, root, temp, "filter-only.cpp", "unsupported_shape", {}, "legacy model-local stage filter is not supported");
    check(tool, root, temp, "two-loops.cpp", "unsupported_shape", {}, "multiple equally ranked layer block loops");
    check(tool, root, temp, "nonlocal-exit.cpp", "unsupported_shape", {}, "block loop contains a non-local exit");
    check(tool, root, temp, "embedding-prelude-else.cpp", "unsupported_shape", {}, "pre-loop activation conditional has an else branch");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "fixture failure: " << error.what() << '\n';
    return 1;
  }
}
