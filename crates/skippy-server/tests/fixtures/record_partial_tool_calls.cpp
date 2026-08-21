// Record the exact sequence of partial tool_calls arrays that
// common_chat_parse(text, is_partial=true, ...) produces for a qwen35
// generation, so a Rust unit test can be driven by recorded parser output
// rather than a hand transcription.
//
// Mirrors skippy_parse_chat_response_json (patch 0010):
//   common_chat_parse(generated_text, is_partial, params)
//     -> common_chat_msgs_to_json_oaicompat({msg}).at(0).dump()

#include "chat.h"
#include "log.h"
#include <nlohmann/json.hpp>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

static std::string read_file(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        exit(1);
    }
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char ** argv) {
    common_log_set_verbosity_thold(-100);
    const std::string tmpl_path = argc > 1 ? argv[1] : "models/templates/Qwen3.5-4B.jinja";

    common_chat_templates_ptr tmpls = common_chat_templates_init(nullptr, read_file(tmpl_path));

    common_chat_templates_inputs inputs;
    inputs.use_jinja = true;
    inputs.messages = { [] {
        common_chat_msg m;
        m.role = "user";
        m.content = "read AGENTS.md then list the dir";
        return m;
    }() };
    inputs.tools = {
        { "read_file", "Read a file from disk",
          R"({"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer"}},"required":["path"]})" },
        { "list_dir", "List a directory",
          R"({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]})" },
    };
    inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    inputs.parallel_tool_calls = true;
    inputs.enable_thinking = false;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;

    common_chat_params params = common_chat_templates_apply(tmpls.get(), inputs);

    common_chat_parser_params pp;
    pp.format = params.format;
    pp.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    pp.generation_prompt = params.generation_prompt;
    pp.parse_tool_calls = true;
    if (!params.parser.empty()) {
        pp.parser.load(params.parser);
    }

    struct Fixture { const char * name; std::string text; };
    std::vector<Fixture> fixtures = {
        { "single_call",
          "Let me read that file.\n"
          "<tool_call>\n"
          "<function=read_file>\n"
          "<parameter=path>\n/tmp/AGENTS.md\n</parameter>\n"
          "<parameter=limit>\n200\n</parameter>\n"
          "</function>\n"
          "</tool_call>" },
        { "parallel_calls",
          "<tool_call>\n"
          "<function=read_file>\n"
          "<parameter=path>\n/tmp/a.md\n</parameter>\n"
          "</function>\n"
          "</tool_call>\n"
          "<tool_call>\n"
          "<function=list_dir>\n"
          "<parameter=path>\n/tmp\n</parameter>\n"
          "</function>\n"
          "</tool_call>" },
    };

    printf("# Recorded output of common_chat_parse(prefix, is_partial=true) for qwen35\n");
    printf("# tool-call generations, captured through the same call shape\n");
    printf("# skippy_parse_chat_response_json uses (patch 0010).\n");
    printf("#   chat format: %s\n", common_chat_format_name(params.format));
    printf("#   template:    %s\n", tmpl_path.c_str());
    printf("# Regenerate with the build command in this directory's README.\n");
    printf("# Format: one record per line, \"<fixture> <prefix_len> <tool_calls JSON>\".\n");
    printf("# Only prefixes whose parsed tool_calls array CHANGED are recorded; \"final\"\n");
    printf("# is the terminal is_partial=false parse.\n");

    for (const auto & fx : fixtures) {
        std::string last;
        for (size_t i = 1; i <= fx.text.size(); ++i) {
            try {
                common_chat_msg msg = common_chat_parse(fx.text.substr(0, i), true, pp);
                json j = common_chat_msgs_to_json_oaicompat({ msg }).at(0);
                if (!j.contains("tool_calls")) {
                    continue;
                }
                const std::string dump = j.at("tool_calls").dump();
                if (dump == last) {
                    continue;
                }
                last = dump;
                printf("%s %zu %s\n", fx.name, i, dump.c_str());
            } catch (const std::exception & e) {
                printf("%s %zu THROW %s\n", fx.name, i, e.what());
            }
        }
        try {
            common_chat_msg msg = common_chat_parse(fx.text, false, pp);
            json j = common_chat_msgs_to_json_oaicompat({ msg }).at(0);
            printf("%s final %s\n", fx.name,
                   (j.contains("tool_calls") ? j.at("tool_calls").dump() : std::string("[]")).c_str());
        } catch (const std::exception & e) {
            printf("%s final THROW %s\n", fx.name, e.what());
        }
    }

    return 0;
}
