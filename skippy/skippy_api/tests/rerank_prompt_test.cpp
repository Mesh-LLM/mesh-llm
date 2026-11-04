#include "rerank.h"

#include <gtest/gtest.h>

#include <algorithm>

#include <string>
#include <string_view>

namespace {

using skippy::build_rerank_prompt;
using skippy::RerankPromptFormat;

struct FallbackCase {
    const char* name;
    std::string_view eos;
    std::string_view sep;
    std::string_view expected;
};

class RerankPromptFallback : public ::testing::TestWithParam<FallbackCase> {};

TEST_P(RerankPromptFallback, ComposesExactlyThePiecesTheVocabAdds) {
    const RerankPromptFormat format{
        .prompt_template = std::nullopt,
        .eos_piece = GetParam().eos,
        .sep_piece = GetParam().sep,
    };

    EXPECT_EQ(build_rerank_prompt(format, "query", "document"), GetParam().expected)
        << GetParam().name;
}

INSTANTIATE_TEST_SUITE_P(
    Combinations, RerankPromptFallback,
    ::testing::Values(FallbackCase{"neither", "", "", "querydocument"},
                      FallbackCase{"eos only", "</s>", "", "query</s>document"},
                      FallbackCase{"sep only", "", "[SEP]", "query[SEP]document"},
                      FallbackCase{"both, eos first", "</s>", "[SEP]", "query</s>[SEP]document"}),
    [](const ::testing::TestParamInfo<FallbackCase>& info) {
        std::string id = info.param.name;
        std::ranges::replace_if(id, [](char c) { return c == ' ' || c == ','; }, '_');
        return id;
    });

TEST(RerankPrompt, TemplateWinsOverPieces) {
    const RerankPromptFormat format{
        .prompt_template = "<q>{query}</q><d>{document}</d>",
        .eos_piece = "</s>",
        .sep_piece = "[SEP]",
    };

    EXPECT_EQ(build_rerank_prompt(format, "a", "b"), "<q>a</q><d>b</d>");
}

TEST(RerankPrompt, EmptyTemplateIsStillATemplate) {
    const RerankPromptFormat format{
        .prompt_template = std::string_view{}, .eos_piece = "</s>", .sep_piece = ""};

    EXPECT_EQ(build_rerank_prompt(format, "a", "b"), "");
}

TEST(RerankPrompt, EmptyQueryAndDocumentStillGetThePieces) {
    const RerankPromptFormat format{.eos_piece = "</s>", .sep_piece = "[SEP]"};

    EXPECT_EQ(build_rerank_prompt(format, "", ""), "</s>[SEP]");
}

static_assert(build_rerank_prompt({.eos_piece = "</s>", .sep_piece = "[SEP]"}, "q", "d") ==
              "q</s>[SEP]d");
static_assert(
    build_rerank_prompt({.prompt_template = "{document}|{query}", .eos_piece = "", .sep_piece = ""},
                        "q", "d") == "d|q");

} // namespace
