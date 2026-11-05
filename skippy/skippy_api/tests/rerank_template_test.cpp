#include "rerank_template.h"

#include <gtest/gtest.h>

namespace {

using skippy::detail::render_rerank_template;

// these are the tests coming from the existing patch (10)
TEST(RerankTemplate, SubstitutesMarkersWithoutRescanningTheSubstitutions) {
    EXPECT_EQ(render_rerank_template("Q={query}|D={document}|Q={query}", "literal {document}",
                                     "literal {query}"),
              "Q=literal {document}|D=literal {query}|Q=literal {document}");
}

TEST(RerankTemplate, AdjacentMarkers) {
    EXPECT_EQ(render_rerank_template("{query}{document}", "a", "b"), "ab");
}

TEST(RerankTemplate, TemplateWithoutMarkersIsCopiedThrough) {
    EXPECT_EQ(render_rerank_template("no markers", "a", "b"), "no markers");
}

TEST(RerankTemplate, DocumentBeforeQuery) {
    EXPECT_EQ(render_rerank_template("{document} <> {query}", "q", "d"), "d <> q");
}

TEST(RerankTemplate, MarkerAtStartAndEnd) {
    EXPECT_EQ(render_rerank_template("{query}", "q", "d"), "q");
    EXPECT_EQ(render_rerank_template("tail {document}", "q", "d"), "tail d");
    EXPECT_EQ(render_rerank_template("{document} head", "q", "d"), "d head");
}

TEST(RerankTemplate, EmptyInputs) {
    EXPECT_EQ(render_rerank_template("", "q", "d"), "");
    EXPECT_EQ(render_rerank_template("[{query}|{document}]", "", ""), "[|]");
}

TEST(RerankTemplate, UnterminatedMarkerIsLiteralText) {
    EXPECT_EQ(render_rerank_template("{query is not a marker: {query}", "q", "d"),
              "{query is not a marker: q");
    EXPECT_EQ(render_rerank_template("{document", "q", "d"), "{document");
}

TEST(RerankTemplate, PartialOverlapTakesTheRealMarker) {
    EXPECT_EQ(render_rerank_template("{query{document}}", "q", "d"), "{queryd}");
}

TEST(RerankTemplate, MarkerNamesAreCaseSensitive) {
    EXPECT_EQ(render_rerank_template("{Query} {DOCUMENT}", "q", "d"), "{Query} {DOCUMENT}");
}

TEST(RerankTemplate, RepeatedMarkersExpandEveryTime) {
    EXPECT_EQ(render_rerank_template("{document}{document}{query}{query}", "q", "d"), "ddqq");
}

TEST(RerankTemplate, SubstitutionsMayContainAnyBytes) {
    const std::string query{"a\0b", 3};
    const auto rendered = render_rerank_template("<{query}>", query, "");
    EXPECT_EQ(rendered.size(), 5U);
    EXPECT_EQ(rendered, (std::string{"<a\0b>", 5}));
}

static_assert(render_rerank_template("Q={query}|D={document}|Q={query}", "literal {document}",
                                     "literal {query}") ==
              "Q=literal {document}|D=literal {query}|Q=literal {document}");
static_assert(render_rerank_template("{query}{document}", "a", "b") == "ab");
static_assert(render_rerank_template("no markers", "a", "b") == "no markers");

} // namespace
