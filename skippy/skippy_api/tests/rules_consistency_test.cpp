// Gap A: the rules can contradict each other and nothing noticed (P1-2: a
// code rule cited as *the pattern* the shape a trap forbids). This test parses
// AGENTS.md and TYPE-RULES.md for the role tags every symbol citation must
// carry — (pattern: `X`) / (anti-pattern: `X`) — and for backtick-quoted
// paths, and fails when:
//  1. the same symbol is tagged pattern in one place and anti-pattern in
//     another;
//  2. a cited path no longer exists in the tree;
//  3. an anti-pattern symbol still exists under include/ or src/.
// The parser is pinned by unit tests on synthetic text, so each failure class
// is provable without depending on the docs' current state; the tree
// assertions then run against the real files.
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

namespace {

namespace fs = std::filesystem;

struct Tag {
    std::string symbol;
    std::string kind; // "pattern" | "anti-pattern"
    std::string file;
    int line = 0;
};

std::vector<Tag> parse_tags(std::string_view text, std::string_view file) {
    std::vector<Tag> tags;
    int line = 1;
    std::string_view::size_type at = 0;
    while (at < text.size()) {
        const auto eol = text.find('\n', at);
        const std::string_view row =
            text.substr(at, eol == std::string_view::npos ? std::string_view::npos : eol - at);
        for (const std::string_view kind : {"pattern", "anti-pattern"}) {
            std::string_view::size_type cursor = 0;
            const std::string marker = std::string(kind) + ": ``";
            const std::string plain = std::string(kind) + ": `";
            while (true) {
                auto hit = row.find(plain, cursor);
                if (hit == std::string_view::npos) break;
                // "anti-pattern:" contains "pattern:"; the inner hit is not a tag.
                if (kind == "pattern" && hit > 0 && row[hit - 1] == '-') {
                    cursor = hit + plain.size();
                    continue;
                }
                const auto start = hit + plain.size();
                const auto end = row.find('`', start);
                if (end == std::string_view::npos) break;
                if (row.compare(hit, marker.size(), marker) == 0) {
                    cursor = end + 1;
                    continue; // double-backtick markdown escape, not a tag
                }
                tags.push_back(Tag{std::string(row.substr(start, end - start)), std::string(kind),
                                   std::string(file), line});
                cursor = end + 1;
            }
        }
        if (eol == std::string_view::npos) break;
        ++line;
        at = eol + 1;
    }
    return tags;
}

std::vector<std::string> parse_paths(std::string_view text) {
    // Backtick-quoted tokens that look like tree paths (contain '/', end in a
    // known source or doc extension). Globs and URLs are not paths.
    std::vector<std::string> paths;
    std::string_view::size_type cursor = 0;
    while (true) {
        const auto start = text.find('`', cursor);
        if (start == std::string_view::npos) break;
        const auto end = text.find('`', start + 1);
        if (end == std::string_view::npos) break;
        const std::string_view token = text.substr(start + 1, end - start - 1);
        static constexpr std::string_view extensions[] = {".h",   ".hpp", ".cpp",  ".rs", ".md",
                                                          ".txt", ".sh",  ".json", ".c",  ".just"};
        for (const auto extension : extensions) {
            if (token.ends_with(extension) && token.find('/') != std::string_view::npos &&
                token.find('*') == std::string_view::npos &&
                token.find(' ') == std::string_view::npos && !token.starts_with("http")) {
                paths.emplace_back(token);
                break;
            }
        }
        cursor = end + 1;
    }
    return paths;
}

std::string read_file(const fs::path& path) {
    std::ifstream stream(path);
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

const fs::path& skippy_api_dir() {
    static const fs::path dir = fs::path(__FILE__).parent_path().parent_path();
    return dir;
}

fs::path resolve(const std::string& cited) {
    const fs::path base = skippy_api_dir();
    for (const fs::path& prefix : {base, base.parent_path(), base.parent_path().parent_path()}) {
        const fs::path candidate = prefix / cited;
        if (fs::exists(candidate)) return candidate;
    }
    return {};
}

TEST(RulesConsistency, ParserDetectsTagConflicts) {
    const auto tags = parse_tags("a (pattern: `X`) b\n c (anti-pattern: `X`) d", "synthetic");
    ASSERT_EQ(tags.size(), 2u);
    EXPECT_EQ(tags[0].kind, "pattern");
    EXPECT_EQ(tags[1].kind, "anti-pattern");
    EXPECT_EQ(tags[0].symbol, tags[1].symbol);
}

TEST(RulesConsistency, ParserExtractsCitedPaths) {
    const auto paths = parse_tags("", "x"); // not paths
    const auto cited = parse_paths(
        "see `tidy/test/no_mutex.cpp` and `just/skippy.just` but not ``code`` or `https://x/y.md`");
    ASSERT_EQ(cited.size(), 2u);
    EXPECT_EQ(cited[0], "tidy/test/no_mutex.cpp");
    EXPECT_EQ(cited[1], "just/skippy.just");
}

TEST(RulesConsistency, NoSymbolIsBothPatternAndAntiPattern) {
    std::vector<Tag> tags;
    for (const char* doc : {"AGENTS.md", "TYPE-RULES.md"}) {
        const auto more = parse_tags(read_file(skippy_api_dir() / doc), doc);
        tags.insert(tags.end(), more.begin(), more.end());
    }
    EXPECT_FALSE(tags.empty()) << "the docs carry no role tags at all";
    std::unordered_map<std::string, std::string> kind_of;
    for (const auto& tag : tags) {
        const auto existing = kind_of.find(tag.symbol);
        if (existing != kind_of.end() && existing->second != tag.kind) {
            ADD_FAILURE() << tag.symbol << " is tagged " << existing->second << " and " << tag.kind
                          << " (" << tag.file << ":" << tag.line << ")";
            continue;
        }
        kind_of[tag.symbol] = tag.kind;
    }
}

TEST(RulesConsistency, CitedPathsExist) {
    for (const char* doc : {"AGENTS.md", "TYPE-RULES.md"}) {
        for (const auto& cited : parse_paths(read_file(skippy_api_dir() / doc))) {
            EXPECT_FALSE(resolve(cited).empty())
                << doc << " cites a path that does not exist: " << cited;
        }
    }
}

TEST(RulesConsistency, AntiPatternSymbolsAreGoneFromSources) {
    std::vector<Tag> tags;
    for (const char* doc : {"AGENTS.md", "TYPE-RULES.md"}) {
        const auto more = parse_tags(read_file(skippy_api_dir() / doc), doc);
        tags.insert(tags.end(), more.begin(), more.end());
    }
    std::vector<std::string> haystacks;
    for (const char* dir : {"include", "src"}) {
        for (const auto& entry : fs::recursive_directory_iterator(skippy_api_dir() / dir)) {
            if (entry.is_regular_file()) haystacks.push_back(read_file(entry.path()));
        }
    }
    for (const auto& tag : tags) {
        if (tag.kind != "anti-pattern") continue;
        const auto leaf = tag.symbol.substr(tag.symbol.rfind("::") + 2);
        for (const auto& haystack : haystacks) {
            if (haystack.find(leaf) != std::string::npos) {
                ADD_FAILURE() << "anti-pattern symbol " << tag.symbol << " (" << tag.file << ":"
                              << tag.line << ") still exists in include/ or src/";
                break;
            }
        }
    }
}

} // namespace
