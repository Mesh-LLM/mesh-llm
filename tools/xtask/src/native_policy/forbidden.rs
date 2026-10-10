//! `FORBIDDEN_IMPORTS`: the backend libraries the neutral host must never
//! import, each matched as its case-insensitive `re.search` pattern would.
//! Every pattern but `Metal\.framework` is anchored at the start of the
//! name or after a `/` or `\` separator.

use super::imports::{fold_eq, starts_with_folded};

/// How the anchored remainder of one pattern must continue.
enum Tail {
    /// `literal` followed by anything.
    Prefix(&'static str),
    /// `literal` then one of `suffixes`, the empty suffix meaning `$`.
    Ending(&'static str, &'static [&'static str]),
    /// `literal[^/\\]+\.dll$`.
    Versioned(&'static str),
}

const ANCHORED: &[Tail] = &[
    Tail::Ending("libcuda", &[".so", ".dylib", ".dll", ""]),
    Tail::Prefix("libcudart"),
    Tail::Prefix("libcublas"),
    Tail::Prefix("libnccl"),
    Tail::Prefix("libamdhip64"),
    Tail::Prefix("libhip"),
    Tail::Prefix("libhsa-runtime"),
    Tail::Ending("nvcuda.dll", &[""]),
    Tail::Versioned("cudart64_"),
    Tail::Versioned("cublaslt64_"),
    Tail::Versioned("cublas64_"),
    Tail::Ending("amdhip64.dll", &[""]),
    Tail::Ending("hipblas.dll", &[""]),
    Tail::Ending("rocblas.dll", &[""]),
    Tail::Prefix("libvulkan"),
    Tail::Ending("vulkan-1.dll", &[""]),
    Tail::Prefix("ggml"),
    Tail::Prefix("libggml"),
    Tail::Prefix("llama"),
    Tail::Prefix("libllama"),
];

/// `forbidden_imports(imports)`, in input order.
pub(super) fn forbidden(imports: &[String]) -> Vec<String> {
    imports
        .iter()
        .filter(|dependency| is_forbidden(dependency))
        .cloned()
        .collect()
}

fn is_forbidden(dependency: &str) -> bool {
    let chars: Vec<char> = dependency.chars().collect();
    if (0..chars.len()).any(|at| starts_with_folded(&chars, at, "metal.framework")) {
        return true;
    }
    anchors(&chars).any(|at| ANCHORED.iter().any(|tail| tail_matches(&chars, at, tail)))
}

/// `(^|[/\\])`: the start, and every position after a separator.
fn anchors(chars: &[char]) -> impl Iterator<Item = usize> + '_ {
    std::iter::once(0).chain(
        chars
            .iter()
            .enumerate()
            .filter(|(_, ch)| matches!(ch, '/' | '\\'))
            .map(|(index, _)| index + 1),
    )
}

/// `$` without `re.MULTILINE`: the end, or just before a final newline.
fn at_end(chars: &[char], at: usize) -> bool {
    at == chars.len() || (at + 1 == chars.len() && chars[at] == '\n')
}

fn tail_matches(chars: &[char], at: usize, tail: &Tail) -> bool {
    match tail {
        Tail::Prefix(literal) => starts_with_folded(chars, at, literal),
        Tail::Ending(literal, suffixes) => {
            let after = at + literal.chars().count();
            starts_with_folded(chars, at, literal)
                && suffixes.iter().any(|suffix| {
                    if suffix.is_empty() {
                        at_end(chars, after)
                    } else {
                        starts_with_folded(chars, after, suffix)
                    }
                })
        }
        Tail::Versioned(literal) => {
            let after = at + literal.chars().count();
            starts_with_folded(chars, at, literal) && versioned_dll(chars, after)
        }
    }
}

/// `[^/\\]+\.dll$` at `start`.
fn versioned_dll(chars: &[char], start: usize) -> bool {
    let ends = [Some(chars.len()), chars.len().checked_sub(1)];
    ends.into_iter().flatten().any(|end| {
        at_end(chars, end)
            && end >= start + 5
            && chars[start..end - 4]
                .iter()
                .all(|ch| !matches!(ch, '/' | '\\'))
            && ".dll"
                .chars()
                .zip(&chars[end - 4..end])
                .all(|(pattern, ch)| fold_eq(pattern, *ch))
    })
}

#[cfg(test)]
mod tests {
    use super::forbidden;

    #[test]
    fn migration_native_policy_forbidden_patterns_match_python_re() {
        let imports: Vec<String> = [
            "/usr/lib/libcuda.so.1",
            "libcuda",
            "libcudax.so",
            "C:\\Windows\\NVCUDA.DLL",
            "cudart64_12.dll",
            "cudart64_.dll",
            "cudart64_1/2.dll",
            "cublasLt64_12.dll",
            "@rpath/libggml-base.dylib",
            "/System/Library/Frameworks/Metal.framework/Metal",
            "xllama.so",
            "libvulkan.so.1",
            "vulkan-1.dll\n",
            "libSystem.B.dylib",
        ]
        .map(str::to_owned)
        .to_vec();
        assert_eq!(
            forbidden(&imports),
            [
                "/usr/lib/libcuda.so.1",
                "libcuda",
                "C:\\Windows\\NVCUDA.DLL",
                "cudart64_12.dll",
                "cublasLt64_12.dll",
                "@rpath/libggml-base.dylib",
                "/System/Library/Frameworks/Metal.framework/Metal",
                "libvulkan.so.1",
                "vulkan-1.dll\n",
            ]
        );
    }
}
