#[derive(Clone, Copy)]
pub enum Case {
    Lf,
    UnicodeCrlf,
    ChangedHunk,
    Empty,
    Malformed,
}

pub const SUBJECT: &str = "skippy: generate model-family stage controls";
pub const LF_PATCH: &[u8] = include_bytes!("../fixtures/migration/rewriter-patch/two-model.patch");

impl Case {
    pub fn diff(self) -> &'static [u8] {
        match self {
            Self::Lf => include_bytes!("../fixtures/migration/rewriter-patch/two-model.diff"),
            Self::UnicodeCrlf => b"diff --git a/src/models/cafe.cpp b/src/models/cafe.cpp\n--- a/src/models/cafe.cpp\n+++ b/src/models/cafe.cpp\n@@ -1 +1 @@\n-caf\xc3\xa9\r\n+caf\xc3\xa8\r\n",
            Self::ChangedHunk => b"diff --git a/src/models/cafe.cpp b/src/models/cafe.cpp\n--- a/src/models/cafe.cpp\n+++ b/src/models/cafe.cpp\n@@ -1 +1 @@\n-caf\xc3\xa9\r\n+caf\xc3\xa9\r\n",
            Self::Empty => b"",
            Self::Malformed => b"\xff",
        }
    }

    pub fn digest(self) -> Option<&'static str> {
        match self {
            Self::Lf => Some("c1bf5a727845c2ab193c43dc8884812906fcc5f4b30a9a8e55a4358ff6a5c9ce"),
            Self::UnicodeCrlf => {
                Some("5e71a74782aab731bcd6afd25dab0249829c17695539bf268c442ac525d4f1bc")
            }
            Self::ChangedHunk => {
                Some("220c3cc5e136e4a27fc45ec0f14f71d9780460c20831ada226f01faa9836431f")
            }
            Self::Empty | Self::Malformed => None,
        }
    }
}
