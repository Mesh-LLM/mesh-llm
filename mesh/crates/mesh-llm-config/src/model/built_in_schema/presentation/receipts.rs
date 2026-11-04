//! How the `[receipts]` settings are presented.

use super::{CategoryPresentation, SettingPresentation, sp};

pub(super) const RECEIPTS_CATEGORY: CategoryPresentation = CategoryPresentation {
    id: "receipts",
    label: "Receipts",
    summary: "Verifiable inference receipts for the models this node serves",
    order: 65,
};

pub(super) fn receipts_presentation(rendered: &str) -> Option<SettingPresentation> {
    match rendered {
        "receipts.enabled" => Some(sp(
            "Issue receipts",
            "Sign a receipt with the node key for every turn a client asks one for, so the client can audit that the claimed model served it.",
            RECEIPTS_CATEGORY,
            10,
        )),
        "receipts.audit" => Some(sp(
            "Audit peers' receipts",
            "Audit about one in ten of the turns peers serve with receipts, reading the claimed model from Hugging Face, and block a peer whose audit fails for seven days.",
            RECEIPTS_CATEGORY,
            20,
        )),
        "receipts.require" => Some(sp(
            "Require receipts",
            "Route this node's chat completions only to models served with receipts, and audit the turns peers serve. Requests for other endpoints fail.",
            RECEIPTS_CATEGORY,
            30,
        )),
        "receipts.offline" => Some(sp(
            "Never audit receipts",
            "Never audit a receipt, and so never contact Hugging Face for one, whatever the other settings or a request ask. A request that requires receipts is then served only by this node's own runtime.",
            RECEIPTS_CATEGORY,
            40,
        )),
        _ => None,
    }
}
