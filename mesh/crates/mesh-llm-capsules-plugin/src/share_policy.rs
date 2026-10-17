//! This plugin's `config_schema` declaration for
//! the one sharing-policy object -- six switches, one key
//! (`docs/SHARING-POLICY.md`). Declaring it here
//! makes mesh's console render the six switches under Configuration >
//! Plugins with the host's own controls -- nothing here is a score, and
//! nothing here ranks anyone.
//!
//! **How a console value reaches the plugin.** mesh-llm saves what the
//! operator sets in the console to its config file (this plugin's
//! `[plugin.settings]`) and does not pass it to the plugin process (no env,
//! no protocol message). `crate::settings::var` reads it there when the
//! environment does not set the setting. Each key is its env var's suffix
//! lower-cased (`share_record_at_completion` ->
//! `CAPSULES_SHARE_RECORD_AT_COMPLETION`), except `witness`, which
//! is `CAPSULES_CHECKPOINT_WITNESS_URLS`. The witness is read when
//! the plugin starts: restart mesh-llm after changing it.
//!
//! The same schema also declares the opt-in stop-routing rule's two settings
//! (`routing_rule`), under their own "Routing rule" category, on the same
//! env-var convention: `CAPSULES_STOP_ROUTING_AFTER_CONTRADICTIONS`
//! (off unless set) and `CAPSULES_STOP_ROUTING_WINDOW_DAYS`.

use mesh_llm_plugin::{
    config_array, config_enum, config_integer, config_object, config_object_property,
    config_schema, config_setting, config_string, config_url, ManifestEntry,
};

pub const RECORD_AT_COMPLETION_KEY: &str = "share_record_at_completion";
pub const HISTORY_SEGMENTS_KEY: &str = "share_history_segments";
pub const ADJUDICATIONS_KEY: &str = "share_adjudications";
pub const WITNESS_KEY: &str = "witness";
/// Standard Service History (checkpoints, `history_card/1`,
/// `served_summary/1`) to a requester that is not this node's counterparty.
pub const STANDARD_HISTORY_KEY: &str = "share_standard_history";
/// `exchange_corroboration/1` (one exchange's half digests) to a requester
/// that no record of this node names as its counterparty.
pub const EXCHANGE_CORROBORATION_KEY: &str = "share_exchange_corroboration";
/// Whether this node advertises that it answers evidence requests at all.
pub const ADVERTISE_EVIDENCE_REQUESTS_KEY: &str = "advertise_evidence_requests";
/// The opt-in stop-routing rule (`routing_rule`): N, and D in days. Same
/// env-suffix convention (`CAPSULES_STOP_ROUTING_*`).
pub const STOP_ROUTING_AFTER_KEY: &str = "stop_routing_after_contradictions";
pub const STOP_ROUTING_WINDOW_KEY: &str = "stop_routing_window_days";
/// The referee (`crate::referee`): whether a differing twin pair asks one
/// (off unless set to on), and the bar window after a contradiction, in days.
/// Same env-suffix convention (`CAPSULES_ADJUDICATE_DIFFERING_TWINS`,
/// `CAPSULES_REFEREE_BAR_DAYS`).
pub const ADJUDICATE_DIFFERING_TWINS_KEY: &str = "adjudicate_differing_twins";
pub const REFEREE_BAR_DAYS_KEY: &str = "referee_bar_days";

/// This process's own runtime env var for `share_record_at_completion` --
/// see the module doc's "declarative only" note: no live host->plugin
/// config channel exists yet, so an operator sets this directly on the
/// node.
pub const ENV_RECORD_AT_COMPLETION: &str = "CAPSULES_SHARE_RECORD_AT_COMPLETION";

/// This process's runtime `share_record_at_completion` value: off unless
/// the operator set it to `"counterparty"`. Unset, `off` or any other value
/// is the default, off: a node pushes nothing to a peer unless its operator
/// turned this on.
pub fn record_at_completion_is_off() -> bool {
    record_at_completion_is_off_for(
        crate::settings::var(ENV_RECORD_AT_COMPLETION)
            .ok()
            .as_deref(),
    )
}

fn record_at_completion_is_off_for(raw: Option<&str>) -> bool {
    normalized(raw).as_deref() != Some("counterparty")
}

/// A switch's value as the operator meant it: trimmed and lower-cased;
/// empty is no value.
pub(crate) fn normalized(raw: Option<&str>) -> Option<String> {
    raw.map(|r| r.trim().to_ascii_lowercase())
        .filter(|r| !r.is_empty())
}

/// This process's own env var for `share_history_segments`.
pub const ENV_HISTORY_SEGMENTS: &str = "CAPSULES_SHARE_HISTORY_SEGMENTS";

/// `prospective`, deprecated: it never served anyone `counterparties` does
/// not, so it is read as `counterparties` and a log line says so. The one
/// place the old value is resolved.
const DEPRECATED_PROSPECTIVE: &str = "prospective";

/// Who may read one of this node's records back: `off`, `counterparties` or
/// `peers`. Unset is the documented default, `counterparties`; `prospective`
/// is read as `counterparties`; an unknown value is `off`, never wider.
pub fn history_segments() -> &'static str {
    history_segments_for(crate::settings::var(ENV_HISTORY_SEGMENTS).ok().as_deref())
}

pub(crate) fn history_segments_for(raw: Option<&str>) -> &'static str {
    match normalized(raw).as_deref() {
        None | Some("counterparties") => "counterparties",
        Some(DEPRECATED_PROSPECTIVE) => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    setting = ENV_HISTORY_SEGMENTS,
                    "prospective is deprecated and read as counterparties, which it always behaved as; set counterparties"
                );
            });
            "counterparties"
        }
        Some("peers") => "peers",
        // "off", and anything this plugin does not know: never wider than
        // the operator could have meant (`setting_problems` says so).
        Some(_) => "off",
    }
}

/// This process's own env var for `share_standard_history`.
pub const ENV_STANDARD_HISTORY: &str = "CAPSULES_SHARE_STANDARD_HISTORY";

/// Whether Standard Service History goes to a requester that is not this
/// node's counterparty: only when the operator turned it on. Off by default;
/// a counterparty is answered either way.
pub fn shares_standard_history() -> bool {
    shares_standard_history_for(crate::settings::var(ENV_STANDARD_HISTORY).ok().as_deref())
}

pub(crate) fn shares_standard_history_for(raw: Option<&str>) -> bool {
    switched_on(raw)
}

/// This process's own env var for `advertise_evidence_requests`: off unless
/// switched on (`on`, `1`, `true` or `yes`). This version has nothing that
/// advertises evidence requests, so nothing reads it yet: the plugin's
/// `evidence-request/1` channel is declared only to its own host, so that a
/// request can reach it, and mesh-llm does not pass a plugin's channels on to
/// other nodes.
pub const ENV_ADVERTISE_EVIDENCE_REQUESTS: &str = "CAPSULES_ADVERTISE_EVIDENCE_REQUESTS";

/// This process's own env var for `share_exchange_corroboration`.
pub const ENV_EXCHANGE_CORROBORATION: &str = "CAPSULES_SHARE_EXCHANGE_CORROBORATION";

/// Whether `exchange_corroboration/1` goes to a requester that is not this
/// node's counterparty: only when the operator turned it on. Off by default.
/// A setting of its own: `share_standard_history` never releases it.
pub fn shares_exchange_corroboration() -> bool {
    switched_on(
        crate::settings::var(ENV_EXCHANGE_CORROBORATION)
            .ok()
            .as_deref(),
    )
}

/// An on/off switch: on only for `on` (or `1`, `true`, `yes`); unset and
/// anything else is off.
fn switched_on(raw: Option<&str>) -> bool {
    matches!(
        normalized(raw).as_deref(),
        Some("on" | "1" | "true" | "yes")
    )
}

/// The sharing switches set to a value this plugin does not know, each read
/// as off: for the page, so a typo is seen rather than silently obeyed.
pub fn setting_problems() -> Vec<String> {
    const ON_OFF: &[&str] = &["on", "off", "1", "0", "true", "false", "yes", "no"];
    let known: [(&str, &[&str]); 8] = [
        (ENV_RECORD_AT_COMPLETION, &["counterparty", "off"]),
        (
            ENV_HISTORY_SEGMENTS,
            &["counterparties", DEPRECATED_PROSPECTIVE, "peers", "off"],
        ),
        (ENV_STANDARD_HISTORY, ON_OFF),
        (ENV_EXCHANGE_CORROBORATION, ON_OFF),
        (ENV_ADVERTISE_EVIDENCE_REQUESTS, ON_OFF),
        (ENV_ADJUDICATIONS, &["deliver_to_subjects", "off"]),
        (
            crate::referee::request::ENV_ADJUDICATE_DIFFERING_TWINS,
            ON_OFF,
        ),
        (crate::checkpoint_cadence::ENV_ENABLE, ON_OFF),
    ];
    known
        .iter()
        .filter_map(|(env, values)| {
            let value = normalized(crate::settings::var(env).ok().as_deref())?;
            (!values.contains(&value.as_str())).then(|| {
                format!(
                    "{env} is {value:?}, which is not one of {}; it is read as off",
                    values.join(", ")
                )
            })
        })
        .collect()
}

/// This process's own env var for `share_adjudications`.
pub const ENV_ADJUDICATIONS: &str = "CAPSULES_SHARE_ADJUDICATIONS";

/// Deliver a verdict to the nodes it judges: only when the operator set
/// `share_adjudications` to `deliver_to_subjects`. The default is off.
pub fn adjudications_delivered() -> bool {
    adjudications_delivered_for(crate::settings::var(ENV_ADJUDICATIONS).ok().as_deref())
}

fn adjudications_delivered_for(raw: Option<&str>) -> bool {
    normalized(raw).as_deref() == Some("deliver_to_subjects")
}

const CATEGORY_ID: &str = "share";
const CATEGORY_LABEL: &str = "Sharing policy";
const CATEGORY_SUMMARY: &str = "What this node shares, with whom, by default. Every default keys \
    on relationship (counterparty in the window), never proximity or latency.";

const REFEREE_CATEGORY_ID: &str = "referee";
const REFEREE_CATEGORY_LABEL: &str = "Referee";
const REFEREE_CATEGORY_SUMMARY: &str =
    "An independent check of two twins that answered the same request differently. \
    Eligibility is yes or no, from this node's own records; no figure about a peer is computed. Off by default: \
    asking a referee sends the twins' request, including the prompt, and both twins' answers to the referee peer.";

const RULE_CATEGORY_ID: &str = "routing_rule";
const RULE_CATEGORY_LABEL: &str = "Routing rule";
const RULE_CATEGORY_SUMMARY: &str =
    "An optional rule of yours for when this node stops routing to a \
    peer. Off by default.";

/// The `config_schema` manifest entry naming all six sharing switches and the other settings. Attach with
/// `DeclarativePluginBuilder::config_item`.
///
/// Every choice list names its default FIRST. mesh-llm 0.78's console does not
/// receive a plugin setting's declared default, so it takes the first choice
/// as the default: it shows that choice for an unset setting and leaves it out
/// of the saved config. With the default first, an unset setting shows its
/// real value and every other choice is saved. Pinned by
/// `every_choice_list_names_its_default_first`.
pub fn share_policy_config_schema(plugin_id: &str) -> ManifestEntry {
    config_schema(plugin_id)
        .setting(
            config_setting(RECORD_AT_COMPLETION_KEY, config_enum(["off", "counterparty"]))
                .default_value(&"off")
                .description(
                    "Push this node's own sealed record of a completed exchange to the \
                     counterparty (counterparty), or don't (off, the default). With checkpointing on, the \
                     push also carries a signed checkpoint of this node's log and the record's \
                     inclusion proof -- the checkpoint reveals the log's total size (records \
                     across all peers) and its time, never any other record's content. \
                     Symmetric: a node with this off also does not receive the other side's \
                     push -- fetch-on-request still works either way.",
                )
                .label("Record at completion")
                .category(CATEGORY_ID, CATEGORY_LABEL, CATEGORY_SUMMARY, 0),
        )
        .setting(
            config_setting(
                HISTORY_SEGMENTS_KEY,
                config_enum(["counterparties", DEPRECATED_PROSPECTIVE, "peers", "off"]),
            )
            .default_value(&"counterparties")
            .description(
                "Who this node answers a chain-segment/record/correlation request from: past \
                 counterparties only (the default), any peer, or nobody. prospective is \
                 deprecated and read as counterparties, which it always behaved as. A refusal \
                 is structural (not_authorized): it says nothing about the peer.",
            )
            .label("History segments")
            .category(CATEGORY_ID, CATEGORY_LABEL, CATEGORY_SUMMARY, 1),
        )
        .setting(
            config_setting(ADJUDICATIONS_KEY, config_enum(["off", "deliver_to_subjects"]))
                .default_value(&"off")
                .description(
                    "Deliver a sealed verdict to every node it judges, or don't (off, the default). The \
                     subject acks or disputes on its own chain -- this is delivery, never a \
                     computed standing.",
                )
                .label("Adjudications")
                .category(CATEGORY_ID, CATEGORY_LABEL, CATEGORY_SUMMARY, 2),
        )
        .setting(
            config_setting(
                WITNESS_KEY,
                config_array(config_object([
                    config_object_property("endpoint", config_url())
                        .required(true)
                        .description("The witness service's URL.")
                        .into(),
                    config_object_property("public_key", config_string())
                        .description(
                            "The witness's Ed25519 public key in hex, as its operator publishes it. Leave empty to have the plugin fetch it from the witness once and keep it; the Evidence page then says the key was pinned on first contact, not configured.",
                        )
                        .into(),
                ])),
            )
            .description(
                "Witness services to offer this node's checkpoints to: each one's URL, and its public key if you have it. Empty is off (the default): there is no default witness, and the plugin contacts none until you add one. The Evidence page shows what each witness holds. Takes effect when mesh-llm is restarted.",
            )
                .label("Witnesses")
                .category(CATEGORY_ID, CATEGORY_LABEL, CATEGORY_SUMMARY, 3),
        )
        .setting(
            config_setting(STANDARD_HISTORY_KEY, config_enum(["off", "on"]))
                .default_value(&"off")
                .description(
                    "Answer a node this node has never exchanged with when it asks for this \
                     node's checkpoints, history card or served summary (on), or decline (off, \
                     the default). These carry no records, but they show how much and how \
                     often this node serves. Off, such a node gets a signed policy_declined \
                     that says nothing about who asked; a counterparty is answered either way. \
                     A counterparty is known by the id its request states, which is not yet \
                     authenticated.",
                )
                .label("Standard history")
                .category(CATEGORY_ID, CATEGORY_LABEL, CATEGORY_SUMMARY, 4),
        )
        .setting(
            config_setting(ADVERTISE_EVIDENCE_REQUESTS_KEY, config_enum(["off", "on"]))
                .default_value(&"off")
                .description(
                    "Tell other nodes that this node answers evidence requests (on), or \
                     don't (off, the default). Off, a node that asks is still answered under \
                     the settings above. This version advertises nothing, so nothing reads this \
                     yet.",
                )
                .label("Advertise evidence requests")
                .category(CATEGORY_ID, CATEGORY_LABEL, CATEGORY_SUMMARY, 5),
        )
        .setting(
            // Wording is a draft awaiting the operator's text (Q7 ruling §5).
            config_setting(EXCHANGE_CORROBORATION_KEY, config_enum(["off", "on"]))
                .default_value(&"off")
                .description(
                    "Answer a node this node has never exchanged with when it asks whether this \
                     node holds a half of one exchange, named by its client nonce (on), or \
                     decline (off, the default). The answer carries the half's id, role and two \
                     digests, never the record or who was on the other side. Off, such a node \
                     gets a signed policy_declined that says nothing about who asked.",
                )
                .label("Exchange corroboration")
                .category(CATEGORY_ID, CATEGORY_LABEL, CATEGORY_SUMMARY, 6),
        )
        .setting(
            config_setting(ADJUDICATE_DIFFERING_TWINS_KEY, config_enum(["off", "on"]))
                .default_value(&"off")
                .description(
                    "Ask a referee when two twins of a pair a client marked answered the same \
                     request at temperature 0, on the same model and weights, and differ (on), or \
                     never (off, the default). Asking sends the twins' request, including the \
                     prompt, and both twins' answers to the referee peer, which answers the \
                     request with its own inference; so only you turn this on. At most one call per pair, never retried. A pair \
                     with no ruling reads \"Not adjudicated\" with the reason, and never counts \
                     against either twin.",
                )
                .label("Ask a referee when twins differ")
                .category(REFEREE_CATEGORY_ID, REFEREE_CATEGORY_LABEL, REFEREE_CATEGORY_SUMMARY, 0),
        )
        .setting(
            config_setting(REFEREE_BAR_DAYS_KEY, config_integer())
                .default_value(&30)
                .description(
                    "After a referee-signed contradiction for a model, a node is not asked to \
                     referee that model for this many days (default 30), counted from when this \
                     node recorded the verdict. A later corroboration does not end it early.",
                )
                .label("Bar window, days")
                .category(REFEREE_CATEGORY_ID, REFEREE_CATEGORY_LABEL, REFEREE_CATEGORY_SUMMARY, 1),
        )
        .setting(
            config_setting(STOP_ROUTING_AFTER_KEY, config_integer())
                .description(
                    "Stop routing to a peer after this many contradictions within the window below. \
                     Off when empty or 0 (the default). Only verdicts this node asked a referee \
                     for, and whose signature it checked, count, once per referee and pair of \
                     answers; with N of 2 or more, no single referee can reach N alone. When it \
                     fires, the plugin asks the host to stop routing to that peer until you undo \
                     it. The host does that only if you also set allow_peer_blocks = true for \
                     this plugin in mesh-llm's config; otherwise it refuses and nothing is \
                     blocked. The host records the block as this plugin's, and the plugin seals a \
                     record that names this rule and cites the verdicts. Undo it like any Stop \
                     routing. No figure about a peer is computed.",
                )
                .label("Stop routing after N contradictions")
                .category(RULE_CATEGORY_ID, RULE_CATEGORY_LABEL, RULE_CATEGORY_SUMMARY, 0),
        )
        .setting(
            config_setting(STOP_ROUTING_WINDOW_KEY, config_integer())
                .default_value(&30)
                .description("The window, in days, the contradictions must fall in (default 30).")
                .label("Within D days")
                .category(RULE_CATEGORY_ID, RULE_CATEGORY_LABEL, RULE_CATEGORY_SUMMARY, 1),
        )
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_plugin::proto;

    fn as_config_schema(entry: ManifestEntry) -> proto::PluginConfigSchemaManifest {
        match entry {
            ManifestEntry::ConfigSchema(schema) => schema,
            other => panic!("expected ManifestEntry::ConfigSchema, got {other:?}"),
        }
    }

    #[test]
    fn declares_all_four_switches_under_the_plugin_id() {
        let schema = as_config_schema(share_policy_config_schema("capsules"));
        assert_eq!(schema.plugin_name, "capsules");
        let keys: Vec<&str> = schema.settings.iter().map(|s| s.key.as_str()).collect();
        assert_eq!(
            keys,
            vec![
                RECORD_AT_COMPLETION_KEY,
                HISTORY_SEGMENTS_KEY,
                ADJUDICATIONS_KEY,
                WITNESS_KEY,
                STANDARD_HISTORY_KEY,
                ADVERTISE_EVIDENCE_REQUESTS_KEY,
                EXCHANGE_CORROBORATION_KEY,
                ADJUDICATE_DIFFERING_TWINS_KEY,
                REFEREE_BAR_DAYS_KEY,
                STOP_ROUTING_AFTER_KEY,
                STOP_ROUTING_WINDOW_KEY
            ]
        );
    }

    /// mesh-llm 0.78's console takes a choice list's first value as the
    /// setting's default (it does not receive the declared one). Listing the
    /// declared default first keeps what the console shows and saves equal to
    /// what this plugin does.
    #[test]
    fn every_choice_list_names_its_default_first() {
        let schema = as_config_schema(share_policy_config_schema("capsules"));
        let mut checked = 0;
        for setting in &schema.settings {
            let Some(values) = setting
                .value_schema
                .as_ref()
                .map(|v| &v.enum_values)
                .filter(|v| !v.is_empty())
            else {
                continue;
            };
            let default: String =
                serde_json::from_str(setting.default_json.as_deref().unwrap_or_else(|| {
                    panic!("{} is a choice with no declared default", setting.key)
                }))
                .expect("default_json is a JSON string");
            assert_eq!(
                values[0], default,
                "{}: the first choice must be the default",
                setting.key
            );
            checked += 1;
        }
        assert_eq!(checked, 7, "the seven choice settings were checked");
    }

    #[test]
    fn the_referee_settings_are_the_env_names_the_referee_reads() {
        assert_eq!(
            format!("CAPSULES_{}", ADJUDICATE_DIFFERING_TWINS_KEY.to_uppercase()),
            crate::referee::request::ENV_ADJUDICATE_DIFFERING_TWINS
        );
        assert_eq!(
            format!("CAPSULES_{}", REFEREE_BAR_DAYS_KEY.to_uppercase()),
            crate::referee::bar::ENV_REFEREE_BAR_DAYS
        );
    }

    #[test]
    fn history_segments_defaults_to_counterparties_reads_the_three_tiers_and_unknown_is_off() {
        assert_eq!(history_segments_for(None), "counterparties");
        assert_eq!(
            history_segments_for(Some("bogus")),
            "off",
            "an unknown value never widens"
        );
        for tier in ["off", "counterparties", "peers"] {
            assert_eq!(history_segments_for(Some(tier)), tier);
        }
    }

    /// `prospective` (R-10): still accepted, so a saved setting keeps working,
    /// and read as exactly `counterparties`.
    #[test]
    fn prospective_is_read_as_counterparties() {
        assert_eq!(history_segments_for(Some("prospective")), "counterparties");
        assert_eq!(
            history_segments_for(Some(" Prospective ")),
            "counterparties"
        );
    }

    #[test]
    fn setting_keys_match_the_python_env_var_suffix_convention() {
        // Each env var is the setting key with the shared
        // CAPSULES_ prefix -- this module's whole "no
        // re-naming exercise later" claim rests on this correspondence.
        let expected_env_suffix = [
            (RECORD_AT_COMPLETION_KEY, "SHARE_RECORD_AT_COMPLETION"),
            (HISTORY_SEGMENTS_KEY, "SHARE_HISTORY_SEGMENTS"),
            (ADJUDICATIONS_KEY, "SHARE_ADJUDICATIONS"),
            (WITNESS_KEY, "CHECKPOINT_WITNESS_URLS"),
            (STANDARD_HISTORY_KEY, "SHARE_STANDARD_HISTORY"),
            (EXCHANGE_CORROBORATION_KEY, "SHARE_EXCHANGE_CORROBORATION"),
            (
                ADVERTISE_EVIDENCE_REQUESTS_KEY,
                "ADVERTISE_EVIDENCE_REQUESTS",
            ),
            (ADJUDICATE_DIFFERING_TWINS_KEY, "ADJUDICATE_DIFFERING_TWINS"),
            (REFEREE_BAR_DAYS_KEY, "REFEREE_BAR_DAYS"),
            (STOP_ROUTING_AFTER_KEY, "STOP_ROUTING_AFTER_CONTRADICTIONS"),
            (STOP_ROUTING_WINDOW_KEY, "STOP_ROUTING_WINDOW_DAYS"),
        ];
        for (key, suffix) in expected_env_suffix {
            assert_eq!(
                crate::settings::console_key(&format!("CAPSULES_{suffix}")).as_deref(),
                Some(key),
                "env suffix {suffix} must map to the console key {key}"
            );
        }
    }

    #[test]
    fn nothing_that_reaches_a_peer_on_its_own_is_on_by_default() {
        let schema = as_config_schema(share_policy_config_schema("capsules"));
        let by_key = |k: &str| schema.settings.iter().find(|s| s.key == k).unwrap();
        assert_eq!(
            by_key(RECORD_AT_COMPLETION_KEY).default_json.as_deref(),
            Some("\"off\"")
        );
        assert_eq!(
            by_key(HISTORY_SEGMENTS_KEY).default_json.as_deref(),
            Some("\"counterparties\"")
        );
        assert_eq!(
            by_key(STANDARD_HISTORY_KEY).default_json.as_deref(),
            Some("\"off\"")
        );
        assert_eq!(
            by_key(EXCHANGE_CORROBORATION_KEY).default_json.as_deref(),
            Some("\"off\"")
        );
        assert_eq!(
            by_key(ADVERTISE_EVIDENCE_REQUESTS_KEY)
                .default_json
                .as_deref(),
            Some("\"off\"")
        );
        assert_eq!(
            by_key(ADJUDICATIONS_KEY).default_json.as_deref(),
            Some("\"off\"")
        );
        assert_eq!(
            by_key(ADJUDICATE_DIFFERING_TWINS_KEY)
                .default_json
                .as_deref(),
            Some("\"off\"")
        );
        // Answering a request for a record is unchanged: `counterparties`
        // serves exactly whom `prospective` did.
        assert_eq!(history_segments_for(None), "counterparties");
        assert!(
            !switched_on(None),
            "standard history and advertising are off"
        );
        assert!(!adjudications_delivered_for(None));
        assert!(!crate::referee::request::adjudicate_differing_twins_from(
            None
        ));
    }

    #[test]
    fn the_stop_routing_rule_is_off_by_default() {
        let schema = as_config_schema(share_policy_config_schema("capsules"));
        let after = schema
            .settings
            .iter()
            .find(|s| s.key == STOP_ROUTING_AFTER_KEY)
            .unwrap();
        assert_eq!(after.default_json, None, "no N by default: the rule is off");
        assert_eq!(
            format!("CAPSULES_{}", STOP_ROUTING_AFTER_KEY.to_uppercase()),
            crate::routing_rule::ENV_AFTER
        );
        assert_eq!(
            format!("CAPSULES_{}", STOP_ROUTING_WINDOW_KEY.to_uppercase()),
            crate::routing_rule::ENV_WINDOW_DAYS
        );
    }

    #[test]
    fn witness_has_no_default_url_and_is_not_required() {
        // Design note S1: "witness: off | <url> # default: off" -- off IS
        // the absence of a default, never a magic sentinel string.
        let schema = as_config_schema(share_policy_config_schema("capsules"));
        let witness = schema
            .settings
            .iter()
            .find(|s| s.key == WITNESS_KEY)
            .unwrap();
        assert_eq!(witness.default_json, None);
        assert!(!witness.required);
        // A list of witnesses, each an endpoint and, optionally, its key.
        let value = witness.value_schema.as_ref().unwrap();
        assert_eq!(value.kind, proto::PluginConfigValueKind::Array as i32);
        let row = value.items.as_ref().unwrap();
        assert_eq!(row.kind, proto::PluginConfigValueKind::Object as i32);
        let props: Vec<_> = row
            .object_properties
            .iter()
            .map(|p| (p.key.as_str(), p.required))
            .collect();
        assert_eq!(props, [("endpoint", true), ("public_key", false)]);
    }

    #[test]
    fn every_setting_is_optional_shipping_this_schema_flips_no_runtime_behavior() {
        let schema = as_config_schema(share_policy_config_schema("capsules"));
        assert!(schema.settings.iter().all(|s| !s.required));
    }

    #[test]
    fn record_at_completion_is_on_only_for_counterparty() {
        assert!(!record_at_completion_is_off_for(Some("counterparty")));
        assert!(!record_at_completion_is_off_for(Some(" counterparty ")));
        assert!(record_at_completion_is_off_for(None));
        assert!(record_at_completion_is_off_for(Some("off")));
        assert!(record_at_completion_is_off_for(Some("garbage")));
    }

    /// A typo, odd case, stray spaces or an empty value never widen who may
    /// read records back.
    #[test]
    fn history_segments_never_widen_on_a_typo() {
        assert_eq!(history_segments_for(None), "counterparties");
        assert_eq!(
            history_segments_for(Some("")),
            "counterparties",
            "empty is unset"
        );
        assert_eq!(history_segments_for(Some("Off ")), "off");
        assert_eq!(
            history_segments_for(Some(" COUNTERPARTIES")),
            "counterparties"
        );
        assert_eq!(
            history_segments_for(Some("counterparty")),
            "off",
            "unknown is off"
        );
        assert_eq!(history_segments_for(Some("peeers")), "off");
        assert!(!record_at_completion_is_off_for(Some(" Counterparty")));
    }

    #[test]
    fn an_on_off_switch_is_on_only_when_switched_on() {
        for on in ["on", " ON ", "1", "true", "yes"] {
            assert!(switched_on(Some(on)), "{on:?}");
        }
        for off in [
            None,
            Some(""),
            Some("off"),
            Some("0"),
            Some("no"),
            Some("onn"),
        ] {
            assert!(!switched_on(off), "{off:?}");
        }
    }

    #[test]
    fn verdict_delivery_is_on_only_for_deliver_to_subjects() {
        assert!(adjudications_delivered_for(Some("deliver_to_subjects")));
        assert!(!adjudications_delivered_for(None));
        assert!(!adjudications_delivered_for(Some("off")));
        assert!(!adjudications_delivered_for(Some("garbage")));
    }
}
