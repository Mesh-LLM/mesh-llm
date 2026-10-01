mod render;

use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

#[derive(Deserialize)]
struct Inventory {
    schema_version: u32,
    spec_sha256: String,
    reason_codes: Vec<String>,
    projected_envelope_keys: Vec<String>,
    base_projected_event_keys: Vec<String>,
    projected_event_keys: Vec<String>,
    forbidden_projection_keys: Vec<String>,
    amendment_procedure: String,
    families: Vec<Family>,
    catalog: Catalog,
    projection_profiles: Vec<Profile>,
    event_projection_overrides: Vec<Override>,
}

#[derive(Deserialize)]
struct Family {
    section: String,
    enum_name: String,
    owning_crates: Vec<String>,
    producer_symbols: Vec<String>,
    source_paths: Vec<String>,
    privacy_class: String,
    default_delivery_class: String,
    default_projection_profile: Option<String>,
    authoritative_outcome: String,
    reducer_effect: String,
    projection: String,
    test: String,
    migration_task: u32,
}

#[derive(Deserialize)]
struct Catalog {
    terminal_event_ids: Vec<String>,
    progress_event_ids: Vec<String>,
    spec_bullets: Vec<Bullet>,
}

#[derive(Deserialize)]
struct Bullet {
    section: String,
    event_ids: Vec<String>,
}

#[derive(Deserialize)]
struct Profile {
    name: String,
    keys: Vec<String>,
}

#[derive(Deserialize)]
struct Override {
    profile: String,
    event_ids: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Event<'a> {
    id: &'a str,
    section: &'a str,
    delivery_class: &'a str,
    privacy_class: &'a str,
    projected_keys: Vec<&'a str>,
    projection_profile: &'a str,
    producer_symbols: &'a [String],
    source_paths: &'a [String],
    authoritative_outcome: &'a str,
    reducer_effect: &'a str,
    projection: &'a str,
    test: &'a str,
    migration_task: u32,
}

fn resolve(inventory: &Inventory) -> DynResult<Vec<Event<'_>>> {
    let allowed: BTreeSet<_> = inventory.projected_event_keys.iter().collect();
    let forbidden: BTreeSet<_> = inventory.forbidden_projection_keys.iter().collect();
    let mut profiles = BTreeMap::new();
    for profile in &inventory.projection_profiles {
        let keys: BTreeSet<_> = profile.keys.iter().collect();
        if !keys.is_disjoint(&forbidden) || !keys.is_subset(&allowed) || keys == allowed {
            return Err(format!("invalid projection profile: {}", profile.name).into());
        }
        profiles.insert(profile.name.as_str(), &profile.keys);
    }
    let ids: BTreeSet<_> = inventory
        .catalog
        .spec_bullets
        .iter()
        .flat_map(|bullet| &bullet.event_ids)
        .collect();
    let mut overrides = BTreeMap::new();
    for entry in &inventory.event_projection_overrides {
        if !profiles.contains_key(entry.profile.as_str()) {
            return Err("unresolved projection profile".into());
        }
        for id in &entry.event_ids {
            if !ids.contains(id)
                || overrides
                    .insert(id.as_str(), entry.profile.as_str())
                    .is_some()
            {
                return Err("unresolved or duplicate event projection override".into());
            }
        }
    }
    let mut events = BTreeMap::new();
    for bullet in &inventory.catalog.spec_bullets {
        let family = inventory
            .families
            .iter()
            .find(|family| family.section == bullet.section)
            .ok_or("unresolved event family")?;
        for id in &bullet.event_ids {
            let delivery = if inventory.catalog.terminal_event_ids.contains(id) {
                "terminal"
            } else if inventory.catalog.progress_event_ids.contains(id) {
                "progress"
            } else {
                &family.default_delivery_class
            };
            let profile = overrides
                .get(id.as_str())
                .copied()
                .or(family.default_projection_profile.as_deref())
                .ok_or("missing event-key entry")?;
            let keys = profiles
                .get(profile)
                .ok_or("unresolved projection profile")?;
            let projected_keys = inventory
                .projected_envelope_keys
                .iter()
                .chain(&inventory.base_projected_event_keys)
                .chain(keys.iter())
                .map(String::as_str)
                .collect();
            events.insert(
                id.as_str(),
                Event {
                    id,
                    section: &bullet.section,
                    delivery_class: delivery,
                    privacy_class: &family.privacy_class,
                    projected_keys,
                    projection_profile: profile,
                    producer_symbols: &family.producer_symbols,
                    source_paths: &family.source_paths,
                    authoritative_outcome: &family.authoritative_outcome,
                    reducer_effect: &family.reducer_effect,
                    projection: &family.projection,
                    test: &family.test,
                    migration_task: family.migration_task,
                },
            );
        }
    }
    Ok(events.into_values().collect())
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let values = super::options(
        args,
        &["--inventory", "--markdown", "--typescript", "--check"],
    )?;
    let path = values.get("--inventory").map_or_else(
        || PathBuf::from("crates/mesh-llm-runtime-event-contracts/inventory/runtime_events.toml"),
        PathBuf::from,
    );
    let inventory: Inventory = toml::from_str(&std::fs::read_to_string(path)?)?;
    let events = resolve(&inventory)?;
    let markdown = values.get("--markdown").map_or_else(
        || PathBuf::from("docs/design/RUNTIME_EVENT_INVENTORY.md"),
        PathBuf::from,
    );
    let typescript = values.get("--typescript").map_or_else(
        || {
            PathBuf::from(
                "crates/mesh-llm-runtime-event-contracts/fixtures/runtime_event_inventory.ts",
            )
        },
        PathBuf::from,
    );
    let check = values.contains_key("--check");
    super::publish(
        &markdown,
        render::markdown(&inventory, &events).as_bytes(),
        check,
    )?;
    super::publish(
        &typescript,
        render::typescript(&inventory, &events)?.as_bytes(),
        check,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Inventory {
        toml::from_str(include_str!(
            "../../../../../crates/mesh-llm-runtime-event-contracts/inventory/runtime_events.toml"
        ))
        .unwrap()
    }

    #[test]
    fn event_profiles_exclude_unrelated_sensitive_keys() {
        let inventory = fixture();
        let events = resolve(&inventory).unwrap();
        let received = events
            .iter()
            .find(|event| event.id == "request_received")
            .unwrap();
        let failed = events
            .iter()
            .find(|event| event.id == "request_failed")
            .unwrap();
        assert!(!received.projected_keys.contains(&"reason_code"));
        assert!(failed.projected_keys.contains(&"reason_code"));
    }

    #[test]
    fn invalid_projection_contract_is_rejected() {
        for key in ["prompt", "mystery"] {
            let mut inventory = fixture();
            inventory.projection_profiles[0].keys.push(key.into());
            let result = resolve(&inventory);
            assert!(result.is_err());
        }
    }
}
