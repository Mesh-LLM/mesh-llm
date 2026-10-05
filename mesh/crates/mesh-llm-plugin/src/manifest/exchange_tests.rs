use super::*;
use crate::openai_exchange::openai_exchange_hook;
use prost::Message;

#[test]
fn lifecycle_manifest_is_additive_and_packaged() {
    let old = proto::PluginManifest::default();
    let decoded = proto::PluginManifest::decode(old.encode_to_vec().as_slice()).unwrap();
    assert!(decoded.openai_exchange_hook.is_none());
    let manifest = plugin_manifest()
        .item(openai_exchange_hook("observe"))
        .build();
    let decoded = proto::PluginManifest::decode(manifest.encode_to_vec().as_slice()).unwrap();
    assert_eq!(
        decoded.openai_exchange_hook.as_deref().unwrap().handler,
        "observe"
    );
    let packaged: serde_json::Value =
        serde_json::from_str(&package_manifest_json(&manifest).unwrap()).unwrap();
    assert_eq!(packaged["openai_exchange_hook"]["contract_version"], 1);
    assert_eq!(packaged["openai_exchange_hook"]["admission"], false);
}

#[test]
fn credentials_cannot_be_requested_in_packaged_manifest() {
    let mut hook = openai_exchange_hook("observe");
    hook.headers = vec!["Authorization".into()];
    let manifest = plugin_manifest().item(hook).build();
    assert!(package_manifest_json(&manifest).is_err());
}
