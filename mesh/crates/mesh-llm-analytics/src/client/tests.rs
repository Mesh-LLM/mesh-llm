use super::*;
use crate::event::Label;

fn envelope(event: Event, properties: Properties) -> Envelope {
    Envelope {
        event,
        properties,
        captured_at: Utc::now(),
    }
}

fn base() -> Properties {
    Properties::new()
        .with("mesh_llm_version", "0.76.0")
        .with("os", "macos")
}

#[test]
fn every_item_disables_geoip_and_drops_the_client_address() {
    let item = batch_item(
        &envelope(Event::ServeStarted, Properties::new()),
        "install-1",
        &base(),
    );
    let properties = &item["properties"];
    assert_eq!(properties["$geoip_disable"], json!(true));
    assert!(properties["$ip"].is_null());
}

#[test]
fn privacy_properties_cannot_be_overridden_by_a_caller() {
    // `$geoip_disable` is written after caller properties are merged, so even
    // a caller that tried to set it loses.
    let hostile = Properties::new().with("$geoip_disable", false);
    let item = batch_item(&envelope(Event::CliCommand, hostile), "install-1", &base());
    assert_eq!(item["properties"]["$geoip_disable"], json!(true));
}

#[test]
fn event_properties_win_over_base_properties() {
    let overriding = Properties::new().with("os", "linux");
    let item = batch_item(
        &envelope(Event::CliCommand, overriding),
        "install-1",
        &base(),
    );
    assert_eq!(item["properties"]["os"], json!("linux"));
}

#[test]
fn distinct_id_rides_in_properties_as_the_batch_api_expects() {
    let item = batch_item(
        &envelope(Event::InstallFirstRun, Properties::new()),
        "install-42",
        &base(),
    );
    assert_eq!(item["properties"]["distinct_id"], json!("install-42"));
    assert_eq!(item["event"], json!("install_first_run"));
}

#[test]
fn body_carries_the_api_key_and_one_item_per_envelope() {
    let envelopes = vec![
        envelope(Event::ServeStarted, Properties::new()),
        envelope(Event::ServeStopped, Properties::new()),
    ];
    let body = batch_body("phc_test", "install-1", &base(), &envelopes);
    assert_eq!(body["api_key"], json!("phc_test"));
    assert_eq!(body["batch"].as_array().expect("batch").len(), 2);
}

#[test]
fn sanitized_labels_serialize_as_plain_strings() {
    let properties =
        Properties::new().with("model", Label::sanitize("Qwen/Qwen2.5-7B").expect("label"));
    let item = batch_item(
        &envelope(Event::ModelLoaded, properties),
        "install-1",
        &base(),
    );
    assert_eq!(item["properties"]["model"], json!("Qwen/Qwen2.5-7B"));
}

#[test]
fn endpoint_is_built_without_a_doubled_slash() {
    assert_eq!(
        batch_endpoint("https://us.i.posthog.com"),
        "https://us.i.posthog.com/batch/"
    );
    assert_eq!(
        batch_endpoint("https://us.i.posthog.com/"),
        "https://us.i.posthog.com/batch/"
    );
}
