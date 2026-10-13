//! Admission of the pinned helper's actual existing-repository repository receipt; complete-folder publication stays owned by existing model publisher.
use super::contract;
use serde_json::Value;
pub(super) fn repository(value: &Value, repo: &str, hash: &str) -> bool {
    let p = &value["repository"];
    value["request_sha256"] == hash
        && value["schema_version"] == 1
        && value["status"] == "REPOSITORY_READY"
        && p["repo"] == repo
        && p["completed"] == true
        && p["error"].is_null()
        && p["observed_parent"]
            .as_str()
            .is_some_and(|s| contract::hex(s, 40))
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn generic_repository_receipt_requires_correlated_observed_parent_without_false_mutation_denial()
     {
        let value = json!({"schema_version":1,"request_sha256":"request","status":"REPOSITORY_READY","repository":{"repo":"fixture/model","completed":true,"error":null,"observed_parent":"b".repeat(40),"mutation_attempted":true}});
        assert!(repository(&value, "fixture/model", "request"));
        for (pointer, change) in [
            ("/request_sha256", json!("wrong")),
            ("/repository/error", json!("uncertain")),
            ("/repository/completed", json!(false)),
            ("/repository/observed_parent", json!("main")),
        ] {
            let mut changed = value.clone();
            *changed.pointer_mut(pointer).unwrap() = change;
            assert!(!repository(&changed, "fixture/model", "request"));
        }
    }
}
