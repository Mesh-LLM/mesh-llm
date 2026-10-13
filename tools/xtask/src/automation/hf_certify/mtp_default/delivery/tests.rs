use super::*;
#[test]
fn composition_jobs_terminal_publication_refuses_late_cancel_deadline_and_prior_failure() {
    for mode in [
        "success",
        "prior",
        "cancel-write",
        "deadline",
        "cancel-emit",
    ] {
        let cancel = Cancellation::default();
        let until = Instant::now() + Duration::from_secs(if mode == "deadline" { 0 } else { 60 });
        let mut receipt = json!({"status":"FAILED","locator":{"delivery_complete":false},"operator":{"rows":[1,2,3]},"export":{"commit_oid":"a".repeat(40)}});
        let writes = std::cell::RefCell::new(Vec::new());
        let emits = std::cell::RefCell::new(Vec::new());
        let result = terminal_publication(
            &mut receipt,
            mode != "prior",
            until,
            &cancel,
            |r| {
                writes.borrow_mut().push(r.clone());
                if mode == "cancel-write" {
                    cancel.cancel();
                }
                Ok(())
            },
            |r| {
                emits.borrow_mut().push(r.clone());
                if mode == "cancel-emit" {
                    cancel.cancel();
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(result, mode == "success");
        assert_eq!(receipt["locator"]["delivery_complete"], mode == "success");
        assert_eq!(
            receipt["status"],
            if mode == "success" {
                "DELIVERED"
            } else {
                "FAILED"
            }
        );
        assert_eq!(receipt["operator"]["rows"], json!([1, 2, 3]));
        assert_eq!(receipt["export"]["commit_oid"], "a".repeat(40));
        assert_eq!(writes.borrow().last().unwrap(), &receipt);
        assert_eq!(emits.borrow().last().unwrap(), &receipt);
        if mode == "cancel-write" {
            assert_eq!(emits.borrow().len(), 1);
            assert_eq!(emits.borrow()[0]["locator"]["delivery_complete"], false);
        }
        if mode == "cancel-emit" {
            assert_eq!(emits.borrow().len(), 2);
            assert_eq!(emits.borrow()[1]["locator"]["delivery_complete"], false);
        }
    }
}

#[cfg(unix)]
#[test]
fn composition_jobs_filesystem_receipt_downgrades_late_terminal_refusal() {
    for mode in ["cancel-write", "deadline-write", "cancel-emit"] {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("delivery.json");
        let mut writer = OwnedDeliveryReceipt::new(path.clone());
        let cancel = Cancellation::default();
        let until = Instant::now()
            + Duration::from_millis(if mode == "deadline-write" { 100 } else { 60000 });
        let mut receipt = json!({"locator":{"delivery_complete":false},"operator":{"rows":[1,2]},"export":{"commit_oid":"a".repeat(40)}});
        let complete = terminal_publication(
            &mut receipt,
            true,
            until,
            &cancel,
            |value| {
                writer.publish(value)?;
                if mode == "cancel-write" {
                    cancel.cancel();
                }
                if mode == "deadline-write"
                    && let Some(remaining) = until.checked_duration_since(Instant::now())
                {
                    std::thread::sleep(remaining + Duration::from_millis(1));
                }
                Ok(())
            },
            |_| {
                if mode == "cancel-emit" {
                    cancel.cancel();
                }
                Ok(())
            },
        )
        .unwrap();
        assert!(!complete);
        let actual: Value =
            serde_json::from_slice(&admission::read(&path, 1048576).unwrap()).unwrap();
        assert_eq!(actual["status"], "FAILED");
        assert_eq!(actual["locator"]["delivery_complete"], false);
        assert_eq!(actual["operator"]["rows"], json!([1, 2]));
        assert_eq!(actual["export"]["commit_oid"], "a".repeat(40));
        // A foreign replacement cannot be overwritten by this owner.
        let foreign = root.path().join("foreign.json");
        std::fs::write(&foreign, b"foreign").unwrap();
        std::fs::rename(&foreign, &path).unwrap();
        assert!(writer.publish(&receipt).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"foreign");
        root.close().unwrap();
    }
}
