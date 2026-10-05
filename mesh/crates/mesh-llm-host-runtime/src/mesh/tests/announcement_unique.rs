#[test]
fn test_peer_announcement_gpu_serde_roundtrip() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct TestAnnouncement {
        #[serde(default)]
        gpu_name: Option<String>,
        #[serde(default)]
        hostname: Option<String>,
    }

    let test = TestAnnouncement {
        gpu_name: Some("NVIDIA A100".to_string()),
        hostname: Some("worker-01".to_string()),
    };

    let json = serde_json::to_string(&test).unwrap();
    let decoded: TestAnnouncement = serde_json::from_str(&json).unwrap();

    assert_eq!(decoded.gpu_name, Some("NVIDIA A100".to_string()));
    assert_eq!(decoded.hostname, Some("worker-01".to_string()));
}

#[test]
fn test_peer_announcement_backward_compat_no_hw_fields() {
    #[derive(Deserialize, Debug)]
    struct TestAnnouncement {
        #[serde(default)]
        gpu_name: Option<String>,
        #[serde(default)]
        hostname: Option<String>,
    }

    let json = r#"{"other_field": "value"}"#;
    let decoded: TestAnnouncement = serde_json::from_str(json).unwrap();

    assert_eq!(decoded.gpu_name, None);
    assert_eq!(decoded.hostname, None);
}

#[test]
fn test_peer_announcement_backward_compat_with_hw_fields() {
    #[derive(Deserialize, Debug)]
    struct TestAnnouncement {
        #[serde(default)]
        gpu_name: Option<String>,
        #[serde(default)]
        hostname: Option<String>,
    }

    let json = r#"{"gpu_name": "NVIDIA H100", "hostname": "gpu-server-02"}"#;
    let decoded: TestAnnouncement = serde_json::from_str(json).unwrap();

    assert_eq!(decoded.gpu_name, Some("NVIDIA H100".to_string()));
    assert_eq!(decoded.hostname, Some("gpu-server-02".to_string()));
}

#[test]
fn test_peer_announcement_hostname_serde_roundtrip() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct TestAnnouncement {
        #[serde(default)]
        gpu_name: Option<String>,
        #[serde(default)]
        hostname: Option<String>,
    }

    let test = TestAnnouncement {
        gpu_name: None,
        hostname: Some("compute-node-42".to_string()),
    };

    let json = serde_json::to_string(&test).unwrap();
    let decoded: TestAnnouncement = serde_json::from_str(&json).unwrap();

    assert_eq!(decoded.hostname, Some("compute-node-42".to_string()));
    assert_eq!(decoded.gpu_name, None);
}

#[test]
fn test_peer_payload_hw_fields() {
    #[derive(Serialize, Debug)]
    struct TestPeerPayload {
        id: String,
        gpu_name: Option<String>,
        hostname: Option<String>,
    }

    let payload = TestPeerPayload {
        id: "peer-123".to_string(),
        gpu_name: Some("NVIDIA A100".to_string()),
        hostname: Some("worker-01".to_string()),
    };

    let json = serde_json::to_string(&payload).unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(value["gpu_name"], "NVIDIA A100");
    assert_eq!(value["hostname"], "worker-01");
}

#[test]
fn test_enumerate_host_false_omits_hw_fields_in_announcement() {
    let enumerate_host = false;
    let gpu_name: Option<String> = Some("NVIDIA RTX 5090".to_string());
    let hostname: Option<String> = Some("carrack".to_string());
    let gpu_vram: Option<String> = Some("34359738368".to_string());

    let gossip_gpu_name = if enumerate_host {
        gpu_name.clone()
    } else {
        None
    };
    let gossip_hostname = if enumerate_host {
        hostname.clone()
    } else {
        None
    };
    let gossip_gpu_vram = if enumerate_host {
        gpu_vram.clone()
    } else {
        None
    };

    assert_eq!(gossip_gpu_name, None);
    assert_eq!(gossip_hostname, None);
    assert_eq!(gossip_gpu_vram, None);
}

#[test]
fn test_enumerate_host_true_includes_hw_fields_in_announcement() {
    let enumerate_host = true;
    let gpu_name: Option<String> = Some("NVIDIA RTX 5090".to_string());
    let hostname: Option<String> = Some("carrack".to_string());
    let gpu_vram: Option<String> = Some("34359738368".to_string());

    let gossip_gpu_name = if enumerate_host {
        gpu_name.clone()
    } else {
        None
    };
    let gossip_hostname = if enumerate_host {
        hostname.clone()
    } else {
        None
    };
    let gossip_gpu_vram = if enumerate_host {
        gpu_vram.clone()
    } else {
        None
    };

    assert_eq!(gossip_gpu_name, Some("NVIDIA RTX 5090".to_string()));
    assert_eq!(gossip_hostname, Some("carrack".to_string()));
    assert_eq!(gossip_gpu_vram, Some("34359738368".to_string()));
}

#[test]
fn test_is_soc_always_included_regardless_of_enumerate_host() {
    for enumerate_host in [false, true] {
        let is_soc: Option<bool> = Some(true);
        let gpu_name: Option<String> = Some("Tegra AGX Orin".to_string());

        let gossip_gpu_name = if enumerate_host {
            gpu_name.clone()
        } else {
            None
        };

        assert_eq!(is_soc, Some(true), "is_soc must always be sent");
        if enumerate_host {
            assert!(gossip_gpu_name.is_some());
        } else {
            assert!(gossip_gpu_name.is_none());
        }
    }
}

#[test]
fn test_peer_announcement_backward_compat_is_soc_gpu_vram() {
    #[derive(Deserialize, Debug)]
    struct TestAnnouncement {
        #[serde(default)]
        is_soc: Option<bool>,
        #[serde(default)]
        gpu_vram: Option<String>,
    }

    let json = r#"{"other_field": "value"}"#;
    let decoded: TestAnnouncement = serde_json::from_str(json).unwrap();
    assert_eq!(
        decoded.is_soc, None,
        "old nodes without is_soc should default to None"
    );
    assert_eq!(
        decoded.gpu_vram, None,
        "old nodes without gpu_vram should default to None"
    );
}

#[test]
fn test_peer_announcement_backward_compat_no_bandwidth_field() {
    #[derive(Deserialize)]
    struct TestAnnouncement {
        #[serde(
            default,
            rename = "gpu_bandwidth_gbps",
            alias = "gpu_mem_bandwidth_gbps"
        )]
        gpu_mem_bandwidth_gbps: Option<String>,
    }

    let json = r#"{"other_field": "value"}"#;
    let decoded: TestAnnouncement = serde_json::from_str(json).unwrap();

    assert_eq!(decoded.gpu_mem_bandwidth_gbps, None);
}
