use super::*;

#[test]
fn local_model_only_accepts_the_direct_openai_surface() {
    let options = RuntimeOptions {
        local_model_only: true,
        model: vec!["/models/model.gguf".into()],
        listen_all: true,
        port: 19_985,
        ..RuntimeOptions::default()
    };

    validate_local_model_only_options(&options).unwrap();
}

#[test]
fn local_model_only_rejects_every_mesh_startup_shape() {
    let cases = [
        RuntimeOptions {
            local_model_only: true,
            auto: true,
            ..RuntimeOptions::default()
        },
        RuntimeOptions {
            local_model_only: true,
            join: vec!["invite".into()],
            ..RuntimeOptions::default()
        },
        RuntimeOptions {
            local_model_only: true,
            publish: true,
            ..RuntimeOptions::default()
        },
        RuntimeOptions {
            local_model_only: true,
            split: true,
            ..RuntimeOptions::default()
        },
        RuntimeOptions {
            local_model_only: true,
            relay: vec!["https://relay.invalid".into()],
            ..RuntimeOptions::default()
        },
    ];

    for options in cases {
        validate_local_model_only_options(&options)
            .expect_err("mesh topology option must fail closed");
    }
}

#[test]
fn local_model_only_rejects_management_and_release_surfaces() {
    let cases = [
        RuntimeOptions {
            local_model_only: true,
            owner_required: true,
            ..RuntimeOptions::default()
        },
        RuntimeOptions {
            local_model_only: true,
            auto_update: true,
            ..RuntimeOptions::default()
        },
        RuntimeOptions {
            local_model_only: true,
            plugin: Some("example".into()),
            ..RuntimeOptions::default()
        },
    ];

    for options in cases {
        validate_local_model_only_options(&options)
            .expect_err("non-serving surface must fail closed");
    }
}

#[test]
fn local_model_only_rejects_invalid_capacity_caps() {
    for max_vram in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let options = RuntimeOptions {
            local_model_only: true,
            max_vram: Some(max_vram),
            ..RuntimeOptions::default()
        };
        validate_local_model_only_options(&options)
            .expect_err("invalid local capacity cap must fail closed");
    }
}
