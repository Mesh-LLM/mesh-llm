use super::super::{coordinator::Step, http_checks::Check};
use super::builder::{Builder, Mode, Node};
use crate::command::DynResult;

pub(super) fn append(builder: &mut Builder<'_>) -> DynResult<()> {
    let options = builder.options;
    builder.node(Node {
        name: "config-target",
        binary: &options.current,
        offset: 80,
        mode: Mode::Serve(""),
    })?;
    let sparse = builder
        .directory
        .join("state/config-target/home/.cache/huggingface/hub/owned-node-qa-Q4_K_M.gguf");
    std::fs::create_dir_all(sparse.parent().ok_or("inventory fixture parent")?)?;
    std::fs::File::create(sparse)?.set_len(600_000_000)?;
    builder.steps.push_back(Step::Check {
        name: "config-runtime-bootstrap",
        check: Check::Bootstrap {
            console: options.base + 81,
        },
    });
    builder.node(Node {
        name: "config-controller",
        binary: &options.current,
        offset: 83,
        mode: Mode::Serve(""),
    })?;
    for (name, verb) in [
        ("config-get-config", "get-config"),
        ("config-scan-refresh", "scan-refresh"),
    ] {
        builder.command(
            name,
            &options.current,
            vec![
                "--log-format".into(),
                "json".into(),
                "runtime".into(),
                verb.into(),
                "--port".into(),
                (options.base + 84).to_string(),
                "--endpoint".into(),
                "QA_CONTROL_ENDPOINT".into(),
                "--json".into(),
            ],
            false,
        )?;
    }
    builder.steps.push_back(Step::Check {
        name: "config-current-scan-refresh",
        check: Check::ValidateScan(
            builder
                .directory
                .join("logs/config-scan-refresh.stdout.log"),
        ),
    });
    builder.node(Node {
        name: "config-wrong-owner",
        binary: &options.current,
        offset: 86,
        mode: Mode::WrongOwner,
    })?;
    builder.steps.push_back(Step::Check {
        name: "config-current-scan-refresh-wrong-owner",
        check: Check::WrongOwner {
            console: options.base + 87,
            endpoint: "QA_CONTROL_ENDPOINT".into(),
        },
    });
    if options.local {
        builder.steps.push_back(Step::Check {
            name: "lifecycle-load-unload-ensure-drain",
            check: Check::Lifecycle {
                console: options.base + 84,
            },
        });
        if options.config || !options.released_model.is_empty() {
            builder.steps.push_back(Step::Check {
                name: "lifecycle-legacy-unsupported",
                check: Check::Legacy {
                    current: options.base + 84,
                    released: options.base + 51,
                },
            });
        }
    }
    Ok(())
}
