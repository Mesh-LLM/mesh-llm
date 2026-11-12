//! A diagnostic fixture retry never substitutes for an unfiltered native pass.
use super::{Input, plan::Step};
use std::{collections::BTreeMap, fs, io::BufRead, path::Path};

pub(super) fn steps(input: &Input, log: &Path) -> Option<Vec<Step>> {
    if input.backend != "metal"
        || !cfg!(target_os = "macos")
        || !super::runner::contains(log, "Bus error").ok()?
    {
        return None;
    }
    let failed =
        fs::read_to_string(input.build.join("Testing/Temporary/LastTestsFailed.log")).ok()?;
    if failed.len() > 65536 {
        return None;
    }
    let names: Vec<_> = failed
        .lines()
        .map(|line| line.split_once(':').map(|(_, name)| name))
        .collect::<Option<_>>()?;
    let generators: Vec<_> = names
        .iter()
        .filter_map(|name| name.strip_prefix("skippy_generate_contract_"))
        .collect();
    let [family] = generators.as_slice() else {
        return None;
    };
    if family.is_empty()
        || !family
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
        || names.iter().any(|name| {
            *name != format!("skippy_generate_contract_{family}")
                && *name != format!("skippy_graph_contract_{family}")
        })
    {
        return None;
    }
    let generator = format!("skippy_generate_contract_{family}");
    let exception = std::io::BufReader::new(fs::File::open(log).ok()?)
        .lines()
        .any(|line| {
            line.is_ok_and(|line| {
                line.contains(&generator)
                    && line.contains("***Exception:")
                    && line.contains("Bus error")
            })
        });
    if !exception {
        return None;
    }
    let base = vec![
        "ctest".into(),
        "--test-dir".into(),
        input.build.to_string_lossy().into_owned(),
        "--output-on-failure".into(),
        "--no-tests=error".into(),
        "--timeout".into(),
        "900".into(),
    ];
    let mut fixture = base.clone();
    fixture.extend(["-R".into(), format!("^skippy_generate_contract_{family}$")]);
    Some(vec![
        Step {
            label: "metal-fixture-diagnostic-retry".into(),
            arguments: fixture,
            environment: BTreeMap::new(),
            identity: false,
        },
        Step {
            label: "unfiltered-native-rerun".into(),
            arguments: base,
            environment: BTreeMap::new(),
            identity: false,
        },
    ])
}
