use crate::command::DynResult;
use serde::Serialize;
use std::collections::BTreeSet;
use std::path::PathBuf;

#[derive(Serialize)]
struct Bullet {
    section: String,
    family: String,
    ordinal: usize,
    text: String,
}

#[derive(Serialize)]
struct Manifest {
    schema_version: u8,
    source: &'static str,
    bullet_count: usize,
    bullets: Vec<Bullet>,
}

fn extract(text: &str) -> DynResult<Manifest> {
    let mut bullets: Vec<Bullet> = Vec::new();
    let mut section = String::new();
    let mut family = String::new();
    let mut ordinal = 0;
    let mut collecting = false;
    for line in text.lines() {
        if line.starts_with("## 9.") {
            break;
        }
        if let Some((number, label)) = line
            .strip_prefix("### ")
            .and_then(|heading| heading.split_once(' '))
            && let Some(index) = number
                .strip_prefix("8.")
                .and_then(|index| index.parse::<u8>().ok())
            && (1..=15).contains(&index)
        {
            section = number.to_owned();
            family = label.to_owned();
            ordinal = 0;
            collecting = false;
            continue;
        }
        if matches!(
            line,
            "Required events:" | "Required derived state events:" | "Required events or counters:"
        ) {
            if section.is_empty() {
                return Err("required-event list appears outside section 8 family".into());
            }
            collecting = true;
            continue;
        }
        if !collecting {
            continue;
        }
        if let Some(item) = line.strip_prefix("- ") {
            ordinal += 1;
            bullets.push(Bullet {
                section: section.clone(),
                family: family.clone(),
                ordinal,
                text: item.trim().to_owned(),
            });
        } else if line.starts_with("  ") && ordinal > 0 {
            if let Some(bullet) = bullets.last_mut() {
                bullet.text.push(' ');
                bullet.text.push_str(line.trim());
            }
        } else if !line.is_empty() {
            collecting = false;
        }
    }
    let sections: BTreeSet<_> = bullets
        .iter()
        .map(|bullet| bullet.section.clone())
        .collect();
    let expected = (1..=15).map(|index| format!("8.{index}")).collect();
    if sections != expected {
        return Err("section 8 family mismatch".into());
    }
    Ok(Manifest {
        schema_version: 1,
        source: ".omo/specs/event-system.md",
        bullet_count: bullets.len(),
        bullets,
    })
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let values = super::options(args, &["--spec", "--output", "--check"])?;
    let spec = values.get("--spec").map_or_else(
        || PathBuf::from(".omo/specs/event-system.md"),
        PathBuf::from,
    );
    let output = values.get("--output").map_or_else(
        || PathBuf::from("crates/mesh-llm-runtime-event-contracts/inventory/spec_manifest.json"),
        PathBuf::from,
    );
    let manifest = extract(&std::fs::read_to_string(spec)?)?;
    let mut rendered = serde_json::to_vec_pretty(&manifest)?;
    rendered.push(b'\n');
    super::publish(&output, &rendered, values.contains_key("--check"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuation_is_owned_by_its_bullet() {
        let input = (1..=15)
            .map(|index| {
                format!("### 8.{index} Family {index}\nRequired events:\n- event\n  continuation\n")
            })
            .collect::<String>();
        let manifest = extract(&input).unwrap();
        assert!(
            manifest
                .bullets
                .iter()
                .all(|bullet| bullet.text == "event continuation" && bullet.ordinal == 1)
        );
    }

    #[test]
    fn incomplete_family_roster_is_rejected() {
        let result = extract("### 8.1 First\nRequired events:\n- event\n## 9. End\n");
        assert!(result.is_err());
    }
}
