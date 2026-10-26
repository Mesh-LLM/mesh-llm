use super::{FamilyShard, ShardError};
use serde::Serialize;

#[derive(Serialize)]
struct Series<'a> {
    full_patch_sha256: &'a str,
    generator_version: &'static str,
    schema_version: u8,
    shards: Vec<ShardRecord<'a>>,
}

#[derive(Serialize)]
struct ShardRecord<'a> {
    families: &'a [String],
    file: &'a str,
    sha256: &'a str,
    sources: &'a [String],
}

pub(super) fn encode(diff_sha256: &str, shards: &[FamilyShard]) -> Result<Vec<u8>, ShardError> {
    let series = Series {
        full_patch_sha256: diff_sha256,
        generator_version: "0.5.0",
        schema_version: 1,
        shards: shards
            .iter()
            .map(|shard| ShardRecord {
                families: &shard.families,
                file: &shard.file,
                sha256: &shard.sha256,
                sources: &shard.sources,
            })
            .collect(),
    };
    let json = serde_json::to_string_pretty(&series)?;
    let mut ascii = String::with_capacity(json.len() + 1);
    for character in json.chars() {
        if character >= '\u{7f}' {
            for unit in character.encode_utf16(&mut [0; 2]) {
                ascii.push_str(&format!("\\u{unit:04x}"));
            }
        } else {
            ascii.push(character);
        }
    }
    ascii.push('\n');
    Ok(ascii.into_bytes())
}
