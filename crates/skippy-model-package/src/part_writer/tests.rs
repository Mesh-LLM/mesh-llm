use super::*;
use crate::source_inventory::SourceInventory;
use crate::tensor_payload::TensorLocation;
use crate::test_gguf::{explicit, fixture, tensor};
use skippy_model::gguf_catalog::read_gguf_catalog;

fn inventory_for(source: &Path) -> SourceInventory {
    let input =
        crate::package::resolve_package_input(source.display().to_string(), explicit(source))
            .unwrap();
    SourceInventory::read(&input).unwrap()
}

#[test]
fn part_round_trips_exact_tensor_payloads_from_the_source() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("model.gguf");
    fixture(
        &source,
        &[
            tensor("blk.0.attn_q.weight", 0),
            tensor("blk.0.attn_k.weight", 32),
            tensor("blk.0.attn_v.weight", 64),
        ],
        None,
    );
    let inventory = inventory_for(&source);
    let out = temp.path().join("layers/layer-00000-part00.gguf");
    write_part(
        &inventory,
        &[
            "blk.0.attn_k.weight".to_string(),
            "blk.0.attn_q.weight".to_string(),
        ],
        &out,
    )
    .unwrap();

    // The part is a legal GGUF catalog: metadata preserved, split bookkeeping
    // keys dropped, alignment metadata present, offsets aligned.
    let directory = read_gguf_catalog(&out).unwrap();
    assert_eq!(directory.tensors.len(), 2);
    assert_eq!(
        directory
            .tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<Vec<_>>(),
        ["blk.0.attn_k.weight", "blk.0.attn_q.weight"]
    );
    assert!(directory.metadata.contains_key("general.alignment"));
    assert!(!directory.metadata.contains_key("split.no"));

    // Payload bytes are byte-identical to the independent source extents.
    let (_, emitted) = crate::source_inventory::inspect(&out, "layer-00000-part00").unwrap();
    let mut emitted_locations = BTreeMap::new();
    for tensor_entry in &emitted.entries {
        emitted_locations.insert(
            tensor_entry.name.clone(),
            TensorLocation {
                path: out.clone(),
                tensor: tensor_entry.clone(),
            },
        );
    }
    let mut source_locations = BTreeMap::new();
    for shard in &inventory.shards {
        for tensor_entry in &shard.tensors.entries {
            source_locations.insert(
                tensor_entry.name.clone(),
                TensorLocation {
                    path: shard.path.clone(),
                    tensor: tensor_entry.clone(),
                },
            );
        }
    }
    for name in ["blk.0.attn_k.weight", "blk.0.attn_q.weight"] {
        crate::tensor_payload::compare_tensor_payload(name, &source_locations, &emitted_locations)
            .unwrap();
    }
    // The unbound tensor stays absent.
    assert!(!emitted_locations.contains_key("blk.0.attn_v.weight"));
}

#[test]
fn a_part_requires_at_least_one_tensor() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("model.gguf");
    fixture(&source, &[tensor("blk.0.attn_q.weight", 0)], None);
    let inventory = inventory_for(&source);
    let out = temp.path().join("layers/layer-00000-part00.gguf");
    assert!(write_part(&inventory, &[], &out).is_err());
    assert!(!out.exists());
}
