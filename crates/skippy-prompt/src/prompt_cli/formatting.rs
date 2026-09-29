fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn format_stage_mask(mask: i64) -> String {
    if mask <= 0 {
        return "-".to_string();
    }
    let stages = (0..63)
        .filter(|index| (mask & (1_i64 << index)) != 0)
        .map(|index| index.to_string())
        .collect::<Vec<_>>();
    if stages.is_empty() {
        "-".to_string()
    } else {
        stages.join(",")
    }
}

fn stage_mask_count(mask: i64) -> u64 {
    if mask <= 0 {
        return 0;
    }
    (mask as u64).count_ones() as u64
}
