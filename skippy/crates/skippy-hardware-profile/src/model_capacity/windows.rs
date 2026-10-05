//! Windows model capacity, matching adapter names to 64-bit registry memory.
use serde_json::Value;
#[cfg(target_os = "windows")]
fn powershell_output(script: &str) -> Option<String> {
    crate::platform::command_output("powershell", &["-NoProfile", "-Command", script])
}
fn merge_adapter_memory(
    controllers: &[(String, u64)],
    adapter_memory: &[(String, u64)],
) -> Vec<(String, u64)> {
    let mut taken = vec![false; adapter_memory.len()];
    let mut merged = Vec::with_capacity(controllers.len());
    for (name, adapter_ram) in controllers {
        let mut bytes = *adapter_ram;
        for (index, (candidate, candidate_bytes)) in adapter_memory.iter().enumerate() {
            if taken[index] || *candidate_bytes == 0 || candidate != name {
                continue;
            }
            taken[index] = true;
            bytes = *candidate_bytes;
            break;
        }
        merged.push((name.clone(), bytes));
    }
    merged
}

/// Per-adapter `HardwareInformation.qwMemorySize` from the display class key.
///
/// The script ends with an explicit `exit 0`. PowerShell propagates the last
/// statement s $? as its exit code, and reading a class subkey that does not
/// carry the property leaves it false even under `-ErrorAction SilentlyContinue`,
/// which would make `powershell_output` discard a perfectly good JSON body. An
/// adapter without an entry is an ordinary outcome here, not a failure.
#[cfg(target_os = "windows")]
fn read_windows_adapter_memory() -> Vec<(String, u64)> {
    let Some(output) = powershell_output(
        r"Get-ChildItem 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}' -ErrorAction SilentlyContinue | ForEach-Object { $p = Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue; $q = $p.'HardwareInformation.qwMemorySize'; if ($q) { [pscustomobject]@{ Name = $p.DriverDesc; Bytes = [uint64]$q } } } | ConvertTo-Json -Compress; exit 0",
    ) else {
        return Vec::new();
    };
    parse_windows_adapter_memory_json(&output)
}

#[cfg(target_os = "windows")]
pub(super) fn read_windows_video_controllers() -> Vec<(String, u64)> {
    let Some(output) = powershell_output(
        "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json -Compress",
    ) else {
        return Vec::new();
    };
    let controllers = parse_windows_video_controller_json(&output);
    merge_adapter_memory(&controllers, &read_windows_adapter_memory())
}

fn parse_windows_video_controller_json(output: &str) -> Vec<(String, u64)> {
    fn parse_u64(value: &Value) -> Option<u64> {
        match value {
            Value::Number(n) => n.as_u64(),
            Value::String(s) => s.trim().parse::<u64>().ok(),
            _ => None,
        }
    }

    fn parse_entry(value: &Value) -> Option<(String, u64)> {
        let name = value.get("Name")?.as_str()?.trim();
        if name.is_empty() {
            return None;
        }
        let adapter_ram = value.get("AdapterRAM").and_then(parse_u64).unwrap_or(0);
        Some((name.to_string(), adapter_ram))
    }

    let Ok(value) = serde_json::from_str::<Value>(output) else {
        return Vec::new();
    };

    match value {
        Value::Array(values) => values.iter().filter_map(parse_entry).collect(),
        Value::Object(_) => parse_entry(&value).into_iter().collect(),
        _ => Vec::new(),
    }
}
fn parse_windows_adapter_memory_json(output: &str) -> Vec<(String, u64)> {
    fn parse_u64(value: &Value) -> Option<u64> {
        match value {
            Value::Number(n) => n.as_u64(),
            Value::String(s) => s.trim().parse::<u64>().ok(),
            _ => None,
        }
    }

    fn parse_entry(value: &Value) -> Option<(String, u64)> {
        let name = value.get("Name")?.as_str()?.trim();
        if name.is_empty() {
            return None;
        }
        Some((name.to_string(), value.get("Bytes").and_then(parse_u64)?))
    }

    let Ok(value) = serde_json::from_str::<Value>(output) else {
        return Vec::new();
    };

    match value {
        Value::Array(values) => values.iter().filter_map(parse_entry).collect(),
        Value::Object(_) => parse_entry(&value).into_iter().collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn registry_capacity_overrides_cim_in_adapter_name_order() {
        let controllers = parse_windows_video_controller_json(
            r#"[{"Name":"B","AdapterRAM":4293918720},{"Name":"A","AdapterRAM":0}]"#,
        );
        let registry = parse_windows_adapter_memory_json(
            r#"[{"Name":"A","Bytes":"17179869184"},{"Name":"B","Bytes":34359738368}]"#,
        );
        assert_eq!(
            merge_adapter_memory(&controllers, &registry),
            vec![("B".into(), 34359738368), ("A".into(), 17179869184)]
        );
    }
    #[test]
    fn adapters_without_registry_memory_keep_their_reported_capacity() {
        assert_eq!(
            merge_adapter_memory(
                &[("A".into(), 1000), ("A".into(), 2000)],
                &[("A".into(), 5000)]
            ),
            vec![("A".into(), 5000), ("A".into(), 2000)]
        );
    }
}
