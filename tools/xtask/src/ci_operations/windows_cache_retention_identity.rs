use crate::command::DynResult;

pub(super) const NAMESPACE: &str = "mesh-llm-windows-2022-skippy-abi-";

/// The complete compatibility prefix, including toolkit, architecture set and epoch.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, serde::Serialize)]
pub(in crate::ci_operations) struct Family(pub(in crate::ci_operations) String);

pub(in crate::ci_operations) fn family(key: &str) -> DynResult<Family> {
    parts(key).map(|(prefix, _)| Family(format!("{prefix}-")))
}

pub(in crate::ci_operations) fn compatibility(key: &str) -> DynResult<&str> {
    let (prefix, epoch) = parts(key)?;
    prefix
        .strip_suffix(epoch)
        .ok_or_else(|| "cache epoch suffix missing".into())
}

fn parts(key: &str) -> DynResult<(&str, &str)> {
    if key.len() > 512 || !key.is_ascii() {
        return Err("Windows ABI key must contain at most 512 ASCII bytes".into());
    }
    let (prefix, hash) = key.rsplit_once('-').ok_or("missing input hash")?;
    if hash.len() != 64
        || !hash
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err("Windows ABI key requires a lowercase 64-hex input hash".into());
    }
    let body = prefix
        .strip_prefix(NAMESPACE)
        .ok_or("unknown Windows ABI namespace")?;
    let (backend, rest) = body.split_once('-').ok_or("missing backend")?;
    let (architecture, toolchain) = rest.split_once('-').ok_or("missing architecture set")?;
    let epoch = match backend {
        "cpu" if architecture.is_empty() => toolchain.strip_prefix("cpu-"),
        "cuda" if architecture_set(architecture, false) => {
            toolkit_epoch(toolchain, "cuda-", "-Jimver-v0.2.36-")
        }
        "vulkan" if architecture.is_empty() => {
            toolkit_epoch(toolchain, "vulkan-", "-jakoch-v1.5.2-")
        }
        "rocm" if architecture_set(architecture, true) => toolchain
            .strip_prefix("rocm-")
            .and_then(|value| value.split_once(".exe-"))
            .filter(|(installer, _)| segment(installer))
            .map(|(_, epoch)| epoch),
        _ => None,
    }
    .ok_or("malformed Windows ABI backend/toolkit/architecture identity")?;
    if !segment(epoch) {
        return Err("missing or invalid Windows ABI toolchain epoch".into());
    }
    Ok((prefix, epoch))
}

fn toolkit_epoch<'a>(value: &'a str, backend: &str, installer: &str) -> Option<&'a str> {
    let (version, epoch) = value.strip_prefix(backend)?.split_once(installer)?;
    version
        .split('.')
        .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
        .then_some(epoch)
}

fn architecture_set(value: &str, rocm: bool) -> bool {
    let mut seen = std::collections::BTreeSet::new();
    value.split('_').all(|architecture| {
        let digits = if rocm {
            architecture.strip_prefix("gfx").unwrap_or("")
        } else {
            architecture
        };
        !digits.is_empty()
            && digits.bytes().all(|byte| {
                if rocm {
                    byte.is_ascii_hexdigit()
                } else {
                    byte.is_ascii_digit()
                }
            })
            && seen.insert(architecture)
    })
}

fn segment(value: &str) -> bool {
    value
        .as_bytes()
        .first()
        .is_some_and(u8::is_ascii_alphanumeric)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._+-".contains(&byte))
}
