use crate::command::DynResult;
use sha2::{Digest as _, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::Read,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

pub(super) struct Input {
    pub binary: PathBuf,
    pub model: PathBuf,
    pub native: PathBuf,
    pub python: PathBuf,
    python_target: PathBuf,
    pub node: PathBuf,
    pub node_modules: PathBuf,
    pub parent: PathBuf,
    pub output: PathBuf,
    pub controller: PathBuf,
    pub wrapper: PathBuf,
    pub sdk_source: PathBuf,
    pub device: String,
    pub cuda: Option<String>,
    pub api: u16,
    pub console: u16,
    pub timeout: Duration,
    pins: Vec<(PathBuf, String)>,
    deadline: Instant,
}
impl Input {
    pub(super) fn admit(root: &Path, args: &[String]) -> DynResult<Self> {
        let sdk = super::super::python_sdk_source::admit(root)?;
        Self::admit_source(root, args, sdk)
    }
    pub(super) fn admit_source(
        root: &Path,
        args: &[String],
        sdk: super::super::python_sdk_source::Source,
    ) -> DynResult<Self> {
        let [verb, args @ ..] = args else {
            return Err("SDK compatibility verb required".into());
        };
        if verb != "run" {
            return Err("SDK compatibility only admits run".into());
        }
        let (pairs, tail) = args.as_chunks::<2>();
        if !tail.is_empty() {
            return Err("named SDK compatibility option pairs required".into());
        }
        let mut values = BTreeMap::new();
        for [flag, value] in pairs {
            if !matches!(
                flag.as_str(),
                "--binary"
                    | "--native-runtime-root"
                    | "--model"
                    | "--python"
                    | "--node"
                    | "--node-modules"
                    | "--state-parent"
                    | "--output"
                    | "--device"
                    | "--api-port"
                    | "--console-port"
                    | "--cuda-visible-devices"
                    | "--timeout-secs"
            ) || values.insert(flag.as_str(), value.as_str()).is_some()
            {
                return Err("unknown or duplicate SDK compatibility option".into());
            }
        }
        let required = |key| {
            values
                .get(key)
                .copied()
                .ok_or("SDK compatibility required option missing")
        };
        let directory = |key| -> DynResult<PathBuf> {
            let path = absolute(required(key)?)?.canonicalize()?;
            if !path.is_dir() {
                return Err("SDK directory required".into());
            }
            Ok(path)
        };
        let binary = executable(required("--binary")?)?;
        let python = absolute(required("--python")?)?;
        let python_target = executable(required("--python")?)?;
        let node = executable(required("--node")?)?;
        if node.file_name().is_none_or(|n| n != "node") {
            return Err("SDK Node must be explicit native node executable".into());
        }
        let model = absolute(required("--model")?)?.canonicalize()?;
        let node_modules = directory("--node-modules")?;
        let package = node_modules.join("openai/package.json");
        let metadata: serde_json::Value =
            serde_json::from_slice(&bounded_file(&package, 1048576)?)?;
        if metadata.get("name").and_then(|v| v.as_str()) != Some("openai") {
            return Err("admitted OpenAI Node module required".into());
        }
        let parent = directory("--state-parent")?;
        let output = absolute(required("--output")?)?;
        if output.symlink_metadata().is_ok() || !output.parent().is_some_and(Path::is_dir) {
            return Err("fresh SDK evidence output with existing parent required".into());
        }
        let device = required("--device")?.to_owned();
        if !matches!(device.as_str(), "CPU" | "MTL0" | "CUDA0") {
            return Err("SDK device refused".into());
        }
        let cuda = values
            .get("--cuda-visible-devices")
            .map(|s| (*s).to_owned());
        if (device == "CUDA0") != cuda.is_some()
            || cuda.as_ref().is_some_and(|s| {
                !s.starts_with("GPU-")
                    || s.len() != 40
                    || !s[4..].bytes().all(|b| b.is_ascii_hexdigit() || b == b'-')
            })
        {
            return Err("CUDA SDK execution requires explicit full GPU UUID; other devices forbid CUDA selection".into());
        }
        let port = |key| -> DynResult<u16> {
            let n = required(key)?.parse::<u16>()?;
            if n == 0 {
                return Err("nonzero SDK ports required".into());
            }
            Ok(n)
        };
        let api = port("--api-port")?;
        let console = port("--console-port")?;
        if api == console {
            return Err("SDK ports must differ".into());
        }
        let timeout = values
            .get("--timeout-secs")
            .map_or(Ok(1200), |s| s.parse::<u64>())?;
        if !(1..=3600).contains(&timeout) {
            return Err("SDK whole budget must be1..3600".into());
        }
        let deadline = Instant::now() + Duration::from_secs(timeout);
        let wrapper = root.join("scripts/ci-compat-smoke.sh");
        if bounded_file(&wrapper, 65536)?
            != include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../scripts/ci-compat-smoke.sh"
            ))
        {
            return Err("SDK wrapper source differs from compiled owner".into());
        }
        let controller = std::env::current_exe()?.canonicalize()?;
        let native = directory("--native-runtime-root")?;
        let mut paths = vec![
            wrapper.clone(),
            binary.clone(),
            model.clone(),
            python_target.clone(),
            node.clone(),
            package,
            controller.clone(),
            sdk.root.join("ci/required-sdk-python/requirements.lock"),
            root.join("scripts/ci-openai-node-smoke.cjs"),
        ];
        for leaf in [
            "ci-openai-python-smoke.py",
            "ci-litellm-smoke.py",
            "ci-langchain-openai-smoke.py",
        ] {
            paths.push(sdk.root.join("scripts").join(leaf));
        }
        paths.extend(sdk.pins.into_iter().map(|(path, _)| path));
        let pins = paths
            .into_iter()
            .map(|p| Ok((p.clone(), digest(&p, deadline)?)))
            .collect::<DynResult<_>>()?;
        Ok(Self {
            binary,
            model,
            native,
            python,
            python_target,
            node,
            node_modules,
            parent,
            output,
            controller,
            wrapper,
            sdk_source: sdk.root,
            device,
            cuda,
            api,
            console,
            timeout: Duration::from_secs(timeout),
            pins,
            deadline,
        })
    }
    pub(super) fn pins(&self) -> &[(PathBuf, String)] {
        &self.pins
    }
    pub(super) fn custody(&self) -> DynResult<()> {
        if self.python.canonicalize()? != self.python_target {
            return Err("SDK interpreter invocation target changed".into());
        }
        for (path, hash) in &self.pins {
            if digest(path, self.deadline)? != *hash {
                return Err("SDK input custody changed".into());
            }
        }
        Ok(())
    }
}
fn absolute(value: &str) -> DynResult<PathBuf> {
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err("SDK inputs must be absolute".into());
    }
    Ok(path)
}
fn executable(value: &str) -> DynResult<PathBuf> {
    use std::os::unix::fs::PermissionsExt;
    let p = absolute(value)?.canonicalize()?;
    let m = fs::metadata(&p)?;
    if !m.is_file() || m.permissions().mode() & 0o111 == 0 {
        return Err("SDK executable refused".into());
    }
    Ok(p)
}
fn reader(path: &Path) -> DynResult<fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    let f = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    if !f.metadata()?.is_file() {
        return Err("SDK input must be regular nonsymlink file".into());
    }
    Ok(f)
}
pub(super) fn bounded_file(path: &Path, cap: u64) -> DynResult<Vec<u8>> {
    let f = reader(path)?;
    if f.metadata()?.len() > cap {
        return Err("SDK input exceeds bound".into());
    }
    let mut b = vec![];
    f.take(cap + 1).read_to_end(&mut b)?;
    if b.len() as u64 > cap {
        return Err("SDK input grew beyond bound".into());
    }
    Ok(b)
}
fn digest(path: &Path, deadline: Instant) -> DynResult<String> {
    let mut f = reader(path)?;
    if f.metadata()?.len() > 8589934592 {
        return Err("SDK input exceeds8GiB bound".into());
    }
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    let mut total = 0;
    loop {
        if Instant::now() >= deadline {
            return Err("SDK custody deadline expired".into());
        }
        let n = f.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        total += n as u64;
        if total > 8589934592 {
            return Err("SDK input grew beyond bound".into());
        }
        hash.update(&buffer[..n]);
    }
    Ok(hex::encode(hash.finalize()))
}
