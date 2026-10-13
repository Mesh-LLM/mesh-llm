//! Current independent-runtime product contract; frozen schema-one receipts stay historical.
use crate::support::{Scratch, TestResult};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::Path,
    process::{Command, Output},
};

struct Product {
    scratch: Scratch,
    runtime: Value,
}
impl Product {
    fn new(kind: &str, build: Option<Value>) -> Result<Self, Box<dyn std::error::Error>> {
        let scratch = Scratch::new()?;
        let root = scratch.path();
        fs::create_dir_all(root.join("bundle/native-runtimes/rt/lib"))?;
        fs::write(
            root.join("bundle/mesh-llm"),
            "#!/bin/sh\n[ \"$*\" = \"--log-format json --print-build-contract\" ] || exit 97\nprintf '%s\\n' invoked > host-probed\nprintf '%s\\n' '{\"schema_version\":1,\"product_version\":\"1.0\",\"runtime_release\":\"0.9\",\"skippy_abi\":\"7\"}'\nexit 0\n",
        )?;
        fs::set_permissions(
            root.join("bundle/mesh-llm"),
            fs::Permissions::from_mode(0o755),
        )?;
        fs::write(
            root.join("bundle/native-runtimes/rt/lib/runtime"),
            b"runtime fixture\n",
        )?;
        let mut runtime = json!({"schema_version":2,"runtime":{"id":"rt","release_version":"9.0","skippy_abi":"7","backend":{"kind":kind}}});
        if let Some(build) = build {
            runtime["build"] = build;
        }
        let product = Self { scratch, runtime };
        product.write_runtime()?;
        Ok(product)
    }
    fn root(&self) -> &Path {
        self.scratch.path()
    }
    fn write_runtime(&self) -> TestResult {
        fs::write(
            self.root().join("bundle/native-runtimes/rt/manifest.json"),
            serde_json::to_vec(&self.runtime)?,
        )?;
        Ok(())
    }
    fn args(requested: &str) -> Vec<String> {
        [
            "--bundle",
            "bundle",
            "--host",
            "bundle/mesh-llm",
            "--runtime",
            "bundle/native-runtimes/rt",
            "--version",
            "v1.0",
            "--backend",
            requested,
        ]
        .map(str::to_owned)
        .into()
    }
    fn invoke(&self, args: &[String]) -> Result<Output, Box<dyn std::error::Error>> {
        Ok(Command::new(env!("CARGO_BIN_EXE_xtask"))
            .current_dir(self.root())
            .args(["product", "compose"])
            .args(args)
            .output()?)
    }
    fn compose(&self, requested: &str) -> Result<Output, Box<dyn std::error::Error>> {
        self.invoke(&Self::args(requested))
    }
    fn expected(&self, requested: &str) -> Result<Value, Box<dyn std::error::Error>> {
        // This finite tree has exactly two regular input files, with explicit wire ordering.
        let mut tree = Sha256::new();
        for relative in ["lib/runtime", "manifest.json"] {
            tree.update(u64::try_from(relative.len())?.to_be_bytes());
            tree.update(relative.as_bytes());
            tree.update(Sha256::digest(fs::read(
                self.root().join("bundle/native-runtimes/rt").join(relative),
            )?));
        }
        Ok(
            json!({"schema_version":2,"contract":"mesh-llm-product-v2","mesh_version":"1.0","backend":requested,
            "host":{"path":"mesh-llm","sha256":hex::encode(Sha256::digest(fs::read(self.root().join("bundle/mesh-llm"))?)),"required_skippy_abi":"7"},
            "runtime":{"id":"rt","release_version":self.runtime["runtime"]["release_version"],"skippy_abi":"7","path":"native-runtimes/rt","sha256":hex::encode(tree.finalize()),"manifest_sha256":hex::encode(Sha256::digest(fs::read(self.root().join("bundle/native-runtimes/rt/manifest.json"))?))}}),
        )
    }
    fn assert_success(&self, requested: &str, output: &Output) -> TestResult {
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.is_empty() && output.stderr.is_empty());
        let bytes = fs::read(self.root().join("bundle/product-manifest.json"))?;
        assert_eq!(
            serde_json::from_slice::<Value>(&bytes)?,
            self.expected(requested)?
        );
        assert!(bytes.ends_with(b"\n"));
        assert_eq!(fs::read(self.root().join("host-probed"))?, b"invoked\n");
        Ok(())
    }
    fn assert_refusal(&self, args: &[String], diagnostic: &str) -> TestResult {
        let path = self.root().join("bundle/product-manifest.json");
        fs::write(&path, b"prior publication\n")?;
        let output = self.invoke(args)?;
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(diagnostic),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(fs::read(path)?, b"prior publication\n");
        Ok(())
    }
}

pub(super) fn writes_manifests() -> TestResult {
    for (requested, kind, build) in [
        ("cpu", "cpu", Some("cpu")),
        ("metal", "metal", None),
        ("vulkan", "vulkan", Some("vulkan")),
        ("hip", "rocm", Some("hip")),
        ("rocm", "rocm", Some("hip")),
        ("cuda-blackwell", "cuda", Some("cuda-blackwell")),
    ] {
        let product = Product::new(kind, build.map(|backend| json!({"backend":backend})))?;
        product.assert_success(requested, &product.compose(requested)?)?;
        // Absolute and dotted path variants retain the same relative manifest contract.
        let mut args = Product::args(requested);
        args[1] = product.root().join("bundle").to_string_lossy().into();
        args[3] = product
            .root()
            .join("bundle/mesh-llm")
            .to_string_lossy()
            .into();
        args[5] = product
            .root()
            .join("bundle/./native-runtimes/rt/")
            .to_string_lossy()
            .into();
        args[7] = "1.0".into();
        product.assert_success(requested, &product.invoke(&args)?)?;
    }
    let product = Product::new("cpu", Some(Value::Null))?;
    product.assert_success("cpu", &product.compose("cpu")?)?;
    Ok(())
}

pub(super) fn checks_existing_manifest() -> TestResult {
    for mutation in ["host", "runtime", "manifest"] {
        let product = Product::new("cpu", None)?;
        product.assert_success("cpu", &product.compose("cpu")?)?;
        let mut args = Product::args("cpu");
        args.push("--check".into());
        let checked = product.invoke(&args)?;
        assert!(checked.status.success() && checked.stdout.is_empty() && checked.stderr.is_empty());
        let target = product.root().join(match mutation {
            "host" => "bundle/mesh-llm",
            "runtime" => "bundle/native-runtimes/rt/lib/runtime",
            _ => "bundle/product-manifest.json",
        });
        let mut bytes = fs::read(&target)?;
        // Keep a still-executable host contract; alter only its comment bytes.
        if mutation == "host" {
            bytes.extend_from_slice(b"# rebuilt host\n");
        } else if mutation == "runtime" {
            bytes[0] ^= 1;
        } else {
            bytes = b"{\"schema_version\":99}\n".to_vec();
        }
        fs::write(&target, bytes)?;
        let before = fs::read(product.root().join("bundle/product-manifest.json"))?;
        let rejected = product.invoke(&args)?;
        assert_eq!(rejected.status.code(), Some(1));
        assert!(
            String::from_utf8_lossy(&rejected.stderr)
                .contains("product manifest does not match composed bytes")
        );
        assert_eq!(
            fs::read(product.root().join("bundle/product-manifest.json"))?,
            before
        );
    }
    Ok(())
}

pub(super) fn rejects_mismatch() -> TestResult {
    for mutation in [
        "schema",
        "abi",
        "runtime-backend",
        "build-backend",
        "product-version",
        "missing-host",
        "missing-runtime",
        "foreign-host",
        "foreign-runtime",
    ] {
        let mut product = Product::new("cpu", None)?;
        let mut args = Product::args("cpu");
        let diagnostic = match mutation {
            "schema" => {
                product.runtime["schema_version"] = json!(1);
                "schema_version 2"
            }
            "abi" => {
                product.runtime["runtime"]["skippy_abi"] = json!("8");
                "host-required ABI"
            }
            "runtime-backend" => {
                product.runtime["runtime"]["backend"]["kind"] = json!("cuda");
                "backend mismatch"
            }
            "build-backend" => {
                product.runtime["build"] = json!({"backend":"cuda"});
                "build backend mismatch"
            }
            "product-version" => {
                args[7] = "v2.0".into();
                "product version"
            }
            "missing-host" => {
                fs::remove_file(product.root().join("bundle/mesh-llm"))?;
                "No such file"
            }
            "missing-runtime" => {
                args[5] = "absent".into();
                "No such file"
            }
            "foreign-host" => {
                fs::copy(
                    product.root().join("bundle/mesh-llm"),
                    product.root().join("outside-host"),
                )?;
                args[3] = "outside-host".into();
                "is not in the subpath"
            }
            _ => {
                args[1] = "bundle/native-runtimes/rt".into();
                "is not in the subpath"
            }
        };
        product.write_runtime()?;
        product.assert_refusal(&args, diagnostic)?;
        assert!(
            !product
                .root()
                .join("bundle/native-runtimes/rt/product-manifest.json")
                .exists()
        );
        if mutation == "schema" {
            assert!(!product.root().join("host-probed").exists());
        }
    }
    Ok(())
}

pub(super) fn argv_contract() -> TestResult {
    let product = Product::new("cpu", None)?;
    for args in [vec![], vec!["--bundle".into()], vec!["--unknown".into()]] {
        let output = product.invoke(&args)?;
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty() && !output.stderr.is_empty());
        assert!(!product.root().join("bundle/product-manifest.json").exists());
    }
    let help = product.invoke(&["--help".into()])?;
    assert_eq!(help.status.code(), Some(0));
    let help = String::from_utf8(help.stdout)?;
    for option in [
        "--bundle",
        "--host",
        "--runtime",
        "--version",
        "--backend",
        "--check",
    ] {
        assert!(help.contains(option));
    }
    let mut args = vec![
        "--bundle".into(),
        "wrong".into(),
        "--backend".into(),
        "cuda".into(),
    ];
    args.extend(Product::args("cpu"));
    product.assert_success("cpu", &product.invoke(&args)?)?;
    Ok(())
}
