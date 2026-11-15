//! Exercise the real CMD driver and Rust metadata selector on native Windows.
use super::{execute, function, quote, script};
use std::{
    fs,
    path::{Path, PathBuf},
};

struct LinkerFixture {
    _directory: tempfile::TempDir,
    root: PathBuf,
    sysroot: PathBuf,
    tools: PathBuf,
    driver: PathBuf,
    helper: PathBuf,
    response: PathBuf,
}

impl LinkerFixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("linker-é-模型");
        let tools = root.join("tools");
        let sysroot = root.join("sysroot");
        let linker = sysroot.join("lib/rustlib/x86_64-pc-windows-msvc/bin");
        fs::create_dir_all(&tools).unwrap();
        fs::create_dir_all(&linker).unwrap();
        let fixture = crate::fixture_paths::examples_dir().join("migration_generator_fixture.exe");
        fs::copy(&fixture, tools.join("rustc.exe"))
            .expect("prebuild the native fixture through Just");
        fs::copy(&fixture, linker.join("rust-lld.exe")).unwrap();
        let helper = crate::fixture_paths::target_profile_dir().join("xtask.exe");
        assert!(
            helper.is_file(),
            "prebuild the real xtask executable through Just"
        );
        let driver = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .unwrap()
            .join("scripts")
            .join("cargo-linker.cmd");
        let response = root.join("link arguments.rsp");
        fs::write(&response, b"\xff\xfe/\0D\0E\0B\0U\0G\0\r\0\n\0").unwrap();
        Self {
            _directory: directory,
            root,
            sysroot,
            tools,
            driver,
            helper,
            response,
        }
    }

    fn preamble(&self, log: &Path) -> String {
        format!(
            "$ErrorActionPreference='Stop'; $env:ComSpec=[IO.Path]::Combine($env:SystemRoot,'System32','cmd.exe'); $env:PATHEXT='.COM;.EXE;.BAT;.CMD'; $env:PATH={}+';'+$env:PATH; $env:MESH_WINDOWS_LINK_FIXTURE_SYSROOT={}; $env:MESH_WINDOWS_LINK_FIXTURE_LOG={}; $env:MESH_LLM_AUTOMATION_BIN={}; $env:MESH_CI_STABLE_PROC_MACRO_METADATA='1'; $env:SOURCE_DATE_EPOCH='0'; $env:CARGO_MANIFEST_DIR={}; $env:CARGO_CRATE_NAME='macro_fixture'; Remove-Item Env:CARGO_PRIMARY_PACKAGE -ErrorAction SilentlyContinue;",
            quote(self.tools.to_str().unwrap()),
            quote(self.sysroot.to_str().unwrap()),
            quote(log.to_str().unwrap()),
            quote(self.helper.to_str().unwrap()),
            quote(self.root.to_str().unwrap())
        )
    }

    fn call(
        &self,
        label: &str,
        is_macro: bool,
        name: &str,
        primary: bool,
        enabled: bool,
        code: i32,
    ) -> serde_json::Value {
        fs::write(
            self.root.join("Cargo.toml"),
            format!("[package]\nname='macro-fixture'\n[lib]\nproc-macro={is_macro}\n"),
        )
        .unwrap();
        let log = self.root.join(format!("{label}.json"));
        let before = fs::read(&self.response).unwrap();
        let primary = if primary {
            "$env:CARGO_PRIMARY_PACKAGE='1';"
        } else {
            ""
        };
        let enabled = if enabled {
            ""
        } else {
            "Remove-Item Env:MESH_CI_STABLE_PROC_MACRO_METADATA; Remove-Item Env:MESH_LLM_AUTOMATION_BIN;"
        };
        let body = format!(
            "{} $env:CARGO_CRATE_NAME={}; $env:MESH_WINDOWS_LINK_FIXTURE_EXIT='{code}'; {primary} {enabled} & {} '/DEBUG' {}; if ($LASTEXITCODE -ne {code}) {{ throw ('linker exit mismatch: '+$LASTEXITCODE) }}; exit 0",
            self.preamble(&log),
            quote(name),
            quote(self.driver.to_str().unwrap()),
            quote(&format!("@{}", self.response.display()))
        );
        let report = execute(&body, &self.root);
        assert!(report.success(), "{report:?}");
        assert!(
            String::from_utf8_lossy(&report.stdout.bytes_retained).contains("native-link stdout")
        );
        assert!(
            String::from_utf8_lossy(&report.stderr.bytes_retained).contains("native-link stderr")
        );
        assert_eq!(
            fs::read(&self.response).unwrap(),
            before,
            "never rewrite a linker response file"
        );
        serde_json::from_slice(&fs::read(log).unwrap()).unwrap()
    }

    fn missing_helper_refuses_without_fallback(&self) {
        let log = self.root.join("missing-helper.json");
        let body = format!(
            "{} Remove-Item Env:MESH_LLM_AUTOMATION_BIN; & {} '/DEBUG'; if ($LASTEXITCODE -ne 1) {{ throw 'missing helper must fail closed' }}; exit 0",
            self.preamble(&log),
            quote(self.driver.to_str().unwrap())
        );
        let report = execute(&body, &self.root);
        assert!(report.success(), "{report:?}");
        assert!(
            String::from_utf8_lossy(&report.stderr.bytes_retained)
                .contains("requires restored product-source automation")
        );
        assert!(
            !log.exists(),
            "never invoke the linker after a missing-helper refusal"
        );
    }

    fn mesh_host_selects_adapter_only_with_opt_in(&self) {
        let llvm = self
            .sysroot
            .join("lib/rustlib/x86_64-pc-windows-msvc/bin/rust-lld.exe");
        let body = format!(
            "$ErrorActionPreference='Stop'; $scriptDir={}; function Resolve-LldLinker {{ {} }}; {}; Remove-Item Env:MESH_CI_STABLE_PROC_MACRO_METADATA -ErrorAction SilentlyContinue; Configure-LldLinker; if ($env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER -ne {} -or $env:CARGO_TARGET_AARCH64_PC_WINDOWS_MSVC_LINKER -ne {}) {{ throw 'default Mesh linker must remain LLVM' }}; $env:MESH_CI_STABLE_PROC_MACRO_METADATA='1'; Configure-LldLinker; $expectedDriver=[IO.Path]::GetFullPath({}); if (![string]::Equals([IO.Path]::GetFullPath($env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER),$expectedDriver,[StringComparison]::OrdinalIgnoreCase) -or ![string]::Equals([IO.Path]::GetFullPath($env:CARGO_TARGET_AARCH64_PC_WINDOWS_MSVC_LINKER),$expectedDriver,[StringComparison]::OrdinalIgnoreCase)) {{ throw 'opted-in Mesh host must use the adapter driver' }}",
            quote(self.driver.parent().unwrap().to_str().unwrap()),
            quote(llvm.to_str().unwrap()),
            function(&script(), "Configure-LldLinker"),
            quote(llvm.to_str().unwrap()),
            quote(llvm.to_str().unwrap()),
            quote(self.driver.to_str().unwrap())
        );
        let report = execute(&body, &self.root);
        assert!(report.success(), "{report:?}");
    }
}

#[cfg_attr(windows, test)]
fn native_windows_linker_metadata_preserves_product_links_and_failure() {
    let fixture = LinkerFixture::new();
    fixture.mesh_host_selects_adapter_only_with_opt_in();
    for (label, is_macro, name, primary, enabled, code, debug_none) in [
        ("macro", true, "macro_fixture", false, true, 0, true),
        ("normal", false, "macro_fixture", false, true, 0, false),
        ("primary", true, "macro_fixture", true, true, 0, false),
        (
            "build-script",
            true,
            "build_script_build",
            false,
            true,
            0,
            false,
        ),
        ("default", true, "macro_fixture", false, false, 0, false),
        ("failure", true, "macro_fixture", false, true, 7, true),
    ] {
        let result = fixture.call(label, is_macro, name, primary, enabled, code);
        let mut expected = vec![
            "-flavor".to_owned(),
            "link".to_owned(),
            "/DEBUG".to_owned(),
            format!("@{}", fixture.response.display()),
        ];
        if debug_none {
            expected.push("/DEBUG:NONE".into());
        }
        assert_eq!(result["arguments"], serde_json::json!(expected), "{label}");
        assert_eq!(
            result["source_date_epoch"], "0",
            "preserve the caller's environment"
        );
    }
    fixture.missing_helper_refuses_without_fallback();
}
