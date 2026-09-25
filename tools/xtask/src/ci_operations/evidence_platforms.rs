//! Per-architecture children of a bound index candidate: every child must
//! have one platform candidate, identity receipt and runtime identity in
//! the cohort, all bound to the same producer source.

use crate::ci_operations::evidence_binding::{CandidateBinding, field};
use crate::ci_operations::evidence_input::{IMAGE, digest, fields, require};
use crate::ci_operations::python_access::{Outcome, eq, is_int_one, is_str, item, object, string};
use crate::ci_plan::document::Json;
use crate::prepared_input::python_value::display;

impl CandidateBinding<'_> {
    pub(crate) fn check_children(&self) -> Outcome<()> {
        let children = field(self.candidate, "children");
        let list = children
            .as_array()
            .filter(|list| !list.is_empty() && list.len() <= 2);
        require(list.is_some(), "invalid index children")?;
        let list = list.unwrap_or_default();
        let mut seen: Vec<String> = Vec::new();
        for child in list {
            fields(child, "os architecture digest")?;
            let architecture = field(child, "architecture");
            let known = ["amd64", "arm64"]
                .iter()
                .any(|arch| is_str(architecture, arch));
            let arch = architecture.as_str().unwrap_or_default().to_owned();
            require(
                is_str(field(child, "os"), "linux") && known && !seen.contains(&arch),
                "invalid or duplicate child platform",
            )?;
            seen.push(arch.clone());
            digest(field(child, "digest"))?;
            self.check_platform(child, &arch)?;
        }
        let environment = display(Some(field(self.candidate, "environment")));
        let id = display(Some(field(self.family(), "id")));
        let prefix = format!("{environment}-{id}-");
        let mut declared: Vec<&str> = field(self.cohort, "platforms")
            .as_object()
            .unwrap_or_default()
            .iter()
            .map(|(key, _)| key.as_str())
            .filter(|key| key.starts_with(&prefix))
            .collect();
        let mut expected: Vec<String> = seen.iter().map(|arch| format!("{prefix}{arch}")).collect();
        declared.sort_unstable();
        expected.sort_unstable();
        require(
            declared == expected,
            "index child set differs from cohort family platforms",
        )?;
        let mut digests: Vec<String> = list
            .iter()
            .map(|child| display(Some(field(child, "digest"))))
            .collect();
        digests.sort_unstable();
        digests.dedup();
        require(digests.len() == list.len(), "duplicate child digest")
    }

    fn check_platform(&self, child: &Json, arch: &str) -> Outcome<()> {
        let (candidate, family) = (self.candidate, self.family());
        let environment = display(Some(field(candidate, "environment")));
        let id = display(Some(field(family, "id")));
        let entry = field(
            field(self.cohort, "platforms"),
            &format!("{environment}-{id}-{arch}"),
        );
        fields(entry, "candidate receipt index_base64 manifest_base64")?;
        let platform = object(&[("os", string("linux")), ("architecture", string(arch))]);
        let platform_candidate = field(entry, "candidate");
        fields(
            platform_candidate,
            "schema type image environment backend mesh_revision runner_images_revision platform digest child_digest",
        )?;
        let at = |key: &str| field(platform_candidate, key);
        require(
            is_int_one(at("schema"))
                && is_str(at("type"), "mesh-llm-runner-image-platform-candidate"),
            "invalid platform candidate",
        )?;
        require(
            eq(at("platform"), &platform)
                && eq(at("backend"), family)
                && eq(at("environment"), field(candidate, "environment"))
                && is_str(at("image"), IMAGE)
                && eq(at("child_digest"), field(child, "digest")),
            "platform candidate mismatch",
        )?;
        let receipt = field(entry, "receipt");
        fields(
            receipt,
            "schema type image platform backend_id oci runtime layers scope",
        )?;
        let got = |key: &str| field(receipt, key);
        require(
            is_int_one(got("schema"))
                && is_str(got("type"), "mesh-llm-runner-image-identity")
                && is_str(got("image"), IMAGE)
                && eq(got("platform"), &platform)
                && eq(got("backend_id"), field(family, "id")),
            "platform receipt mismatch",
        )?;
        let oci = got("oci");
        let bound = eq(item(item(oci, "root")?, "digest")?, at("digest"))
            && eq(
                item(item(oci, "manifest")?, "digest")?,
                field(child, "digest"),
            );
        require(bound, "OCI digest binding mismatch")?;
        self.check_runtime(got("runtime"), &platform, platform_candidate)
    }

    fn check_runtime(
        &self,
        runtime: &Json,
        platform: &Json,
        platform_candidate: &Json,
    ) -> Outcome<()> {
        fields(
            runtime,
            "schema type platform family source verification expected_tools tools dependencies cache",
        )?;
        let at = |key: &str| field(runtime, key);
        require(
            is_int_one(at("schema"))
                && is_str(at("type"), "mesh-llm-runner-runtime-identity")
                && eq(at("platform"), platform),
            "runtime platform mismatch",
        )?;
        let family = self.family();
        let expected_family = object(&[
            ("environment", field(self.candidate, "environment").clone()),
            ("backend", field(family, "name").clone()),
            ("cuda_series", field(family, "cuda_series").clone()),
            ("rocm_version", field(family, "rocm_version").clone()),
        ]);
        require(
            eq(at("family"), &expected_family),
            "runtime family mismatch",
        )?;
        let mesh = item(self.proof_origin, "mesh_revision")?;
        let images = item(self.proof_origin, "runner_images_revision")?;
        let source = object(&[
            ("mesh_revision", mesh.clone()),
            ("runner_images_revision", images.clone()),
        ]);
        require(
            eq(at("source"), &source)
                && eq(field(platform_candidate, "mesh_revision"), mesh)
                && eq(field(platform_candidate, "runner_images_revision"), images),
            "runtime source mismatch",
        )?;
        let verification = at("verification");
        require(
            eq(item(verification, "verifier_revision")?, images),
            "verifier revision mismatch",
        )?;
        for key in ["tool_pins_sha256", "cache_policy_sha256"] {
            digest(item(verification, key)?)?;
        }
        for key in ["expected_tools", "tools", "dependencies", "cache"] {
            require(
                at(key).as_object().is_some(),
                "missing producer tool/cache observations",
            )?;
        }
        Ok(())
    }
}
