//! Detected GPU backends that the selected runtime leaves unused.
//!
//! When no runtime for a detected GPU fits the host (a CUDA 13 runtime on a
//! CUDA 12 driver, say), selection falls back to a lower-ranked backend such
//! as the CPU. That is the right choice, but the node then runs slower than
//! the host allows; this module names the GPU backends left behind and why
//! their runtimes were rejected, so callers can say so.

use crate::{
    CandidateEvaluation, HostRuntimeProfile, NativeRuntimeArtifact, NativeRuntimeBackendKind,
    RuntimeSelection,
};
use std::fmt;

/// A detected GPU backend that ranks above the selected runtime's backend,
/// with the runtimes of that backend for this host and why each was rejected.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnusedGpuBackend {
    pub backend: NativeRuntimeBackendKind,
    pub rejected: Vec<CandidateEvaluation>,
    /// Runtimes of this backend that startup skips because they lack the
    /// compatibility metadata it requires (`has_startup_compatibility_metadata`).
    pub missing_metadata: Vec<NativeRuntimeArtifact>,
}

/// The detected GPU backends that rank above the `selected` runtime's backend
/// and have no runtime that fits this host.
///
/// Only the recommended selection is checked: an explicit backend or runtime
/// id is the user's choice. A backend with a fitting runtime that was not
/// selected (outranked, or it failed to load) is not reported: its runtimes
/// do fit, so this has nothing to explain. `evaluated` holds the candidates the selection
/// considered and `missing_metadata` the local runtimes it skipped for lack of
/// startup compatibility metadata; those built for another OS or architecture
/// are left out of the reasons.
pub fn unused_gpu_backends(
    evaluated: &[CandidateEvaluation],
    missing_metadata: &[NativeRuntimeArtifact],
    profile: &HostRuntimeProfile,
    selection: &RuntimeSelection,
    selected: &NativeRuntimeArtifact,
) -> Vec<UnusedGpuBackend> {
    if *selection != RuntimeSelection::Recommended {
        return Vec::new();
    }
    let selected_rank = selected.backend.kind.default_rank();
    let for_this_host = |artifact: &NativeRuntimeArtifact, kind: &NativeRuntimeBackendKind| {
        artifact.backend.kind == *kind
            && artifact.platform.os == profile.os
            && artifact.platform.arch == profile.arch
    };
    profile
        .available_flavors
        .iter()
        .filter(|kind| **kind != NativeRuntimeBackendKind::Cpu)
        .filter(|kind| kind.default_rank() > selected_rank)
        .filter(|kind| {
            !evaluated
                .iter()
                .any(|candidate| candidate.compatible && for_this_host(&candidate.artifact, kind))
        })
        .map(|kind| UnusedGpuBackend {
            backend: kind.clone(),
            rejected: evaluated
                .iter()
                .filter(|candidate| for_this_host(&candidate.artifact, kind))
                .cloned()
                .collect(),
            missing_metadata: missing_metadata
                .iter()
                .filter(|artifact| for_this_host(artifact, kind))
                .cloned()
                .collect(),
        })
        .collect()
}

impl UnusedGpuBackend {
    /// Why this backend is unused, naming the backend, for a line that covers
    /// several backends.
    fn reason(&self) -> String {
        if self.rejected.is_empty() && self.missing_metadata.is_empty() {
            return format!("no {} runtime is available", self.backend);
        }
        format!(
            "{} runtimes don't fit:{}",
            self.backend,
            self.rejected_runtimes()
        )
    }

    fn rejected_runtimes(&self) -> String {
        let mut text = String::new();
        for candidate in &self.rejected {
            let reasons = candidate
                .rejection_reasons
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; ");
            text.push_str(&format!(" {} ({reasons})", candidate.artifact.id));
        }
        for artifact in &self.missing_metadata {
            text.push_str(&format!(
                " {} (no minimum glibc in its manifest, which startup requires)",
                artifact.id
            ));
        }
        text
    }
}

/// One line for all unused `backends`. A single GPU often offers several
/// backends (an AMD card on Windows: ROCm and Vulkan), so a line per backend
/// would read as several GPUs.
pub fn describe_unused_gpu_backends(backends: &[UnusedGpuBackend]) -> Option<String> {
    match backends {
        [] => None,
        [backend] => Some(backend.to_string()),
        several => {
            let kinds = several
                .iter()
                .map(|backend| backend.backend.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let reasons = several
                .iter()
                .map(UnusedGpuBackend::reason)
                .collect::<Vec<_>>()
                .join("; ");
            Some(format!(
                "GPU detected ({kinds}), but no runtime for it fits: {reasons}"
            ))
        }
    }
}

impl fmt::Display for UnusedGpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.rejected.is_empty() && self.missing_metadata.is_empty() {
            return write!(
                f,
                "{} GPU detected, but no {} runtime is available",
                self.backend, self.backend
            );
        }
        write!(
            f,
            "{} GPU detected, but its runtimes don't fit:{}",
            self.backend,
            self.rejected_runtimes()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CudaRuntimeRequirements, HostCudaProfile, NativeRuntimeBackend, NativeRuntimePlatform,
        evaluate_native_runtime_artifact,
    };
    use std::collections::BTreeSet;

    fn artifact(id: &str, backend: NativeRuntimeBackend, arch: &str) -> NativeRuntimeArtifact {
        NativeRuntimeArtifact {
            id: id.to_string(),
            release_version: Some("0.76.1".to_string()),
            skippy_abi: "0.1.71".to_string(),
            platform: NativeRuntimePlatform {
                os: "linux".to_string(),
                arch: arch.to_string(),
                target: None,
                min_glibc: None,
            },
            backend,
            rank: 0,
            libraries: vec!["lib/libllama.so".to_string()],
            files: Default::default(),
            tools: Default::default(),
            url: None,
            sha256: None,
            signature: None,
        }
    }

    fn cuda(toolkit_major: u32) -> NativeRuntimeBackend {
        NativeRuntimeBackend {
            kind: NativeRuntimeBackendKind::Cuda,
            cuda: Some(CudaRuntimeRequirements {
                toolkit_major,
                min_driver: None,
                gpu_arches: Vec::new(),
            }),
            rocm: None,
            vulkan: None,
        }
    }

    /// An aarch64 host whose driver supports CUDA 12, like a Jetson on JetPack 6.
    fn cuda12_host() -> HostRuntimeProfile {
        HostRuntimeProfile {
            os: "linux".to_string(),
            arch: "aarch64".to_string(),
            target_triple: None,
            glibc_version: None,
            available_flavors: BTreeSet::from([
                NativeRuntimeBackendKind::Cpu,
                NativeRuntimeBackendKind::Cuda,
            ]),
            gpus: Vec::new(),
            cuda: Some(HostCudaProfile {
                driver_max_major: Some(12),
                ..HostCudaProfile::default()
            }),
            rocm: None,
            vulkan: None,
        }
    }

    fn evaluate(
        artifacts: &[NativeRuntimeArtifact],
        profile: &HostRuntimeProfile,
    ) -> Vec<CandidateEvaluation> {
        artifacts
            .iter()
            .map(|artifact| {
                evaluate_native_runtime_artifact(
                    artifact,
                    profile,
                    "0.76.1",
                    Some("0.1.71"),
                    &RuntimeSelection::Recommended,
                )
            })
            .collect()
    }

    #[test]
    fn a_cpu_fallback_names_the_rejected_cuda_runtime_and_its_reason() {
        let profile = cuda12_host();
        let cpu = artifact("cpu", NativeRuntimeBackend::cpu(), "aarch64");
        let artifacts = [
            cpu.clone(),
            artifact("cuda13", cuda(13), "aarch64"),
            artifact("cuda13-x86", cuda(13), "x86_64"),
        ];
        let evaluated = evaluate(&artifacts, &profile);

        let unused = unused_gpu_backends(
            &evaluated,
            &[],
            &profile,
            &RuntimeSelection::Recommended,
            &cpu,
        );

        assert_eq!(unused.len(), 1);
        assert_eq!(unused[0].backend, NativeRuntimeBackendKind::Cuda);
        let ids = unused[0]
            .rejected
            .iter()
            .map(|candidate| candidate.artifact.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ids, ["cuda13"]);
        let message = unused[0].to_string();
        assert!(message.starts_with("cuda GPU detected, but its runtimes don't fit: cuda13 ("));
        assert!(message.contains("CUDA driver too old"), "{message}");
    }

    #[test]
    fn a_cpu_fallback_without_any_cuda_runtime_says_so() {
        let profile = cuda12_host();
        let cpu = artifact("cpu", NativeRuntimeBackend::cpu(), "aarch64");
        let evaluated = evaluate(std::slice::from_ref(&cpu), &profile);

        let unused = unused_gpu_backends(
            &evaluated,
            &[],
            &profile,
            &RuntimeSelection::Recommended,
            &cpu,
        );

        assert_eq!(
            unused.iter().map(ToString::to_string).collect::<Vec<_>>(),
            ["cuda GPU detected, but no cuda runtime is available"]
        );
    }

    #[test]
    fn one_gpu_with_two_unused_backends_gets_one_line() {
        // An AMD card on Windows offers ROCm and Vulkan; with only the CPU
        // runtime that is one GPU, not two.
        let profile = HostRuntimeProfile {
            os: "windows".to_string(),
            arch: "x86_64".to_string(),
            target_triple: None,
            glibc_version: None,
            available_flavors: BTreeSet::from([
                NativeRuntimeBackendKind::Cpu,
                NativeRuntimeBackendKind::Rocm,
                NativeRuntimeBackendKind::Vulkan,
            ]),
            gpus: Vec::new(),
            cuda: None,
            rocm: None,
            vulkan: None,
        };
        let mut cpu = artifact("cpu", NativeRuntimeBackend::cpu(), "x86_64");
        cpu.platform.os = "windows".to_string();
        let evaluated = evaluate(std::slice::from_ref(&cpu), &profile);

        let unused = unused_gpu_backends(
            &evaluated,
            &[],
            &profile,
            &RuntimeSelection::Recommended,
            &cpu,
        );

        assert_eq!(unused.len(), 2);
        assert_eq!(
            describe_unused_gpu_backends(&unused).as_deref(),
            Some(
                "GPU detected (rocm, vulkan), but no runtime for it fits: \
                 no rocm runtime is available; no vulkan runtime is available"
            )
        );
    }

    #[test]
    fn a_single_unused_backend_is_described_as_before() {
        let profile = cuda12_host();
        let cpu = artifact("cpu", NativeRuntimeBackend::cpu(), "aarch64");
        let evaluated = evaluate(std::slice::from_ref(&cpu), &profile);
        let unused = unused_gpu_backends(
            &evaluated,
            &[],
            &profile,
            &RuntimeSelection::Recommended,
            &cpu,
        );

        assert_eq!(
            describe_unused_gpu_backends(&unused).as_deref(),
            Some("cuda GPU detected, but no cuda runtime is available")
        );
        assert_eq!(describe_unused_gpu_backends(&[]), None);
    }

    #[test]
    fn nothing_is_unused_when_the_gpu_runtime_is_selected() {
        let profile = cuda12_host();
        let cuda12 = artifact("cuda12", cuda(12), "aarch64");
        let artifacts = [
            artifact("cpu", NativeRuntimeBackend::cpu(), "aarch64"),
            cuda12.clone(),
        ];
        let evaluated = evaluate(&artifacts, &profile);

        assert!(
            unused_gpu_backends(
                &evaluated,
                &[],
                &profile,
                &RuntimeSelection::Recommended,
                &cuda12
            )
            .is_empty()
        );
    }

    #[test]
    fn a_runtime_skipped_for_missing_metadata_is_named_with_the_reason() {
        let profile = cuda12_host();
        let cpu = artifact("cpu", NativeRuntimeBackend::cpu(), "aarch64");
        let evaluated = evaluate(std::slice::from_ref(&cpu), &profile);
        let legacy_cuda = artifact("cuda12-legacy", cuda(12), "aarch64");

        let unused = unused_gpu_backends(
            &evaluated,
            &[legacy_cuda],
            &profile,
            &RuntimeSelection::Recommended,
            &cpu,
        );

        assert_eq!(
            unused.iter().map(ToString::to_string).collect::<Vec<_>>(),
            [
                "cuda GPU detected, but its runtimes don't fit: cuda12-legacy (no minimum glibc in its manifest, which startup requires)"
            ]
        );
    }

    #[test]
    fn a_fitting_gpu_runtime_that_was_not_selected_is_not_reported() {
        let mut profile = cuda12_host();
        if let Some(cuda) = profile.cuda.as_mut() {
            cuda.toolkit_majors = BTreeSet::from([12]);
        }
        let cpu = artifact("cpu", NativeRuntimeBackend::cpu(), "aarch64");
        let artifacts = [cpu.clone(), artifact("cuda12", cuda(12), "aarch64")];
        let evaluated = evaluate(&artifacts, &profile);
        assert!(evaluated.iter().all(|candidate| candidate.compatible));

        assert!(
            unused_gpu_backends(
                &evaluated,
                &[],
                &profile,
                &RuntimeSelection::Recommended,
                &cpu
            )
            .is_empty()
        );
    }

    #[test]
    fn an_explicit_cpu_selection_is_not_reported() {
        let profile = cuda12_host();
        let cpu = artifact("cpu", NativeRuntimeBackend::cpu(), "aarch64");
        let selection = RuntimeSelection::parse(Some("cpu")).unwrap();

        assert!(unused_gpu_backends(&[], &[], &profile, &selection, &cpu).is_empty());
    }
}
