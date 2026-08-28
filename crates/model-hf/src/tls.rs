//! CPU-safe TLS provider selection for Hugging Face clients.

#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
use std::mem::size_of;

/// The provider used for process-default rustls clients after configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HfTlsProvider {
    /// Let the Hugging Face HTTP client select its normal provider.
    Automatic,
    /// Keep a provider that another application component installed first.
    Existing,
    /// Keep an existing provider that cannot be verified as safe on this CPU.
    ExistingUnverified,
    /// Install rustls' runtime-dispatched ring provider.
    Ring,
}

/// Configure a CPU-safe process default before constructing Hugging Face clients.
///
/// The published `mesh-llm-hf-hub` and `hf-xet` clients use reqwest 0.13's
/// rustls backend. When no process provider is installed, reqwest selects its
/// AWS-LC provider. AWS-LC's AArch64 SHA-512 path has caused illegal
/// instructions on CPUs that do not advertise FEAT_SHA512. On those CPUs we
/// install ring, whose SHA-512 implementation performs its own runtime
/// capability check. All other targets retain reqwest's normal provider
/// selection, and an already-installed provider is never replaced.
pub fn configure_hf_tls_provider() -> HfTlsProvider {
    configure_hf_tls_provider_for(arm_sha512_available())
}

fn configure_hf_tls_provider_for(has_arm_sha512: bool) -> HfTlsProvider {
    let action = provider_action(
        has_arm_sha512,
        rustls::crypto::CryptoProvider::get_default().is_some(),
    );

    match action {
        HfTlsProvider::Automatic | HfTlsProvider::Existing => action,
        HfTlsProvider::ExistingUnverified => {
            warn_about_unverified_provider("was installed before Hugging Face initialization");
            action
        }
        HfTlsProvider::Ring => match rustls::crypto::ring::default_provider().install_default() {
            Ok(()) => HfTlsProvider::Ring,
            Err(_) => {
                warn_about_unverified_provider("won a race with the ring fallback");
                HfTlsProvider::ExistingUnverified
            }
        },
    }
}

fn warn_about_unverified_provider(reason: &str) {
    tracing::warn!(
        target: "model_hf::tls",
        %reason,
        "an existing rustls provider cannot be verified as safe on AArch64 without SHA-512; Hugging Face/Xet TLS may terminate with SIGILL"
    );
}

fn provider_action(has_arm_sha512: bool, has_existing_provider: bool) -> HfTlsProvider {
    if has_existing_provider {
        if has_arm_sha512 {
            HfTlsProvider::Existing
        } else {
            HfTlsProvider::ExistingUnverified
        }
    } else if has_arm_sha512 {
        HfTlsProvider::Automatic
    } else {
        HfTlsProvider::Ring
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn arm_sha512_available() -> bool {
    true
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "android")
))]
fn arm_sha512_available() -> bool {
    // Linux and Android expose the ARM feature set through AT_HWCAP. Treat a
    // missing/zero value as unsupported so the fallback remains portable.
    // SAFETY: getauxval is a read-only libc query with no pointer arguments.
    let hwcap = unsafe { libc::getauxval(libc::AT_HWCAP) };
    hwcap & libc::HWCAP_SHA512 != 0
}

#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
fn arm_sha512_available() -> bool {
    let name = b"hw.optional.armv8_2_sha512\0";
    let mut value: libc::c_int = 0;
    let mut length = size_of::<libc::c_int>();
    // SAFETY: all pointers reference valid, writable values for the duration
    // of the syscall; the name is NUL-terminated and has static storage.
    let result = unsafe {
        libc::sysctlbyname(
            name.as_ptr().cast(),
            (&mut value as *mut libc::c_int).cast(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    };
    result == 0 && length == size_of::<libc::c_int>() && value != 0
}

#[cfg(all(
    target_arch = "aarch64",
    not(any(target_os = "linux", target_os = "android", target_vendor = "apple"))
))]
fn arm_sha512_available() -> bool {
    // Unknown AArch64 platforms get the conservative software-dispatched
    // provider. ring has a portable fallback for these targets.
    false
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    use hf_hub::HFClientBuilder;
    use serial_test::serial;

    use super::{
        HfTlsProvider, configure_hf_tls_provider, configure_hf_tls_provider_for, provider_action,
    };

    #[test]
    fn leaves_provider_selection_unchanged_when_sha512_is_supported() {
        assert_eq!(provider_action(true, false), HfTlsProvider::Automatic);
    }

    #[test]
    fn never_replaces_an_existing_provider() {
        assert_eq!(provider_action(true, true), HfTlsProvider::Existing);
    }

    #[test]
    fn keeps_an_existing_provider_on_unsupported_arm() {
        assert_eq!(
            provider_action(false, true),
            HfTlsProvider::ExistingUnverified
        );
    }

    #[test]
    fn selects_ring_when_unsupported_arm_has_no_provider() {
        assert_eq!(provider_action(false, false), HfTlsProvider::Ring);
    }

    #[test]
    #[serial]
    fn real_hf_client_builds_after_provider_initialization() {
        let selected = configure_hf_tls_provider();
        assert!(matches!(
            selected,
            HfTlsProvider::Automatic
                | HfTlsProvider::Existing
                | HfTlsProvider::ExistingUnverified
                | HfTlsProvider::Ring
        ));

        let client = HFClientBuilder::new()
            .endpoint("https://huggingface.co")
            .cache_enabled(false)
            .build();
        assert!(
            client.is_ok(),
            "HFClientBuilder failed after selecting {selected:?}: {client:?}"
        );
    }

    #[test]
    #[serial]
    fn forced_ring_provider_builds_a_real_hf_client() {
        const CHILD_ENV: &str = "MESH_LLM_TEST_HF_TLS_RING_CHILD";

        if std::env::var_os(CHILD_ENV).is_some() {
            assert_eq!(configure_hf_tls_provider_for(false), HfTlsProvider::Ring);
            assert!(rustls::crypto::CryptoProvider::get_default().is_some());
            assert!(
                HFClientBuilder::new()
                    .endpoint("https://huggingface.co")
                    .cache_enabled(false)
                    .build()
                    .is_ok()
            );
            return;
        }

        let status = Command::new(std::env::current_exe().expect("test executable"))
            .args([
                "--exact",
                "tls::tests::forced_ring_provider_builds_a_real_hf_client",
                "--nocapture",
            ])
            .env(CHILD_ENV, "1")
            .status()
            .expect("spawn isolated ring-provider test");
        assert!(status.success(), "isolated ring-provider test failed");
    }
}
