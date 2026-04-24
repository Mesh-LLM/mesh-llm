//! User-presence detection for yield-on-presence mode.
//!
//! Answers one question on a tick: "Is the user at this machine active right now?"
//!
//! Design notes:
//! - macOS and Windows have real implementations that query input-idle time via
//!   OS APIs. They do not require elevated privileges and do not hook input.
//! - Linux and everything else return `Unknown`. Dedicated Linux servers are
//!   exactly the nodes you want serving 24/7, so we deliberately do not try to
//!   guess desktop-session state.
//! - `Unknown` is a fail-open signal: the yield controller treats it as "keep
//!   serving" so a broken probe never takes a node offline silently.
//!
//! All platform implementations are in-crate so we can stay in the single
//! presence module and avoid leaking platform crates into the wider build.

use std::time::Duration;

/// Derived presence state consumed by the yield controller.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Presence {
    /// User has been interacting with the machine recently.
    Active,
    /// No recent input — user appears to be away from the keyboard.
    Away,
    /// Probe failed or platform is not supported. Treated as "keep serving".
    Unknown,
}

/// A synchronous presence probe. Implementations must be cheap — we call
/// `sample` on a timer from a normal tokio task.
pub trait PresenceProbe: Send + Sync + 'static {
    /// Return the current presence state. `idle_threshold` is the number of
    /// seconds of input-idle time after which we consider the user `Away`.
    fn sample(&self, idle_threshold: Duration) -> Presence;
}

/// Build the default probe for this platform. Returns a probe that always
/// reports `Unknown` on platforms without a real implementation.
pub fn default_probe() -> Box<dyn PresenceProbe> {
    #[cfg(target_os = "macos")]
    {
        return Box::new(macos::MacProbe::new());
    }
    #[cfg(target_os = "windows")]
    {
        return Box::new(windows::WinProbe::new());
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        Box::new(UnknownProbe)
    }
}

/// Probe that always reports `Unknown`. Used on unsupported platforms and in
/// tests.
#[allow(dead_code)] // constructed via default_probe() on non-macOS/Windows
pub struct UnknownProbe;

impl PresenceProbe for UnknownProbe {
    fn sample(&self, _idle_threshold: Duration) -> Presence {
        Presence::Unknown
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use super::{Presence, PresenceProbe};
    use std::time::Duration;

    // `CGEventSourceSecondsSinceLastEventType(kCGEventSourceStateHIDSystemState,
    // kCGAnyInputEventType)` returns the seconds since the last HID input event
    // (keyboard, mouse, trackpad) observed by the window server. It is the
    // same value `ioreg -c IOHIDSystem` reports as `HIDIdleTime`. No entitlement
    // or accessibility permission is required.
    //
    // Signature:
    //   CFTimeInterval CGEventSourceSecondsSinceLastEventType(
    //       CGEventSourceStateID source_state,
    //       CGEventType event_type,
    //   );
    //
    // We link directly rather than pulling in core-graphics, which brings a
    // lot of transitive surface for a single function.
    #[link(name = "ApplicationServices", kind = "framework")]
    extern "C" {
        fn CGEventSourceSecondsSinceLastEventType(
            source_state: u32,
            event_type: u32,
        ) -> std::os::raw::c_double;
    }

    // kCGEventSourceStateHIDSystemState = 1 (combined HID state).
    const HID_SYSTEM_STATE: u32 = 1;
    // kCGAnyInputEventType = u32::MAX (`~0`) — matches any input event type.
    const ANY_INPUT_EVENT: u32 = !0u32;

    pub struct MacProbe;

    impl MacProbe {
        pub fn new() -> Self {
            Self
        }

        fn idle_seconds(&self) -> Option<f64> {
            // SAFETY: calling a pure C function that reads no caller state and
            // returns a CFTimeInterval (double). No threading requirements.
            let secs = unsafe {
                CGEventSourceSecondsSinceLastEventType(HID_SYSTEM_STATE, ANY_INPUT_EVENT)
            };
            if secs.is_finite() && secs >= 0.0 {
                Some(secs)
            } else {
                None
            }
        }
    }

    impl PresenceProbe for MacProbe {
        fn sample(&self, idle_threshold: Duration) -> Presence {
            match self.idle_seconds() {
                Some(secs) if secs >= idle_threshold.as_secs_f64() => Presence::Away,
                Some(_) => Presence::Active,
                None => Presence::Unknown,
            }
        }
    }
}

#[cfg(target_os = "windows")]
mod windows {
    use super::{Presence, PresenceProbe};
    use std::time::Duration;

    // `GetLastInputInfo` fills a LASTINPUTINFO struct with the tick count of
    // the last input event. Subtract from GetTickCount() to get idle ms.
    //
    // This is the same signal the screensaver and Windows itself use to
    // detect idle. No privileges required.
    #[repr(C)]
    struct LastInputInfo {
        cb_size: u32,
        dw_time: u32,
    }

    #[link(name = "user32")]
    extern "system" {
        fn GetLastInputInfo(plii: *mut LastInputInfo) -> i32;
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetTickCount() -> u32;
    }

    pub struct WinProbe;

    impl WinProbe {
        pub fn new() -> Self {
            Self
        }

        fn idle_seconds(&self) -> Option<f64> {
            let mut info = LastInputInfo {
                cb_size: std::mem::size_of::<LastInputInfo>() as u32,
                dw_time: 0,
            };
            // SAFETY: `info` is a valid `LastInputInfo` with `cb_size` set.
            // `GetLastInputInfo` writes only to `info` and returns BOOL.
            // `GetTickCount` is a pure read.
            let ok = unsafe { GetLastInputInfo(&mut info as *mut _) };
            if ok == 0 {
                return None;
            }
            let now = unsafe { GetTickCount() };
            // Both are DWORD tick counts and wrap at ~49.7 days. Use
            // wrapping_sub so the result stays correct across wrap.
            let idle_ms = now.wrapping_sub(info.dw_time);
            Some(idle_ms as f64 / 1000.0)
        }
    }

    impl PresenceProbe for WinProbe {
        fn sample(&self, idle_threshold: Duration) -> Presence {
            match self.idle_seconds() {
                Some(secs) if secs >= idle_threshold.as_secs_f64() => Presence::Away,
                Some(_) => Presence::Active,
                None => Presence::Unknown,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_probe_always_returns_unknown() {
        let probe = UnknownProbe;
        assert_eq!(probe.sample(Duration::from_secs(60)), Presence::Unknown);
    }

    #[test]
    fn default_probe_on_unsupported_platform_is_unknown() {
        // On linux CI this exercises the real default_probe path.
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        {
            let probe = default_probe();
            assert_eq!(probe.sample(Duration::from_secs(60)), Presence::Unknown);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mac_probe_returns_finite_idle_or_unknown() {
        // We can't assert Active vs Away deterministically in CI, but a sane
        // probe should not panic and should return one of the three states.
        let probe = macos::MacProbe::new();
        let s = probe.sample(Duration::from_secs(120));
        assert!(matches!(
            s,
            Presence::Active | Presence::Away | Presence::Unknown
        ));
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn win_probe_returns_finite_idle_or_unknown() {
        let probe = windows::WinProbe::new();
        let s = probe.sample(Duration::from_secs(120));
        assert!(matches!(
            s,
            Presence::Active | Presence::Away | Presence::Unknown
        ));
    }
}
