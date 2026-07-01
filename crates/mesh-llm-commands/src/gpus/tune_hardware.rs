pub(crate) mod evaluate {
    include!("tune_hardware/evaluate.rs");
}

pub(crate) mod device_request {
    include!("tune_hardware/device_request.rs");
}

pub(crate) mod mlock {
    include!("tune_hardware/mlock.rs");
}

pub(crate) mod types {
    include!("tune_hardware/types.rs");
}

pub(crate) fn dispatch_symbol_anchor() {
    let _ = evaluate::hardware_symbol_anchor as fn();
    let _ = types::TuneHardwareEvaluation::device_field_status
        as fn(&types::TuneHardwareEvaluation) -> crate::gpus::tune::TuneFieldStatus;
    let _ = types::TuneHardwareEvaluation::mlock_field_status
        as fn(&types::TuneHardwareEvaluation) -> crate::gpus::tune::TuneFieldStatus;
    let _ = types::TuneHardwareEvaluation::diagnostics
        as fn(&types::TuneHardwareEvaluation) -> Vec<crate::gpus::tune::TuneDiagnostic>;
    let _ = types::TuneHardwareEvaluation::recommended_device_value
        as fn(&types::TuneHardwareEvaluation) -> String;
    let _ = mlock::TuneMlockProbe::Supported {
        limit: mlock::TuneMlockLimit::Unlimited,
    };
    let _ = mlock::TuneMlockProbe::Unsupported {
        reason: String::new(),
    };
    let _ = mlock::TuneMlockLimit::Bytes(0);
}

#[cfg(test)]
mod tests {
    mod helpers {
        include!("tune_hardware/tests/helpers.rs");
    }

    mod mlock_reporting {
        include!("tune_hardware/tests/mlock_reporting.rs");
    }

    mod selection {
        include!("tune_hardware/tests/selection.rs");
    }
}
