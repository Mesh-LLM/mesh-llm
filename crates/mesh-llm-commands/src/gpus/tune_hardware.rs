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
