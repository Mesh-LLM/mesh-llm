use super::{HostActivity, HostActivityDetector, PriorityController, PriorityFailure};

#[derive(Debug, Default)]
pub struct UnsupportedHostActivityDetector;

impl HostActivityDetector for UnsupportedHostActivityDetector {
    fn sample(&mut self) -> HostActivity {
        HostActivity::Unknown
    }
}

#[derive(Debug, Default)]
pub struct UnsupportedPriorityController;

impl PriorityController for UnsupportedPriorityController {
    fn reduce_priority(&mut self) -> Result<(), PriorityFailure> {
        Err(PriorityFailure::Unsupported)
    }

    fn restore_priority(&mut self) -> Result<(), PriorityFailure> {
        Err(PriorityFailure::Unsupported)
    }
}
