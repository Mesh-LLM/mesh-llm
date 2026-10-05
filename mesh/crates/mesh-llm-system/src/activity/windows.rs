use super::{HostActivity, HostActivityDetector, PriorityController, PriorityFailure};

#[derive(Debug, Default)]
pub struct WindowsHostActivityDetector;

impl HostActivityDetector for WindowsHostActivityDetector {
    fn sample(&mut self) -> HostActivity {
        HostActivity::Unknown
    }
}

#[derive(Debug, Default)]
pub struct WindowsPriorityController;

impl PriorityController for WindowsPriorityController {
    fn reduce_priority(&mut self) -> Result<(), PriorityFailure> {
        Err(PriorityFailure::Unsupported)
    }

    fn restore_priority(&mut self) -> Result<(), PriorityFailure> {
        Err(PriorityFailure::Unsupported)
    }
}
