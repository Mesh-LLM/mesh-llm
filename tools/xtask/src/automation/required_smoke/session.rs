use super::evidence::{self, Evidence};
use super::{Attestation, Budget, Check, Rejection, Transfer, Variant};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Progress {
    Pending,
    Advanced(Check),
    Complete,
}

enum State {
    Checking(Check),
    Complete,
    Failed(Rejection),
}

pub(crate) struct Session {
    pub(super) variant: Variant,
    attestation: Attestation,
    budget: Budget,
    state: State,
    stage_started: Duration,
    last_observed: Duration,
    model: Option<String>,
    headless_launched: bool,
}

impl Session {
    pub(crate) fn new(variant: Variant, attestation: Attestation, budget: Budget) -> Self {
        let check = match &attestation {
            Attestation::Disabled => Check::Runtime,
            Attestation::Required { .. } => Check::InspectAttestation,
        };
        Self {
            variant,
            attestation,
            budget,
            state: State::Checking(check),
            stage_started: Duration::ZERO,
            last_observed: Duration::ZERO,
            model: None,
            headless_launched: false,
        }
    }

    pub(crate) fn expected(&self) -> Result<Option<Check>, Rejection> {
        match self.state {
            State::Checking(check) => Ok(Some(check)),
            State::Complete => Ok(None),
            State::Failed(reason) => Err(reason),
        }
    }

    pub(crate) fn model_id(&self) -> Option<&str> {
        self.model.as_deref()
    }

    pub(super) fn headless_launched(&mut self, elapsed: Duration) -> Result<(), Rejection> {
        if self.headless_launched || self.expected()? != Some(Check::HeadlessModels) {
            return self.fail(Rejection::OutOfOrder);
        }
        if elapsed < self.last_observed {
            return self.fail(Rejection::ClockRegression);
        }
        self.headless_launched = true;
        self.stage_started = elapsed;
        self.last_observed = elapsed;
        Ok(())
    }

    pub(super) fn remaining(&self, elapsed: Duration) -> Result<Duration, Rejection> {
        let check = self.expected()?.ok_or(Rejection::Incomplete)?;
        Ok(self
            .budget
            .for_check(check)
            .saturating_sub(elapsed.saturating_sub(self.stage_started)))
    }

    pub(crate) fn tick(&mut self, elapsed: Duration) -> Result<Progress, Rejection> {
        match self.state {
            State::Failed(reason) => return Err(reason),
            State::Checking(_) | State::Complete => (),
        }
        if elapsed < self.last_observed {
            return self.fail(Rejection::ClockRegression);
        }
        self.last_observed = elapsed;
        match self.state {
            State::Checking(check) => {
                if elapsed.saturating_sub(self.stage_started) >= self.budget.for_check(check) {
                    return self.fail(Rejection::Deadline(check));
                }
                Ok(Progress::Pending)
            }
            State::Complete => Ok(Progress::Complete),
            State::Failed(reason) => Err(reason),
        }
    }

    pub(crate) fn observe(
        &mut self,
        observation: (Check, Transfer<'_>),
        elapsed: Duration,
    ) -> Result<Progress, Rejection> {
        self.tick(elapsed)?;
        let (check, transfer) = observation;
        if self.expected()? != Some(check) {
            return self.fail(Rejection::OutOfOrder);
        }
        match evidence::classify(check, transfer, &self.attestation) {
            Err(reason) => self.fail(reason),
            Ok(Evidence::Pending) => {
                match check {
                    Check::HeadlessStatus => self.state = State::Checking(Check::HeadlessModels),
                    Check::InspectAttestation
                    | Check::Runtime
                    | Check::Models
                    | Check::RuntimeAttestation
                    | Check::Chat
                    | Check::Stream
                    | Check::Auto
                    | Check::HeadlessModels
                    | Check::HeadlessAttestation => (),
                }
                Ok(Progress::Pending)
            }
            Ok(Evidence::Accepted) => Ok(self.advance(check, elapsed)),
            Ok(Evidence::Model(model)) => {
                self.model = Some(model);
                Ok(self.advance(check, elapsed))
            }
        }
    }

    pub(crate) fn cancel(&mut self) -> Result<Progress, Rejection> {
        self.fail(Rejection::Cancelled)
    }

    pub(super) fn completion(&self) -> Result<(), Rejection> {
        match self.state {
            State::Complete => Ok(()),
            State::Checking(_) => Err(Rejection::Incomplete),
            State::Failed(reason) => Err(reason),
        }
    }

    fn fail<T>(&mut self, reason: Rejection) -> Result<T, Rejection> {
        self.state = State::Failed(reason);
        Err(reason)
    }

    fn advance(&mut self, check: Check, elapsed: Duration) -> Progress {
        let attestation = match self.attestation {
            Attestation::Disabled => false,
            Attestation::Required { .. } => true,
        };
        let next = match check {
            Check::InspectAttestation => Some(Check::Runtime),
            Check::Runtime => Some(Check::Models),
            Check::Models if attestation => Some(Check::RuntimeAttestation),
            Check::Models | Check::RuntimeAttestation => Some(Check::Chat),
            Check::Chat => Some(Check::Stream),
            Check::Stream => Some(Check::Auto),
            Check::Auto => Some(Check::HeadlessModels),
            Check::HeadlessModels => Some(Check::HeadlessStatus),
            Check::HeadlessStatus if attestation => Some(Check::HeadlessAttestation),
            Check::HeadlessStatus | Check::HeadlessAttestation => None,
        };
        if !matches!(check, Check::HeadlessModels) {
            self.stage_started = elapsed;
        }
        match next {
            Some(next) => {
                self.state = State::Checking(next);
                Progress::Advanced(next)
            }
            None => {
                self.state = State::Complete;
                Progress::Complete
            }
        }
    }
}
