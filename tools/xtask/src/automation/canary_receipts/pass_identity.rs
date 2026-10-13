//! The closed distributed repair/verification sequence.
use super::{Error, ErrorKind};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in super::super) enum PassId {
    Repair1,
    Verify1,
    Repair2,
    Verify2,
    Repair3,
    Verify3,
}

impl PassId {
    pub(in super::super) fn parse(value: &str) -> Result<Self, Error> {
        match value {
            "repair-1" => Ok(Self::Repair1),
            "verify-1" => Ok(Self::Verify1),
            "repair-2" => Ok(Self::Repair2),
            "verify-2" => Ok(Self::Verify2),
            "repair-3" => Ok(Self::Repair3),
            "verify-3" => Ok(Self::Verify3),
            _ => Err(Error::new(
                ErrorKind::PackageIdentity,
                "invalid bounded canary pass identity",
            )),
        }
    }

    pub(in super::super) fn is_repair(self) -> bool {
        matches!(self, Self::Repair1 | Self::Repair2 | Self::Repair3)
    }

    /// A resumed repair consumes the preceding candidate or its failed verifier.
    /// An independent verifier consumes only its own repair's exact package.
    pub(in super::super) fn accepts_previous(self, previous: Self) -> bool {
        matches!(
            (self, previous),
            (Self::Repair2, Self::Repair1 | Self::Verify1)
                | (Self::Repair3, Self::Repair2 | Self::Verify2)
                | (Self::Verify1, Self::Repair1)
                | (Self::Verify2, Self::Repair2)
                | (Self::Verify3, Self::Repair3)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::PassId;

    #[test]
    fn only_immediate_producers_can_resume_or_start_independent_verification() {
        use PassId::{Repair1, Repair2, Repair3, Verify1, Verify2, Verify3};
        let passes = [Repair1, Verify1, Repair2, Verify2, Repair3, Verify3];
        let allowed = [
            (Repair2, Repair1),
            (Repair2, Verify1),
            (Repair3, Repair2),
            (Repair3, Verify2),
            (Verify1, Repair1),
            (Verify2, Repair2),
            (Verify3, Repair3),
        ];
        for current in passes {
            for previous in passes {
                assert_eq!(
                    current.accepts_previous(previous),
                    allowed.contains(&(current, previous)),
                    "{current:?} <- {previous:?}"
                );
            }
        }
        for invalid in ["repair-0", "repair-4", "verify-4", "pinned", "repair-01"] {
            assert!(PassId::parse(invalid).is_err());
        }
    }
}
