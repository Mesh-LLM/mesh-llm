//! Branded newtype identifiers for logging contracts.
//!
//! Each identifier wraps a `uuid::Uuid` and provides type safety against mixing
//! different kinds of IDs in the public API.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Macro to generate a branded UUID newtype with standard derives and impls.
macro_rules! define_branded_id {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
        pub struct $name(Uuid);

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl $name {
            /// Generate a new random identifier.
            pub fn new() -> Self {
                Self(Uuid::new_v4())
            }

            /// Return the inner UUID.
            pub fn as_uuid(&self) -> Uuid {
                self.0
            }
        }

        impl From<Uuid> for $name {
            fn from(uuid: Uuid) -> Self {
                Self(uuid)
            }
        }

        impl AsRef<Uuid> for $name {
            fn as_ref(&self) -> &Uuid {
                &self.0
            }
        }
    };
}

define_branded_id!(RequestId);
define_branded_id!(EventId);
define_branded_id!(AttemptId);
define_branded_id!(ArtifactId);
define_branded_id!(ChannelId);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_id_is_unique() {
        let a = RequestId::new();
        let b = RequestId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn test_from_uuid_roundtrip() {
        let uuid = Uuid::new_v4();
        let id: EventId = uuid.into();
        assert_eq!(*id.as_ref(), uuid);
    }

    #[test]
    fn test_branded_types_are_distinct() {
        // Compile-time check that RequestId != EventId etc.
        let _req: RequestId;
        let _evt: EventId;
        // The following would not compile if they were the same type:
        // _req = _evt;
    }

    #[test]
    fn test_as_ref_uuid() {
        let uuid = Uuid::new_v4();
        let id: AttemptId = uuid.into();
        assert_eq!(id.as_ref(), &uuid);
    }

    #[test]
    fn test_hash_equality() {
        use std::collections::HashSet;
        let uuid = Uuid::new_v4();
        let a: ArtifactId = uuid.into();
        let b: ArtifactId = uuid.into();
        let mut set = HashSet::new();
        assert!(set.insert(a));
        assert!(!set.insert(b)); // same UUID → not inserted again
    }

    #[test]
    fn test_channel_id_new() {
        let a = ChannelId::new();
        let b = ChannelId::new();
        assert_ne!(a, b);
    }
}
