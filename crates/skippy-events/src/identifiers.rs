//! Inference request correlation identity.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct RequestId(Uuid);

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestId {
    /// Generate a fresh request identifier.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl From<Uuid> for RequestId {
    fn from(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

impl AsRef<Uuid> for RequestId {
    fn as_ref(&self) -> &Uuid {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_identity_preserves_uuid_wire_shape() {
        let uuid = Uuid::parse_str("4ba6be8e-11c7-4aac-9688-b7bf920d190a").unwrap();
        let id = RequestId::from(uuid);
        let wire = serde_json::to_value(id).unwrap();
        assert_eq!(wire, serde_json::to_value(uuid).unwrap());
        assert_eq!(serde_json::from_value::<RequestId>(wire).unwrap(), id);
        assert_eq!(id.as_uuid(), uuid);
    }

    #[test]
    fn request_ids_are_unique() {
        assert_ne!(RequestId::new(), RequestId::new());
    }
}
