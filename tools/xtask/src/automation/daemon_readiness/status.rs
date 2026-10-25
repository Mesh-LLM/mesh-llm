use super::Rejection;
use serde::{Deserialize, Deserializer};

#[cfg(test)]
#[path = "../../../tests/migration_lifecycle/daemon/status.rs"]
mod tests;

#[derive(Deserialize)]
struct Status {
    api_port: u16,
    #[serde(deserialize_with = "instances")]
    local_instances: Vec<Instance>,
}

#[derive(Deserialize)]
struct Instance {
    pid: u32,
    is_self: bool,
}

pub(super) fn identify(body: &[u8], identity: (u32, u16)) -> Result<(), Rejection> {
    let body = std::str::from_utf8(body).map_err(|_| Rejection::MalformedStatus)?;
    if !body.trim_start().starts_with('{') {
        return Err(Rejection::MalformedStatus);
    }
    let status: Status = serde_json::from_str(body).map_err(|_| Rejection::MalformedStatus)?;
    let (pid, api_port) = identity;
    let mut selves = status.local_instances.iter().filter(|row| row.is_self);
    if status.api_port != api_port
        || api_port == 0
        || pid == 0
        || !selves.next().is_some_and(|row| row.pid == pid)
        || selves.next().is_some()
    {
        return Err(Rejection::OwnershipMismatch);
    }
    Ok(())
}

fn instances<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<Instance>, D::Error> {
    struct Object(Instance);
    impl<'de> Deserialize<'de> for Object {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct Visitor;
            impl<'de> serde::de::Visitor<'de> for Visitor {
                type Value = Object;
                fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    formatter.write_str("an instance object")
                }
                fn visit_map<M: serde::de::MapAccess<'de>>(
                    self,
                    map: M,
                ) -> Result<Self::Value, M::Error> {
                    Instance::deserialize(serde::de::value::MapAccessDeserializer::new(map))
                        .map(Object)
                }
            }
            deserializer.deserialize_map(Visitor)
        }
    }
    Vec::<Object>::deserialize(deserializer).map(|rows| rows.into_iter().map(|row| row.0).collect())
}
