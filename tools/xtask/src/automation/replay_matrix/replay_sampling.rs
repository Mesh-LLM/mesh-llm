//! Explicit, manifest-pinned sampling for recorded performance comparisons.
use crate::command::DynResult;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Sampling {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: u32,
    pub min_p: f64,
    pub presence_penalty: f64,
    pub repeat_penalty: f64,
}

impl Sampling {
    pub(super) fn validate(&self) -> DynResult<()> {
        if !(0.0..=2.0).contains(&self.temperature)
            || !(0.0..=1.0).contains(&self.top_p)
            || self.top_p == 0.0
            || self.top_k == 0
            || i32::try_from(self.top_k).is_err()
            || !(0.0..=1.0).contains(&self.min_p)
            || !(-2.0..=2.0).contains(&self.presence_penalty)
            || !self.repeat_penalty.is_finite()
            || self.repeat_penalty <= 0.0
        {
            return Err("invalid recorded replay sampling parameters".into());
        }
        Ok(())
    }

    pub(super) fn apply(&self, body: &mut serde_json::Value) -> DynResult<()> {
        let fields = serde_json::to_value(self)?;
        body.as_object_mut()
            .ok_or("request must be an object")?
            .extend(
                fields
                    .as_object()
                    .ok_or("sampling must be an object")?
                    .clone(),
            );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn profile() -> serde_json::Value {
        json!({"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repeat_penalty":1.0})
    }

    #[test]
    fn sampling_is_explicit_and_cannot_override_output_or_history() {
        let value = profile();
        let sampling: Sampling = serde_json::from_value(value.clone()).unwrap();
        sampling.validate().unwrap();
        let mut body =
            json!({"messages":[{"role":"user","content":"recorded"}],"seed":42,"temperature":0});
        let history = body["messages"].clone();
        sampling.apply(&mut body).unwrap();
        for (key, value) in value.as_object().unwrap() {
            assert_eq!(&body[key], value);
        }
        assert_eq!(body["messages"], history);
        assert_eq!(body["seed"], 42);
        assert!(body.get("max_tokens").is_none());
        for key in ["messages", "max_tokens", "max_completion_tokens", "seed"] {
            let mut invalid = profile();
            invalid[key] = 1.into();
            assert!(serde_json::from_value::<Sampling>(invalid).is_err());
        }
    }

    #[test]
    fn invalid_sampling_fails_instead_of_silently_clamping() {
        for (key, value) in [
            ("temperature", json!(-1)),
            ("temperature", json!(3)),
            ("top_p", json!(0)),
            ("top_p", json!(1.1)),
            ("top_k", json!(0)),
            ("top_k", json!(u32::MAX)),
            ("min_p", json!(-0.1)),
            ("presence_penalty", json!(3)),
            ("repeat_penalty", json!(0)),
        ] {
            let mut invalid = profile();
            invalid[key] = value;
            assert!(
                serde_json::from_value::<Sampling>(invalid)
                    .unwrap()
                    .validate()
                    .is_err(),
                "{key}"
            );
        }
        let mut sampling: Sampling = serde_json::from_value(profile()).unwrap();
        sampling.temperature = f64::NAN;
        assert!(sampling.validate().is_err());
        sampling.temperature = 0.6;
        sampling.repeat_penalty = f64::INFINITY;
        assert!(sampling.validate().is_err());
    }
}
