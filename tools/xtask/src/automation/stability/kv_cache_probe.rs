//! measured cache evidence with exact and same-prefix request geometry.
use super::{
    kv_cache::Metrics,
    kv_requests::{self, PIN},
    kv_tool_calls,
    transport::Http,
};

pub(super) enum Geometry {
    SamePrefix,
    ExactBody,
    AfterOverlap,
}
pub(super) struct Proof {
    pub status_code: Option<u16>,
    pub metrics: Option<Metrics>,
    pub result: Result<String, String>,
}
impl Geometry {
    fn tails(self) -> (&'static str, &'static str) {
        match self {
            Self::SamePrefix => (
                "Same-prefix warmup tail alpha.",
                "Same-prefix measured tail beta.",
            ),
            Self::ExactBody => ("Exact-prefix warmup tail.", "Exact-prefix warmup tail."),
            Self::AfterOverlap => ("Overlap warmup tail alpha.", "Overlap measured tail beta."),
        }
    }
}
pub(super) async fn measure(
    http: &Http,
    model: &str,
    geometry: Geometry,
    minimum_cached: u64,
    suffix_limit: u64,
) -> Proof {
    let (warm_tail, measured_tail) = geometry.tails();
    let warm = kv_requests::cache(model, warm_tail);
    let measured = kv_requests::cache(model, measured_tail);
    if let Err(error) = http.chat(&warm, false).await {
        return Proof {
            status_code: error.status,
            metrics: None,
            result: Err(error.detail),
        };
    }
    let reply = match http.chat(&measured, false).await {
        Ok(reply) => reply,
        Err(error) => {
            return Proof {
                status_code: error.status,
                metrics: None,
                result: Err(error.detail),
            };
        }
    };
    let Some(response) = reply.json else {
        return Proof {
            status_code: Some(reply.status),
            metrics: None,
            result: Err("measured KV cache response was not JSON".into()),
        };
    };
    let metrics = Metrics::from_response(&response);
    let result = kv_tool_calls::text_message(&response, &[PIN])
        .and_then(|_| metrics.validate(minimum_cached, suffix_limit));
    // Keep observed geometry even when the measured completion or threshold fails.
    Proof {
        status_code: Some(reply.status),
        metrics: Some(metrics),
        result,
    }
}
