//! A System One backend for a model that serves nothing else, such as Laya.

use std::sync::Arc;

use async_trait::async_trait;
use skippy_inference_api::{
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStream, ModelObject,
    OpenAiBackend, OpenAiError, OpenAiRequestContext, OpenAiResult, SystemOneRequest,
    SystemOneResponse,
};
use skippy_runtime::DecisionModel;
use tokio::task;

use super::run_on_model;

/// Serves `POST /systemone` from a loaded decision-only model such as Laya.
/// Every other OpenAI surface is refused: the model generates no text.
#[derive(Clone)]
pub struct LayaSystemOneBackend {
    model_id: String,
    model: Arc<dyn DecisionModel + Send + Sync>,
}

impl LayaSystemOneBackend {
    pub fn new(model_id: impl Into<String>, model: Arc<dyn DecisionModel + Send + Sync>) -> Self {
        Self {
            model_id: model_id.into(),
            model,
        }
    }
}

#[async_trait]
impl OpenAiBackend for LayaSystemOneBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new(self.model_id.clone())])
    }

    async fn system_one(&self, request: SystemOneRequest) -> OpenAiResult<SystemOneResponse> {
        let backend = self.clone();
        task::spawn_blocking(move || {
            run_on_model(backend.model.as_ref(), &backend.model_id, request)
        })
        .await
        .map_err(|error| {
            OpenAiError::backend(format!("System One execution task failed: {error}"))
        })?
    }

    async fn chat_completion(
        &self,
        _request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        Err(decision_only())
    }

    async fn chat_completion_stream(
        &self,
        _request: ChatCompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        Err(decision_only())
    }
}

fn decision_only() -> OpenAiError {
    OpenAiError::unsupported("this Laya decision model only serves POST /systemone")
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use axum::body::{Body, to_bytes};
    use axum::http::{Request, StatusCode};
    use serde_json::{Value, json};
    use skippy_runtime::{DecisionError, DecisionOutput, DecisionQuestionKind, DecisionRequest};
    use tower::ServiceExt;

    use super::*;

    /// Answers every question with a fixed distribution and records what it
    /// was asked, so the tests see exactly what reached the model.
    #[derive(Default)]
    struct FakeModel {
        seen: Mutex<Vec<DecisionRequest>>,
        refuse: Option<&'static str>,
    }

    impl DecisionModel for FakeModel {
        fn decide(&self, request: &DecisionRequest) -> Result<DecisionOutput, DecisionError> {
            self.seen.lock().unwrap().push(request.clone());
            if let Some(reason) = self.refuse {
                return Err(DecisionError::InvalidRequest(reason.into()));
            }
            let probabilities = request
                .questions
                .iter()
                .map(|question| match &question.kind {
                    DecisionQuestionKind::Noul { .. } => vec![0.8, 0.2],
                    DecisionQuestionKind::Choice { options } => {
                        // The last option in request order wins.
                        let mut distribution =
                            vec![0.1 / (options.len() - 1) as f32; options.len()];
                        *distribution.last_mut().unwrap() = 0.9;
                        distribution
                    }
                    DecisionQuestionKind::Score { levels } => {
                        vec![1.0 / levels.len() as f32; levels.len()]
                    }
                })
                .collect();
            Ok(DecisionOutput {
                probabilities,
                input_tokens: 42,
            })
        }
    }

    async fn post(model: Arc<FakeModel>, path: &str, body: Value) -> (StatusCode, Value) {
        post_raw(model, path, body.to_string()).await
    }

    /// Sends the body text as given. `json!` goes through serde_json's sorted
    /// map, so tests about request order must spell the JSON out.
    async fn post_raw(model: Arc<FakeModel>, path: &str, body: String) -> (StatusCode, Value) {
        let router = skippy_inference_api::router_for(Arc::new(LayaSystemOneBackend::new(
            "laya-test",
            model,
        )));
        let response = router
            .oneshot(
                Request::post(path)
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    #[tokio::test]
    async fn systemone_reaches_the_model_and_maps_answers() {
        let model = Arc::new(FakeModel::default());
        let (status, body) = post_raw(
            model.clone(),
            "/systemone",
            r#"{"model":"laya-test","state":{"zeta":1,"alpha":2},"questions":{"billing":{"type":"noul","instructions":"Billing?"},"team":{"type":"choice","criteria":{"support":"","billing":""}},"urgency":{"type":"score","criteria":["low","high"]}}}"#.to_string(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["model"], "laya-test");
        assert_eq!(
            body["usage"],
            json!({"input_tokens": 42, "output_tokens": 0})
        );
        assert!((body["answers"]["billing"]["noul"].as_f64().unwrap() - 0.8).abs() < 1e-6);
        // Request order reaches the model: `billing` is the last option given.
        assert_eq!(body["answers"]["team"]["choice"], "billing");
        assert!((body["answers"]["urgency"]["score"].as_f64().unwrap() - 0.5).abs() < 1e-6);

        let seen = model.seen.lock().unwrap();
        let skippy_runtime::DecisionValue::Object(state) = &seen[0].state else {
            panic!("object state")
        };
        assert_eq!(state[0].0, "zeta");
    }

    #[tokio::test]
    async fn a_jev_alias_is_accepted() {
        let (status, body) = post(
            Arc::new(FakeModel::default()),
            "/systemone",
            json!({"model": "openjev-latest", "state": "s", "questions": {"q": {"type": "noul"}}}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["model"], "openjev-latest");
    }

    #[tokio::test]
    async fn requests_are_validated_before_the_model() {
        let model = Arc::new(FakeModel::default());
        let (status, _) = post(
            model.clone(),
            "/systemone",
            json!({"model": "other", "state": "s", "questions": {"q": {"type": "noul"}}}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (status, _) = post(
            model.clone(),
            "/systemone",
            json!({"model": "laya-test", "state": "s", "questions": {"q": {"type": "choice", "criteria": {"only": ""}}}}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(model.seen.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_model_refusal_is_a_bad_request() {
        let model = Arc::new(FakeModel {
            refuse: Some("question options do not fit in the Laya sequence budget"),
            ..FakeModel::default()
        });
        let (status, body) = post(
            model,
            "/systemone",
            json!({"model": "laya-test", "state": "s", "questions": {"q": {"type": "noul"}}}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.to_string().contains("sequence budget"), "{body}");
    }

    #[tokio::test]
    async fn chat_is_refused_and_models_lists_the_id() {
        let (status, _) = post(
            Arc::new(FakeModel::default()),
            "/v1/chat/completions",
            json!({"model": "laya-test", "messages": [{"role": "user", "content": "hi"}]}),
        )
        .await;
        assert!(!status.is_success(), "{status}");

        let router = skippy_inference_api::router_for(Arc::new(LayaSystemOneBackend::new(
            "laya-test",
            Arc::new(FakeModel::default()),
        )));
        let response = router
            .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        let models: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(models["data"][0]["id"], "laya-test");
    }
}
