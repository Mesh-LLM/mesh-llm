//! Hidden process compatibility fixture; omitted from the ordinary manifest.
use mesh_llm_plugin::{OperationRouter, operation_with_schema, structured_tool_result};
use serde_json::{Value, json};

pub fn router(legacy: bool) -> OperationRouter {
    let mut router = super::openai_exchange_identity::router();
    if legacy {
        router.add_raw(
            operation_with_schema(
                "legacy_echo",
                "Generation-3 compatibility fixture",
                json!({"type":"object"}).as_object().unwrap().clone(),
            ),
            |request, _context| {
                Box::pin(async move {
                    let arguments: Value = request.arguments()?;
                    structured_tool_result(json!({"echo":arguments}))
                })
            },
        );
    }
    router
}
