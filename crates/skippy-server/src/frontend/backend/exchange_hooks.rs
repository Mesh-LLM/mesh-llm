//! Exchange hooks own their cancellation and terminal guards.
use super::*;

impl StageOpenAiBackend {
    #[cfg(test)]
    pub(super) async fn chat_completion_with_hooks<F, Fut>(
        &self,
        request: ChatCompletionRequest,
        dispatch: F,
    ) -> OpenAiResult<ChatCompletionResponse>
    where
        F: FnOnce(ChatCompletionRequest) -> Fut,
        Fut: std::future::Future<Output = OpenAiResult<ChatCompletionResponse>>,
    {
        self.chat_completion_with_prepared_hooks_for_exchange(
            request,
            None,
            move |request, admission| async move {
                admission.admit(&request).await?;
                dispatch(request).await
            },
        )
        .await
    }

    pub(super) async fn chat_completion_with_prepared_hooks_for_exchange<F, Fut>(
        &self,
        mut request: ChatCompletionRequest,
        observation_id: Option<String>,
        dispatch: F,
    ) -> OpenAiResult<ChatCompletionResponse>
    where
        F: FnOnce(ChatCompletionRequest, PreparedExchangeAdmission) -> Fut,
        Fut: std::future::Future<Output = OpenAiResult<ChatCompletionResponse>>,
    {
        let hooks = self.hook_policy.clone().filter(|hooks| {
            chat_mesh_hooks_enabled(&request) || hooks.requires_exchange_lifecycle()
        });
        let exchange_id = observation_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let mut guard = hooks
            .as_ref()
            .map(|hooks| TerminalGuard::new(hooks.clone(), request.clone(), exchange_id.clone()));
        let mut dispatched_request = None;

        if let Some(hooks) = hooks.clone() {
            match hooks.before_chat_completion(&mut request).await {
                Ok(outcome) => {
                    apply_chat_hook_outcome(&mut request, &outcome);
                }
                Err(error) => {
                    let reason = error.to_string();
                    if let Some(mut guard) = guard.take() {
                        guard.set_request(request.clone());
                        guard
                            .fire(&ChatCompletionOutcome::Denied {
                                status: error.status().as_u16(),
                                reason: &reason,
                            })
                            .await;
                    }
                    return Err(error);
                }
            }

            let effective = if hooks.observes_dispatched_request() {
                request.clone()
            } else {
                ChatCompletionRequest::default()
            };
            if let Some(guard) = guard.as_mut() {
                guard.set_request(effective.clone());
            }
            dispatched_request = Some(effective);
        }

        let admission = PreparedExchangeAdmission::new(hooks.clone(), exchange_id.clone());
        let mut result = dispatch(request, admission.clone()).await;
        if let Some(prepared) = admission.request() {
            if let Some(guard) = guard.as_mut() {
                guard.set_request(prepared.clone());
            }
            dispatched_request = Some(prepared);
        }

        // Carry the exchange id on the response only when this exchange is
        // tracked (a `TerminalGuard` is armed); otherwise no terminal event
        // fires for it and there is nothing to join to.
        if guard.is_some()
            && let Ok(response) = &mut result
        {
            response.exchange_id = Some(exchange_id.clone());
        }

        if let (Some(hooks), Some(dispatched_request), Ok(response)) =
            (&hooks, &dispatched_request, &mut result)
            && let Some(marker) = hooks
                .capsule_marker_for_response(dispatched_request, &*response)
                .await
        {
            if capsule_id_is_valid(&marker.capsule_id) {
                response.capsule_marker = Some(marker);
            } else {
                tracing::warn!(
                    capsule_id = %marker.capsule_id,
                    "dropping capsule marker: invalid capsule id"
                );
            }
        }

        if let Some(guard) = guard {
            let error_message;
            let terminal = match &result {
                Ok(response) => ChatCompletionOutcome::Success { response },
                Err(error) if admission.denied() => {
                    error_message = error.to_string();
                    ChatCompletionOutcome::Denied {
                        status: error.status().as_u16(),
                        reason: &error_message,
                    }
                }
                Err(error) => {
                    error_message = error.to_string();
                    ChatCompletionOutcome::Error {
                        status: error.status().as_u16(),
                        message: &error_message,
                    }
                }
            };
            guard.fire(&terminal).await;
        }
        result
    }

    #[cfg(test)]
    pub(super) async fn chat_completion_stream_with_hooks<F, Fut>(
        &self,
        request: ChatCompletionRequest,
        context: &OpenAiRequestContext,
        dispatch: F,
    ) -> OpenAiResult<ChatCompletionStream>
    where
        F: FnOnce(ChatCompletionRequest) -> Fut,
        Fut: std::future::Future<Output = OpenAiResult<ChatCompletionStream>>,
    {
        self.chat_completion_stream_with_prepared_hooks(
            request,
            context,
            move |request, admission| async move {
                admission.admit(&request).await?;
                dispatch(request).await
            },
        )
        .await
    }

    pub(super) async fn chat_completion_stream_with_prepared_hooks<F, Fut>(
        &self,
        mut request: ChatCompletionRequest,
        context: &OpenAiRequestContext,
        dispatch: F,
    ) -> OpenAiResult<ChatCompletionStream>
    where
        F: FnOnce(ChatCompletionRequest, PreparedExchangeAdmission) -> Fut,
        Fut: std::future::Future<Output = OpenAiResult<ChatCompletionStream>>,
    {
        let hooks = self.hook_policy.clone().filter(|hooks| {
            chat_mesh_hooks_enabled(&request) || hooks.requires_exchange_lifecycle()
        });
        let exchange_id = uuid::Uuid::new_v4().to_string();
        let mut guard = hooks
            .as_ref()
            .map(|hooks| TerminalGuard::new(hooks.clone(), request.clone(), exchange_id.clone()));
        if guard.is_some() {
            context.publish_exchange_id(exchange_id.clone());
        }

        if let Some(hooks) = hooks.clone() {
            match hooks.before_chat_completion(&mut request).await {
                Ok(outcome) => {
                    apply_chat_hook_outcome(&mut request, &outcome);
                }
                Err(error) => {
                    let reason = error.to_string();
                    if let Some(mut guard) = guard.take() {
                        guard.set_request(request.clone());
                        guard
                            .fire(&ChatCompletionOutcome::Denied {
                                status: error.status().as_u16(),
                                reason: &reason,
                            })
                            .await;
                    }
                    return Err(error);
                }
            }
            if let Some(guard) = guard.as_mut() {
                guard.set_request(request.clone());
            }
        }

        let admission = PreparedExchangeAdmission::new(hooks.clone(), exchange_id.clone());
        let result = dispatch(request, admission.clone()).await;
        if let Some(prepared) = admission.request()
            && let Some(guard) = guard.as_mut()
        {
            guard.set_request(prepared);
        }
        match result {
            Ok(stream) => Ok(match guard {
                Some(guard) => TerminalGuardedChatStream::pinned(stream, guard),
                None => stream,
            }),
            Err(error) => {
                if let Some(guard) = guard {
                    let message = error.to_string();
                    let outcome = if admission.denied() {
                        ChatCompletionOutcome::Denied {
                            status: error.status().as_u16(),
                            reason: &message,
                        }
                    } else {
                        ChatCompletionOutcome::Error {
                            status: error.status().as_u16(),
                            message: &message,
                        }
                    };
                    guard.fire(&outcome).await;
                }
                Err(error)
            }
        }
    }
}
