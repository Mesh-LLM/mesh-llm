use std::ffi::CString;
use std::path::Path;
use std::ptr;

use anyhow::{Context, Result, anyhow};
use skippy_ffi::{ChatMessage as RawChatMessage, Model as RawModel};

use crate::error::{ensure_ok, free_error};
use crate::logging::write_native_log_note;
use crate::media::MediaProjector;
use crate::path_cstring::path_to_cstring;
use crate::runtime_events;
use crate::session::StageSession;
use crate::{
    ChatReasoningFormat, ChatTemplateJsonOptions, ChatTemplateJsonResult, ChatTemplateMessage,
    ChatTemplateOptions, RuntimeConfig, RuntimeEvent, Status,
};

pub struct StageModel {
    pub(crate) raw: *mut RawModel,
    pub(crate) media: Option<MediaProjector>,
}

// The experimental C ABI owns synchronization internally for model/session use.
// Rust stage-server access is additionally serialized behind a Mutex.
unsafe impl Send for StageModel {}

impl StageModel {
    pub fn new_dummy() -> Self {
        Self {
            raw: std::ptr::null_mut(),
            media: None,
        }
    }

    fn from_opened_raw(
        raw: *mut RawModel,
        config: &RuntimeConfig,
        null_handle_message: &'static str,
    ) -> Result<Self> {
        if raw.is_null() {
            return Err(anyhow!(null_handle_message));
        }
        let media = config
            .projector_path
            .as_deref()
            .map(|projector_path| MediaProjector::open(projector_path, raw))
            .transpose()?;
        Ok(Self { raw, media })
    }

    fn open_path_with_optional_event_reporter(
        path: impl AsRef<Path>,
        config: &RuntimeConfig,
        event_reporter: Option<&mut dyn FnMut(RuntimeEvent)>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let use_events = event_reporter.is_some() && runtime_events::model_open_events_supported();
        let begin_label = if use_events {
            "skippy_model_open_with_events begin"
        } else {
            "skippy_model_open begin"
        };
        let end_label = if use_events {
            "skippy_model_open_with_events returned"
        } else {
            "skippy_model_open returned"
        };
        let null_handle_message = if use_events {
            "skippy_model_open_with_events returned a null handle"
        } else {
            "skippy_model_open returned a null handle"
        };
        write_native_log_note(format!(
            "{begin_label} path={} {}",
            path.display(),
            config.native_log_summary()
        ));
        let path = path_to_cstring(path, "model path")?;
        let raw_config = config.as_raw()?;
        #[cfg(not(test))]
        let (raw, status, error) = runtime_events::run_model_open(
            |out_model, out_error| unsafe {
                skippy_ffi::skippy_model_open(path.as_ptr(), &raw_config.raw, out_model, out_error)
            },
            |reporter, out_model, out_error| unsafe {
                let open_with_events_symbol = runtime_events::model_open_with_events_symbol()
                    .expect("runtime-event symbol availability checked before use");
                open_with_events_symbol(
                    path.as_ptr(),
                    &raw_config.raw,
                    reporter,
                    out_model,
                    out_error,
                )
            },
            event_reporter,
            use_events,
        );
        #[cfg(test)]
        let (raw, status, error) = {
            debug_assert!(event_reporter.is_none());
            runtime_events::run_model_open(
                |out_model, out_error| unsafe {
                    skippy_ffi::skippy_model_open(
                        path.as_ptr(),
                        &raw_config.raw,
                        out_model,
                        out_error,
                    )
                },
                |_reporter, _out_model, _out_error| {
                    unreachable!("test builds do not link _with_events model-open symbols")
                },
                None,
                false,
            )
        };
        write_native_log_note(format!("{end_label} status={status:?}"));
        ensure_ok(status, error)?;
        Self::from_opened_raw(raw, config, null_handle_message)
    }

    fn open_parts_with_optional_event_reporter(
        paths: &[impl AsRef<Path>],
        config: &RuntimeConfig,
        event_reporter: Option<&mut dyn FnMut(RuntimeEvent)>,
    ) -> Result<Self> {
        if paths.is_empty() {
            return Err(anyhow!("at least one GGUF part path is required"));
        }
        let use_events = event_reporter.is_some() && runtime_events::model_open_events_supported();
        let begin_label = if use_events {
            "skippy_model_open_from_parts_with_events begin"
        } else {
            "skippy_model_open_from_parts begin"
        };
        let end_label = if use_events {
            "skippy_model_open_from_parts_with_events returned"
        } else {
            "skippy_model_open_from_parts returned"
        };
        let null_handle_message = if use_events {
            "skippy_model_open_from_parts_with_events returned a null handle"
        } else {
            "skippy_model_open_from_parts returned a null handle"
        };
        let path_list = paths
            .iter()
            .map(|path| path.as_ref().display().to_string())
            .collect::<Vec<_>>()
            .join(",");
        write_native_log_note(format!(
            "{begin_label} parts={} {}",
            path_list,
            config.native_log_summary()
        ));
        let paths = paths
            .iter()
            .map(|path| path_to_cstring(path.as_ref(), "part path"))
            .collect::<Result<Vec<_>>>()?;
        let path_ptrs = paths.iter().map(|path| path.as_ptr()).collect::<Vec<_>>();
        let raw_config = config.as_raw()?;
        #[cfg(not(test))]
        let (raw, status, error) = runtime_events::run_model_open(
            |out_model, out_error| unsafe {
                skippy_ffi::skippy_model_open_from_parts(
                    path_ptrs.as_ptr(),
                    path_ptrs.len(),
                    &raw_config.raw,
                    out_model,
                    out_error,
                )
            },
            |reporter, out_model, out_error| unsafe {
                let open_from_parts_with_events_symbol =
                    runtime_events::model_open_from_parts_with_events_symbol()
                        .expect("runtime-event symbol availability checked before use");
                open_from_parts_with_events_symbol(
                    path_ptrs.as_ptr(),
                    path_ptrs.len(),
                    &raw_config.raw,
                    reporter,
                    out_model,
                    out_error,
                )
            },
            event_reporter,
            use_events,
        );
        #[cfg(test)]
        let (raw, status, error) = {
            debug_assert!(event_reporter.is_none());
            runtime_events::run_model_open(
                |out_model, out_error| unsafe {
                    skippy_ffi::skippy_model_open_from_parts(
                        path_ptrs.as_ptr(),
                        path_ptrs.len(),
                        &raw_config.raw,
                        out_model,
                        out_error,
                    )
                },
                |_reporter, _out_model, _out_error| {
                    unreachable!(
                        "test builds do not link _with_events model-open-from-parts symbols"
                    )
                },
                None,
                false,
            )
        };
        write_native_log_note(format!("{end_label} status={status:?}"));
        ensure_ok(status, error)?;
        Self::from_opened_raw(raw, config, null_handle_message)
    }

    pub fn open(path: impl AsRef<Path>, config: &RuntimeConfig) -> Result<Self> {
        Self::open_path_with_optional_event_reporter(path, config, None)
    }

    pub fn open_with_events(
        path: impl AsRef<Path>,
        config: &RuntimeConfig,
        event_reporter: &mut dyn FnMut(RuntimeEvent),
    ) -> Result<Self> {
        #[cfg(test)]
        {
            let _ = event_reporter;
            Self::open_path_with_optional_event_reporter(path, config, None)
        }

        #[cfg(not(test))]
        Self::open_path_with_optional_event_reporter(path, config, Some(event_reporter))
    }

    pub fn open_from_parts(paths: &[impl AsRef<Path>], config: &RuntimeConfig) -> Result<Self> {
        Self::open_parts_with_optional_event_reporter(paths, config, None)
    }

    pub fn open_from_parts_with_events(
        paths: &[impl AsRef<Path>],
        config: &RuntimeConfig,
        event_reporter: &mut dyn FnMut(RuntimeEvent),
    ) -> Result<Self> {
        #[cfg(test)]
        {
            let _ = event_reporter;
            Self::open_parts_with_optional_event_reporter(paths, config, None)
        }

        #[cfg(not(test))]
        Self::open_parts_with_optional_event_reporter(paths, config, Some(event_reporter))
    }

    pub fn attach_mtp_draft_model(
        &mut self,
        path: impl AsRef<Path>,
        config: &RuntimeConfig,
    ) -> Result<()> {
        if self.raw.is_null() {
            return Err(anyhow!("cannot attach MTP draft model to a null model"));
        }
        let attach_symbol = skippy_ffi::skippy_model_attach_mtp_draft_model_fn()
            .ok_or_else(|| anyhow!("native runtime does not support external MTP draft models"))?;
        let path = path.as_ref();
        write_native_log_note(format!(
            "skippy_model_attach_mtp_draft_model begin path={} {}",
            path.display(),
            config.native_log_summary()
        ));
        let path = path_to_cstring(path, "MTP draft model path")?;
        let raw_config = config.as_raw()?;
        let mut error = ptr::null_mut();
        let status = unsafe { attach_symbol(self.raw, path.as_ptr(), &raw_config.raw, &mut error) };
        write_native_log_note(format!(
            "skippy_model_attach_mtp_draft_model returned status={status:?}"
        ));
        ensure_ok(status, error)
    }

    pub fn create_session(&self) -> Result<StageSession> {
        write_native_log_note("skippy_session_create begin");
        let mut raw = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe { skippy_ffi::skippy_session_create(self.raw, &mut raw, &mut error) };
        write_native_log_note(format!("skippy_session_create returned status={status:?}"));
        ensure_ok(status, error)?;
        if raw.is_null() {
            return Err(anyhow!("skippy_session_create returned a null handle"));
        }
        Ok(StageSession {
            raw,
            token_count: 0,
        })
    }

    pub fn create_session_from_resident_prefix(
        &self,
        cache_seq_id: i32,
        token_ids: &[i32],
    ) -> Result<StageSession> {
        let mut raw = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_session_create_from_resident_prefix(
                self.raw,
                cache_seq_id,
                token_ids.as_ptr(),
                token_ids.len(),
                &mut raw,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        if raw.is_null() {
            return Err(anyhow!(
                "skippy_session_create_from_resident_prefix returned a null handle"
            ));
        }
        Ok(StageSession {
            raw,
            token_count: u64::try_from(token_ids.len()).context("token count exceeds u64")?,
        })
    }

    pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>> {
        let text = CString::new(text).context("text contains an interior NUL byte")?;
        let mut count = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_tokenize(
                self.raw,
                text.as_ptr(),
                add_special,
                ptr::null_mut(),
                0,
                &mut count,
                &mut error,
            )
        };
        if status != Status::BufferTooSmall && status != Status::Ok {
            ensure_ok(status, error)?;
        } else {
            free_error(error);
        }

        let mut tokens = vec![0_i32; count];
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_tokenize(
                self.raw,
                text.as_ptr(),
                add_special,
                tokens.as_mut_ptr(),
                tokens.len(),
                &mut count,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        tokens.truncate(count);
        Ok(tokens)
    }

    pub fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        Ok(String::from_utf8_lossy(&self.detokenize_bytes(tokens)?).into_owned())
    }

    pub fn detokenize_bytes(&self, tokens: &[i32]) -> Result<Vec<u8>> {
        let mut bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_detokenize(
                self.raw,
                tokens.as_ptr(),
                tokens.len(),
                ptr::null_mut(),
                0,
                &mut bytes,
                &mut error,
            )
        };
        if status != Status::BufferTooSmall && status != Status::Ok {
            ensure_ok(status, error)?;
        } else {
            free_error(error);
        }

        let mut output = vec![0_u8; bytes.max(1)];
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_detokenize(
                self.raw,
                tokens.as_ptr(),
                tokens.len(),
                output.as_mut_ptr().cast(),
                output.len(),
                &mut bytes,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        output.truncate(bytes);
        Ok(output)
    }

    pub fn token_is_eog(&self, token: i32) -> Result<bool> {
        let mut is_eog = false;
        let mut error = ptr::null_mut();
        let status =
            unsafe { skippy_ffi::skippy_token_is_eog(self.raw, token, &mut is_eog, &mut error) };
        ensure_ok(status, error)?;
        Ok(is_eog)
    }

    pub fn apply_chat_template(
        &self,
        messages: &[ChatTemplateMessage],
        add_assistant: bool,
    ) -> Result<String> {
        self.apply_chat_template_with_options(
            messages,
            ChatTemplateOptions {
                add_assistant,
                enable_thinking: None,
                reasoning_format: None,
            },
        )
    }

    pub fn apply_chat_template_with_options(
        &self,
        messages: &[ChatTemplateMessage],
        options: ChatTemplateOptions,
    ) -> Result<String> {
        let roles = messages
            .iter()
            .map(|message| {
                CString::new(message.role.as_str())
                    .context("message role contains an interior NUL byte")
            })
            .collect::<Result<Vec<_>>>()?;
        let contents = messages
            .iter()
            .map(|message| {
                CString::new(message.content.as_str())
                    .context("message content contains an interior NUL byte")
            })
            .collect::<Result<Vec<_>>>()?;
        let raw_messages = roles
            .iter()
            .zip(contents.iter())
            .map(|(role, content)| RawChatMessage {
                role: role.as_ptr(),
                content: content.as_ptr(),
            })
            .collect::<Vec<_>>();

        let mut bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_apply_chat_template(
                self.raw,
                raw_messages.as_ptr(),
                raw_messages.len(),
                options.add_assistant,
                options.enable_thinking.is_some(),
                options.enable_thinking.unwrap_or(true),
                ptr::null_mut(),
                0,
                &mut bytes,
                &mut error,
            )
        };
        if status != Status::BufferTooSmall && status != Status::Ok {
            ensure_ok(status, error)?;
        } else {
            free_error(error);
        }

        let mut output = vec![0_u8; bytes.max(1)];
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_apply_chat_template(
                self.raw,
                raw_messages.as_ptr(),
                raw_messages.len(),
                options.add_assistant,
                options.enable_thinking.is_some(),
                options.enable_thinking.unwrap_or(true),
                output.as_mut_ptr().cast(),
                output.len(),
                &mut bytes,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        output.truncate(bytes);
        String::from_utf8(output).context("chat template output is not valid UTF-8")
    }

    pub fn apply_chat_template_json(
        &self,
        messages_json: &str,
        options: ChatTemplateJsonOptions,
    ) -> Result<ChatTemplateJsonResult> {
        let messages_json =
            CString::new(messages_json).context("messages JSON contains an interior NUL byte")?;
        let tools_json = options
            .tools_json
            .as_deref()
            .map(CString::new)
            .transpose()
            .context("tools JSON contains an interior NUL byte")?;
        let tool_choice_json = options
            .tool_choice_json
            .as_deref()
            .map(CString::new)
            .transpose()
            .context("tool choice JSON contains an interior NUL byte")?;
        let tools_ptr = tools_json
            .as_ref()
            .map(|value| value.as_ptr())
            .unwrap_or(ptr::null());
        let tool_choice_ptr = tool_choice_json
            .as_ref()
            .map(|value| value.as_ptr())
            .unwrap_or(ptr::null());
        let reasoning_format = options
            .reasoning_format
            .map(ChatReasoningFormat::parser_name)
            .map(CString::new)
            .transpose()
            .context("reasoning format contains an interior NUL byte")?;
        let reasoning_format_ptr = reasoning_format
            .as_ref()
            .map(|value| value.as_ptr())
            .unwrap_or(ptr::null());

        let mut prompt_bytes = 0usize;
        let mut metadata_bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_apply_chat_template_json(
                self.raw,
                messages_json.as_ptr(),
                tools_ptr,
                tool_choice_ptr,
                options.add_assistant,
                options.enable_thinking.is_some(),
                options.enable_thinking.unwrap_or(true),
                options.parallel_tool_calls,
                reasoning_format_ptr,
                ptr::null_mut(),
                0,
                &mut prompt_bytes,
                ptr::null_mut(),
                0,
                &mut metadata_bytes,
                &mut error,
            )
        };
        if status != Status::BufferTooSmall && status != Status::Ok {
            ensure_ok(status, error)?;
        } else {
            free_error(error);
        }

        let mut prompt = vec![0_u8; prompt_bytes.max(1)];
        let mut metadata = vec![0_u8; metadata_bytes.max(1)];
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_apply_chat_template_json(
                self.raw,
                messages_json.as_ptr(),
                tools_ptr,
                tool_choice_ptr,
                options.add_assistant,
                options.enable_thinking.is_some(),
                options.enable_thinking.unwrap_or(true),
                options.parallel_tool_calls,
                reasoning_format_ptr,
                prompt.as_mut_ptr().cast(),
                prompt.len(),
                &mut prompt_bytes,
                metadata.as_mut_ptr().cast(),
                metadata.len(),
                &mut metadata_bytes,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        prompt.truncate(prompt_bytes);
        metadata.truncate(metadata_bytes);
        Ok(ChatTemplateJsonResult {
            prompt: String::from_utf8(prompt).context("chat template output is not valid UTF-8")?,
            metadata_json: String::from_utf8(metadata)
                .context("chat template metadata is not valid UTF-8")?,
        })
    }

    pub fn parse_chat_response_json(
        &self,
        generated_text: &str,
        metadata_json: &str,
        is_partial: bool,
    ) -> Result<String> {
        let generated_text =
            CString::new(generated_text).context("generated text contains an interior NUL byte")?;
        let metadata_json = CString::new(metadata_json)
            .context("chat template metadata contains an interior NUL byte")?;

        let mut bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_parse_chat_response_json(
                generated_text.as_ptr(),
                metadata_json.as_ptr(),
                is_partial,
                ptr::null_mut(),
                0,
                &mut bytes,
                &mut error,
            )
        };
        if status != Status::BufferTooSmall && status != Status::Ok {
            ensure_ok(status, error)?;
        } else {
            free_error(error);
        }

        let mut output = vec![0_u8; bytes.max(1)];
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_parse_chat_response_json(
                generated_text.as_ptr(),
                metadata_json.as_ptr(),
                is_partial,
                output.as_mut_ptr().cast(),
                output.len(),
                &mut bytes,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        output.truncate(bytes);
        String::from_utf8(output).context("parsed chat response is not valid UTF-8")
    }
}

impl Drop for StageModel {
    fn drop(&mut self) {
        self.media.take();
        if !self.raw.is_null() {
            unsafe {
                let _ = skippy_ffi::skippy_model_free(self.raw, ptr::null_mut());
            }
        }
    }
}
