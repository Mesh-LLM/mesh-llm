use std::ffi::{c_char, c_float, c_int, c_void};

use crate::MtmdProgressCallback;

#[repr(C)]
pub struct MtmdContext {
    _private: [u8; 0],
}

#[repr(C)]
pub struct MtmdBitmap {
    _private: [u8; 0],
}

#[repr(C)]
pub struct MtmdInputChunks {
    _private: [u8; 0],
}

#[repr(C)]
pub struct MtmdHelperVideo {
    _private: [u8; 0],
}

#[repr(C)]
pub struct MtmdHelperGenAudio {
    _private: [u8; 0],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum MtmdGenAudioType {
    None = 0,
    Qwen3Tts = 1,
    PocketTts = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MtmdGenAudioInfo {
    pub audio_type: MtmdGenAudioType,
    pub sample_rate: i32,
    pub model_variant: *const c_char,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum MtmdHelperGenAudioOutputType {
    Pcm = 0,
    Wav = 1,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MtmdHelperGenAudioInput {
    pub seq_id: i32,
    pub prompt: *const c_char,
    pub prompt_len: usize,
    pub speaker_ref: *mut MtmdBitmap,
    pub lang: *const c_char,
    pub top_k: i32,
    pub top_p: f32,
    pub seed: u32,
    pub out_type: MtmdHelperGenAudioOutputType,
}

/// Mirrors llama.cpp's `mtmd_helper_bitmap_wrapper`: the decoded bitmap plus
/// an optional video context (non-null only for video inputs, which own the
/// frame storage the bitmap points into).
#[repr(C)]
pub struct MtmdHelperBitmapWrapper {
    pub bitmap: *mut MtmdBitmap,
    pub video_ctx: *mut MtmdHelperVideo,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MtmdHelperVideoInitParams {
    pub fps_target: c_float,
    pub ffmpeg_bin_dir: *const c_char,
    pub timestamp_interval_ms: i64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MtmdHelperInitOpt {
    pub video_params: MtmdHelperVideoInitParams,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtmdInputChunkType {
    Text = 0,
    Image = 1,
    Audio = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MtmdDecoderPos {
    pub t: u32,
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MtmdInputText {
    pub text: *const c_char,
    /// Length of `text` in bytes, excluding the trailing NUL. Added upstream by
    /// llama.cpp `4114ba18b` (`mtmd: fix silent prompt truncation on embedded
    /// NUL`); `mtmd_tokenize` reads it directly, so leaving it out makes the
    /// native side size the prompt from uninitialised padding.
    pub text_len: usize,
    pub add_special: bool,
    pub parse_special: bool,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MtmdContextParams {
    pub use_gpu: bool,
    /// `ggml_backend_dev_t`, an opaque device handle. Added upstream by
    /// llama.cpp `681c29d36` (`mtmd: add --mmproj-device argument`); omitting
    /// it shifts every following field and truncates the struct.
    pub device: *mut c_void,
    pub print_timings: bool,
    pub n_threads: c_int,
    pub image_marker: *const c_char,
    pub media_marker: *const c_char,
    pub flash_attn_type: c_int,
    pub warmup: bool,
    pub image_min_tokens: c_int,
    pub image_max_tokens: c_int,
    pub cb_eval: *mut c_void,
    pub cb_eval_user_data: *mut c_void,
    pub batch_max_tokens: c_int,
    pub progress_callback: MtmdProgressCallback,
    pub progress_callback_user_data: *mut c_void,
}
