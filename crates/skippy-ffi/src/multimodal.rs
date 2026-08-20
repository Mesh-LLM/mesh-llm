use std::ffi::{c_char, c_int, c_void};

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
    pub add_special: bool,
    pub parse_special: bool,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MtmdContextParams {
    pub use_gpu: bool,
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
