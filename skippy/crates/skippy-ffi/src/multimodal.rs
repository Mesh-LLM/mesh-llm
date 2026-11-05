//! Ergonomic aliases for the generated mtmd (multimodal) ABI types.

use crate::bindings;

pub type MtmdContext = bindings::mtmd::mtmd_context;
pub type MtmdBitmap = bindings::mtmd::mtmd_bitmap;
pub type MtmdInputChunks = bindings::mtmd::mtmd_input_chunks;
pub type MtmdHelperVideo = bindings::mtmd::mtmd_helper_video;
pub type MtmdHelperGenAudio = bindings::mtmd::mtmd_helper_gen_audio;
pub type MtmdGenAudioType = bindings::mtmd::mtmd_gen_audio_type;
pub type MtmdGenAudioInfo = bindings::mtmd::mtmd_gen_audio_info;
pub type MtmdHelperGenAudioOutputType = bindings::mtmd::mtmd_helper_gen_audio_outtype;
pub type MtmdHelperGenAudioInput = bindings::mtmd::mtmd_helper_gen_audio_inp;
pub type MtmdHelperBitmapWrapper = bindings::mtmd::mtmd_helper_bitmap_wrapper;
pub type MtmdHelperVideoInitParams = bindings::mtmd::mtmd_helper_video_init_params;
pub type MtmdHelperInitOpt = bindings::mtmd::mtmd_helper_init_opt;
pub type MtmdInputChunkType = bindings::mtmd::mtmd_input_chunk_type;
pub type MtmdDecoderPos = bindings::mtmd::mtmd_decoder_pos;
pub type MtmdInputText = bindings::mtmd::mtmd_input_text;
pub type MtmdContextParams = bindings::mtmd::mtmd_context_params;
