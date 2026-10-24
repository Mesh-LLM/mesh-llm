use autocxx::prelude::*;

include_cpp! {
    #include "seed.h"
    safety!(unsafe_ffi)
    generate!("skippy::ffi::Bridge")
    generate!("skippy::ffi::new_bridge")
    generate!("skippy::ffi::RuntimeStatus")
    generate!("skippy::ffi::WorkloadStatus")
    generate!("skippy::ffi::WorkloadKind")
    generate!("skippy::ffi::NativePooling")
    generate_pod!("skippy::ffi::SessionConfig")
    generate_pod!("skippy::ffi::WorkloadInfo")
    generate_pod!("skippy::ffi::WorkloadResult")
    generate_pod!("skippy::ffi::EmbedResult")
    generate_pod!("skippy::ffi::RerankResult")
    generate_pod!("skippy::ffi::EncodeResult")
    generate!("skippy::ffi::ModelResult")
    generate!("skippy::ffi::SessionResult")
}
