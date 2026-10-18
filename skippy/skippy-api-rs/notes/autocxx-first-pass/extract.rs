// Extract of autocxx-gen 0.30.0 output for seed.rs (types and bridge fns only).
// Full output was 1342 lines plus 115 KB of generated C++.

                    #[repr(u8)]
                    #[derive(Clone, Hash, PartialEq, Eq)]
                    pub enum RuntimeStatus {
                        Ok = 0,
                        InvalidModel = 1,
                        InvalidSession = 2,
                        ModelInUse = 3,
                        AllocationFailed = 4,
                        LoadFailed = 5,
                        SessionCreationFailed = 6,
                        UnsupportedConfiguration = 7,
                        CapacityExceeded = 8,
                        InvalidBackendResult = 9,
                        InternalError = 10,
                    }
                    #[repr(u8)]
                    #[derive(Clone, Hash, PartialEq, Eq)]
                    pub enum WorkloadStatus {
                        Ok = 0,
                        InvalidSession = 1,
                        UnsupportedWorkload = 2,
                        UnsupportedClassifierCount = 3,
                        EmptyInput = 4,
                        TooManyTokens = 5,
                        TokenizationFailed = 6,
                        NoEmbeddingDimensions = 7,
                        OutputTooSmall = 8,
                        AllocationFailed = 9,
                        DecodeFailed = 10,
                        EncodeFailed = 11,
                        NoDecoderStartToken = 12,
                        NoOutput = 13,
                        DimensionMismatch = 14,
                        InternalError = 15,
                    }
                    #[repr(i8)]
                    #[derive(Clone, Hash, PartialEq, Eq)]
                    pub enum WorkloadKind {
                        CausalGeneration = 0,
                        Embedding = 1,
                        Rerank = 2,
                        EncoderDecoder = 3,
                    }
                    #[repr(i8)]
                    #[derive(Clone, Hash, PartialEq, Eq)]
                    pub enum NativePooling {
                        Unspecified = -1,
                        None = 0,
                        Mean = 1,
                        Cls = 2,
                        Last = 3,
                        Rank = 4,
                    }
                    #[repr(C)]
                    pub struct SessionConfig {
                        pub max_tokens: ::std::os::raw::c_uint,
                    }
                    #[repr(C)]
                    pub struct WorkloadInfo {
                        pub kind: root::skippy::ffi::WorkloadKind,
                        pub pooling: root::skippy::ffi::NativePooling,
                        pub embedding_pooling: root::skippy::ffi::NativePooling,
                        pub output_dimensions: usize,
                        pub classifier_outputs: usize,
                        pub has_encoder: bool,
                        pub has_decoder: bool,
                    }
                    #[repr(C)]
                    pub struct ModelResult {
                        pub status: root::skippy::ffi::RuntimeStatus,
                        pub model: root::std::unique_ptr<root::skippy::Model>,
                    }
                    #[repr(C)]
                    pub struct SessionResult {
                        pub status: root::skippy::ffi::RuntimeStatus,
                        pub session: root::std::unique_ptr<root::skippy::Session>,
                    }
                    #[repr(C)]
                    pub struct WorkloadResult {
                        pub status: root::skippy::ffi::RuntimeStatus,
                        pub info: root::skippy::ffi::WorkloadInfo,
                    }
                    #[repr(C)]
                    pub struct EmbedResult {
                        pub status: root::skippy::ffi::WorkloadStatus,
                        pub dimensions: usize,
                    }
                    #[repr(C)]
                    pub struct RerankResult {
                        pub status: root::skippy::ffi::WorkloadStatus,
                        pub score: f32,
                        pub token_count: usize,
                    }
                    #[repr(C)]
                    pub struct EncodeResult {
                        pub status: root::skippy::ffi::WorkloadStatus,
                        pub decoder_start: ::std::os::raw::c_int,
                    }
                    #[repr(C)]
                    pub struct Bridge {
                        impl_: root::std::unique_ptr<root::skippy::ffi::Bridge_Impl>,
                    }

// ... cxx::bridge section:
            type Bridge = super::skippy::ffi::Bridge;
            pub unsafe fn load_model_autocxx_wrapper_0x9389956aae39d48d(
                autocxx_gen_this: Pin<&mut Bridge>,
                path: *const c_char,
                length: usize,
                placement_return_type: *mut ModelResult,
            );
            #[namespace = "skippy::ffi"]
            pub fn unload_model(self: Pin<&mut Bridge>, model: Pin<&mut Model>) -> RuntimeStatus;
            pub unsafe fn create_session_autocxx_wrapper_0x9389956aae39d48d(
                autocxx_gen_this: Pin<&mut Bridge>,
                model: &Model,
                config: SessionConfig,
                placement_return_type: *mut SessionResult,
            );
            #[namespace = "skippy::ffi"]
            pub fn release_session(
                self: Pin<&mut Bridge>,
                session: Pin<&mut Session>,
            ) -> RuntimeStatus;
            #[namespace = "skippy::ffi"]
            pub fn workload(self: &Bridge, session: &Session) -> WorkloadResult;
            #[namespace = "skippy::ffi"]
            pub unsafe fn embed(
                self: Pin<&mut Bridge>,
                session: &Session,
                tokens: *mut c_int,
                count: usize,
                output: *mut f32,
                capacity: usize,
            ) -> EmbedResult;
            #[namespace = "skippy::ffi"]
            pub unsafe fn rerank(
                self: Pin<&mut Bridge>,
                session: &Session,
                query: *const c_char,
                query_length: usize,
                document: *const c_char,
                document_length: usize,
            ) -> RerankResult;
            #[namespace = "skippy::ffi"]
            pub unsafe fn encode(
                self: Pin<&mut Bridge>,
                session: &Session,
                tokens: *mut c_int,
                count: usize,
            ) -> EncodeResult;
            pub unsafe fn new_autocxx_autocxx_wrapper_0x9389956aae39d48d(
                autocxx_gen_this: *mut Bridge,
            );
            pub unsafe fn Bridge_destructor_autocxx_wrapper_0x9389956aae39d48d(
                autocxx_gen_this: *mut Bridge,
            );
            #[namespace = "skippy::ffi"]
            pub fn new_bridge() -> UniquePtr<Bridge>;

// ... ModelResult/SessionResult became opaque (no field access from Rust):
            #[repr(transparent)]
            pub struct ModelResult {
                _hidden_contents: ::core::cell::UnsafeCell<
                    ::core::mem::MaybeUninit<super::super::bindgen::root::skippy::ffi::ModelResult>,
                >,
            }
            unsafe impl cxx::ExternType for ModelResult {
                type Id = cxx::type_id!("skippy::ffi::ModelResult");
                type Kind = cxx::kind::Opaque;
            }
            #[repr(transparent)]
            pub struct SessionResult {
                _hidden_contents: ::core::cell::UnsafeCell<
                    ::core::mem::MaybeUninit<
                        super::super::bindgen::root::skippy::ffi::SessionResult,
                    >,
                >,
            }
            unsafe impl cxx::ExternType for SessionResult {
                type Id = cxx::type_id!("skippy::ffi::SessionResult");
                type Kind = cxx::kind::Opaque;
            }
