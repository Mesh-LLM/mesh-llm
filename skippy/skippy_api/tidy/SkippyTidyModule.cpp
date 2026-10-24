// The skippy_api house rules that types cannot carry, as clang-tidy checks.
// Load with: clang-tidy --load=SkippyTidyChecks.{dylib,so} -checks='skippy-*'

#include "clang-tidy/ClangTidyModule.h"
#include "clang-tidy/ClangTidyModuleRegistry.h"

#include "checks/BoundaryIsolationCheck.h"
#include "checks/BridgeBarrierCheck.h"
#include "checks/ConstNativeMutationCheck.h"
#include "checks/ExplicitNarrowingCheck.h"
#include "checks/NoAllocInOomHandlerCheck.h"
#include "checks/NoExpectedVoidCheck.h"
#include "checks/NoFailureOptionalCheck.h"
#include "checks/NoMutexCheck.h"
#include "checks/NoLockAcrossNativeCallCheck.h"
#include "checks/NoProcessWideStateCheck.h"
#include "checks/NoRawPointerCheck.h"
#include "checks/NoThrowingAccessorCheck.h"
#include "checks/NoRawLoopsCheck.h"
#include "checks/NoVoidSilencingCheck.h"
#include "checks/OptionalErrorPolarityCheck.h"
#include "checks/PrecededByCheck.h"
#include "checks/RustStringLossyCheck.h"
#include "checks/SingleCallSiteCheck.h"

namespace clang::tidy::skippy {

class SkippyModule : public ClangTidyModule {
public:
    void addCheckFactories(ClangTidyCheckFactories& Factories) override {
        Factories.registerCheck<NoExpectedVoidCheck>("skippy-no-expected-void");
        Factories.registerCheck<NoFailureOptionalCheck>("skippy-no-failure-optional");
        Factories.registerCheck<NoMutexCheck>("skippy-no-mutex");
        Factories.registerCheck<NoRawPointerCheck>("skippy-no-raw-pointer");
        Factories.registerCheck<NoThrowingAccessorCheck>("skippy-no-throwing-accessor");
        Factories.registerCheck<BoundaryIsolationCheck>("skippy-boundary-isolation");
        Factories.registerCheck<NoRawLoopsCheck>("skippy-no-raw-loops");
        Factories.registerCheck<OptionalErrorPolarityCheck>("skippy-optional-error-polarity");
        Factories.registerCheck<PrecededByCheck>("skippy-preceded-by");
        Factories.registerCheck<NoAllocInOomHandlerCheck>("skippy-no-alloc-in-oom-handler");
        Factories.registerCheck<BridgeBarrierCheck>("skippy-bridge-barrier");
        Factories.registerCheck<SingleCallSiteCheck>("skippy-single-call-site");
        Factories.registerCheck<NoLockAcrossNativeCallCheck>("skippy-no-lock-across-native-call");
        Factories.registerCheck<NoVoidSilencingCheck>("skippy-no-void-silencing");
        Factories.registerCheck<NoProcessWideStateCheck>("skippy-no-process-wide-state");
        Factories.registerCheck<RustStringLossyCheck>("skippy-rust-string-lossy");
        Factories.registerCheck<ExplicitNarrowingCheck>("skippy-explicit-narrowing");
        Factories.registerCheck<ConstNativeMutationCheck>("skippy-const-native-mutation");
    }

    // Defaults for this tree; .clang-tidy overrides any of them.
    ClangTidyOptions getModuleOptions() override {
        ClangTidyOptions Options;
        auto& Opts = Options.CheckOptions;
        Opts["skippy-preceded-by.Call"] = "llama_memory_clear";
        Opts["skippy-preceded-by.Before"] = "llama_synchronize";
        Opts["skippy-single-call-site.Functions"] =
            "llama_model_n_cls_out;llama_n_ctx;llama_n_batch;llama_n_ubatch;llama_n_seq_max";
        Opts["skippy-single-call-site.AllowedCallers"] =
            "skippy::llama::ClassifierHead::detect;skippy::llama::ContextCapacity::ContextCapacity;"
            "skippy::llama::BatchLimit::BatchLimit";
        Opts["skippy-single-call-site.Reason"] =
            "native getters with defaults or padded values are read once, behind a type that "
            "records what they mean (ClassifierHead, ContextCapacity, BatchLimit)";
        Opts["skippy-bridge-barrier.FileRegex"] = "skippy-sys/cpp/bridge\\.cpp$";
        Opts["skippy-bridge-barrier.Barriers"] = "guarded;guarded_message";
        Opts["skippy-no-lock-across-native-call.NativeRegex"] = "^::(llama_|ggml_|skippy_)";
        Opts["skippy-const-native-mutation.NativeRegex"] = "^::(llama_|ggml_|skippy_)";
        Opts["skippy-no-process-wide-state.AllowedInFunctions"] = "skippy::detail::next_generation";
        Opts["skippy-no-process-wide-state.AllowedTypes"] = "::std::once_flag";
        Opts["skippy-rust-string-lossy.AsciiProducers"] = "::hex_of$";
        Opts["skippy-explicit-narrowing.AllowedFunctions"] = "skippy::narrow;skippy::checked_mul";
        Opts["skippy-explicit-narrowing.AllowedFiles"] = "narrow\\.h$";
        Opts["skippy-boundary-isolation.AllowedDirectories"] =
            "src/llama/;include/llama/;/tools/;/benchmarks/;skippy-sys/cpp/";
        Opts["skippy-boundary-isolation.NativeRegex"] = "(^|::)(llama_|ggml_|skippy_)";
        Opts["skippy-boundary-isolation.NativeHeaderRegex"] =
            "^(llama|llama-cpp|ggml|ggml-alloc|ggml-backend|gguf|skippy)\\.h$";
        Opts["skippy-no-failure-optional.ErrorTypeRegex"] = "(Error|Failure)$";
        Opts["skippy-no-raw-pointer.AllowedNames"] = "(^|::)(native|argv|optarg)$";
        return Options;
    }
};

} // namespace clang::tidy::skippy

namespace clang::tidy {

static ClangTidyModuleRegistry::Add<skippy::SkippyModule>
    SkippyModuleRegistration("skippy-module", "skippy_api house rules (AGENTS.md) as checks.");

// Keeps the module's object file in a statically linked build.
volatile int SkippyModuleAnchorSource = 0; // NOLINT

} // namespace clang::tidy
