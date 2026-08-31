#![recursion_limit = "256"]

/// Default MeshLLM application and Tokio worker thread stack size: 8 MB.
///
/// The standard Tokio default is 2 MB, which is too small for several spawned
/// futures in this codebase (startup_local_model_loop, api_proxy,
/// nostr_rediscovery, publish_loop, handle_mesh_request). Box::pin on the
/// largest spawn sites keeps most futures on the heap, but 8 MB provides a
/// safety margin for any remaining large futures or deep call chains. This
/// matches the macOS main-thread default.
///
/// Override with MESH_TOKIO_STACK_SIZE for CI or debugging (e.g. set to
/// 1048576 to catch regressions with a 1 MB clamp).
const DEFAULT_THREAD_STACK_SIZE: usize = 8 * 1024 * 1024;

fn main() {
    mesh_llm_host_runtime::configure_hf_tls_provider();
    prepare_model_download_directories();
    configure_metal_pipeline_cache();

    let worker_stack_size = configured_worker_stack_size();
    // MESH_TOKIO_STACK_SIZE intentionally constrains only Tokio workers. CI
    // uses a small worker stack to detect oversized spawned futures, while the
    // application future still needs a stable stack across operating systems.
    let exit_code = run_on_application_thread(DEFAULT_THREAD_STACK_SIZE, move || {
        run_main_with_runtime(worker_stack_size)
    });
    std::process::exit(exit_code);
}

/// Configure the ggml Metal pipeline cache before the Tokio runtime is
/// constructed and before any native runtime library can be loaded. The
/// env mutation is only sound while the process is single-threaded; see
/// `mesh_llm_host_runtime::configure_metal_pipeline_cache`.
fn configure_metal_pipeline_cache() {
    // SAFETY: `main` calls this synchronously before creating the application
    // thread or Tokio runtime, so no other thread can access the environment.
    unsafe { mesh_llm_host_runtime::configure_metal_pipeline_cache() };
}

fn configured_worker_stack_size() -> usize {
    std::env::var("MESH_TOKIO_STACK_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_THREAD_STACK_SIZE)
}

fn run_main_with_runtime(stack_size: usize) -> i32 {
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.enable_all();
    builder.thread_stack_size(stack_size);

    let runtime = builder.build().expect("build tokio runtime");
    runtime.block_on(mesh_llm::run_main())
}

fn run_on_application_thread<T: Send + 'static>(
    stack_size: usize,
    run: impl FnOnce() -> T + Send + 'static,
) -> T {
    std::thread::Builder::new()
        .name("mesh-llm-main".to_string())
        .stack_size(stack_size)
        .spawn(run)
        .expect("spawn MeshLLM application thread")
        .join()
        .unwrap_or_else(|payload| std::panic::resume_unwind(payload))
}

fn prepare_model_download_directories() {
    let prepared =
        match mesh_llm_host_runtime::command_support::models::prepare_download_directories() {
            Ok(prepared) => prepared,
            Err(error) => {
                eprintln!(
                    "⚠ Unable to prepare model download directories: {error:#}. Model downloads may fail; set MESH_LLM_DATA_DIR to a writable directory."
                );
                return;
            }
        };
    for fallback in &prepared.fallbacks {
        eprintln!("⚠ {fallback}");
    }
    // SAFETY: This runs before the Tokio runtime is constructed, while the
    // process is still single-threaded.
    unsafe { prepared.apply_to_process_environment() };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn application_entrypoint_runs_off_the_os_main_thread() {
        let caller = std::thread::current().id();
        let worker =
            run_on_application_thread(DEFAULT_THREAD_STACK_SIZE, || std::thread::current().id());

        assert_ne!(caller, worker);
    }
}
