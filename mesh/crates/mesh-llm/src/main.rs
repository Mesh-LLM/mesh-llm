#![recursion_limit = "256"]

use std::io::Write;

use mesh_llm::DEFAULT_THREAD_STACK_SIZE;

fn main() {
    if let Some(code) = mesh_llm::run_build_contract_if_requested() {
        std::process::exit(code);
    }

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
    let mut err = mesh_llm_events::console_err();
    let prepared =
        match mesh_llm_host_runtime::command_support::models::prepare_download_directories() {
            Ok(prepared) => prepared,
            Err(error) => {
                let _ = writeln!(
                    err,
                    "⚠ Unable to prepare model download directories: {error:#}. Model downloads may fail; set MESH_LLM_DATA_DIR to a writable directory."
                );
                return;
            }
        };
    for fallback in &prepared.fallbacks {
        let _ = writeln!(err, "⚠ {fallback}");
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
