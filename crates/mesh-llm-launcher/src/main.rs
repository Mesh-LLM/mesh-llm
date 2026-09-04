#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod local_api;
mod tray;

fn main() {
    if std::env::args().nth(1).as_deref() == Some("serve") {
        run_embedded_mesh();
        return;
    }
    tray::run();
}

fn run_embedded_mesh() {
    const STACK_SIZE: usize = 8 * 1024 * 1024;
    mesh_llm::configure_hf_tls_provider();
    let exit_code = std::thread::Builder::new()
        .name("mesh-llm-main".to_string())
        .stack_size(STACK_SIZE)
        .spawn(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_stack_size(STACK_SIZE)
                .build()
                .expect("build Mesh Tokio runtime")
                .block_on(mesh_llm::run_main())
        })
        .expect("spawn Mesh application thread")
        .join()
        .unwrap_or_else(|payload| std::panic::resume_unwind(payload));
    std::process::exit(exit_code);
}
