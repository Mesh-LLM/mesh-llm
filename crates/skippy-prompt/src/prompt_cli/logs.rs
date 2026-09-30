fn handle_prompt_error(
    error: anyhow::Error,
    interrupt: &Arc<PromptInterruptState>,
    prompt_index: usize,
) -> Result<()> {
    if interrupt.take_interrupt() {
        eprintln!();
        eprintln!("request {prompt_index}: interrupted");
        Ok(())
    } else {
        Err(error)
    }
}
