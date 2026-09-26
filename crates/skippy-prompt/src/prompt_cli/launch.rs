pub fn prompt_repl(_args: PromptArgs) -> Result<()> {
    bail!(
        "skippy-prompt topology launch is disabled in mesh-llm; use `skippy-prompt binary` against a mesh-managed first stage"
    )
}

