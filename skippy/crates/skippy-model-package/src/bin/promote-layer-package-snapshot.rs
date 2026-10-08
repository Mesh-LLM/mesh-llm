fn main() -> anyhow::Result<()> {
    skippy_model_package::snapshot_promotion::run(
        &mut std::io::stdout().lock(),
        &mut std::io::stderr().lock(),
    )
}
