#[tokio::main]
async fn main() -> anyhow::Result<()> {
    mesh_llm_image_guards::run(mesh_llm_image_guards::Role::Provider).await
}
