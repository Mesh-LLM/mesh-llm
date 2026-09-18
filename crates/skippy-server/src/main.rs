mod console;

use anyhow::Result;
use clap::Parser;

use skippy_server::cli::{Cli, Command};
use skippy_server::{
    binary_transport::serve_binary, config::example_config, frontend::serve_openai, http::serve,
};

#[tokio::main]
async fn main() -> Result<()> {
    console::install();
    let cli = Cli::parse();
    #[cfg(feature = "dynamic-native-runtime")]
    if !matches!(&cli.command, Command::ExampleConfig) {
        skippy_server::native_runtime::load_local_native_runtime(&cli.native_runtime)?;
    }
    match cli.command {
        Command::Serve(args) => serve(args).await,
        Command::ServeBinary(args) => serve_binary(args).await,
        Command::ServeOpenAi(args) => serve_openai(args).await,
        Command::ExampleConfig => console::write_json(&example_config()),
    }
}
