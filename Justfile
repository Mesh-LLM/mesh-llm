# Distributed LLM Inference — build & run tasks

llama_dir := env("MESH_LLM_LLAMA_DIR", ".deps/llama.cpp")
llama_build_root := env("MESH_LLM_LLAMA_BUILD_ROOT", ".deps/llama-build")
mesh_dir := "crates/mesh-llm"
ui_dir := "crates/mesh-llm-ui"
website_dir := "website"
home_dir := if os_family() == "windows" { env("USERPROFILE") } else { env("HOME") }
xdg_cache_dir := env("XDG_CACHE_HOME", home_dir / ".cache")
hf_home := env("HF_HOME", xdg_cache_dir / "huggingface")
models_dir := env("HF_HUB_CACHE", hf_home / "hub")
model := models_dir / "GLM-4.7-Flash-Q4_K_M.gguf"

# Build for the current platform.
default: build

import 'just/build.just'

import 'just/release-build.just'

import 'just/skippy.just'

import 'just/mesh.just'

import 'just/release-bundle.just'

import 'just/website-ui.just'

import 'just/ci.just'

import 'just/mesh-client.just'

import 'just/utilities.just'
