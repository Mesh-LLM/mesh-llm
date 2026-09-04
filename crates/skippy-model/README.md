# skippy-model

`skippy-model` owns validated source checkpoints: SafeTensors shard discovery,
immutable mapped tensor views, Hugging Face-to-llama tensor mapping and
transforms, tokenizer/config metadata, and canonical GGUF metadata generation.

Execution belongs to `skippy-runtime`; offline materialization belongs to
`skippy-quantize`. Both consume this crate so checkpoint-family mappings have a
single implementation.
