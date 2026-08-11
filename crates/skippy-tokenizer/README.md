# skippy-tokenizer

`skippy-tokenizer` defines the model-bound tokenizer capability contract used
by in-process Skippy consumers. It contains request, response, identity, limit,
and typed-error types plus the bounded batch-tokenization trait; runtime crates
provide the implementation backed by an already-loaded stage-zero model.
