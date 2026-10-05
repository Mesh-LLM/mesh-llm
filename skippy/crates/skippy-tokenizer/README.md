# skippy-tokenizer

`skippy-tokenizer` defines the model-bound tokenizer capability contract used
by in-process Skippy consumers. It contains identity, limits, typed errors, a
text REST facade, and a bounded structured encoder. Encoder pieces are either
ordinary bytes or opaque native control descriptors. Mesh does not interpret
control descriptors; a backend that cannot preserve one returns
`unsupported_input` instead of using replacement decoding.
