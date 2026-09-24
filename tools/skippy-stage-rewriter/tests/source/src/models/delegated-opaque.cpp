struct ggml_tensor {};
struct model_type {
  ggml_tensor *tok_embd;
};

struct model_delegated_stacks {
  struct graph {
    graph(const model_type &model);
    ggml_tensor *build_inp_embd(ggml_tensor *);
    ggml_tensor *build_stack(ggml_tensor *cur, int slot_base) const;
    int stack_layers;
  };
};

ggml_tensor *run_blocks(ggml_tensor *cur, int count);

// No layer-bounded loop anywhere in the constructor's call closure: this
// shape must keep its exact refusal.
model_delegated_stacks::graph::graph(const model_type &model) {
  ggml_tensor *inpL = build_inp_embd(model.tok_embd);
  for (int cycle = 0; cycle < 3; ++cycle) {
    inpL = run_blocks(inpL, cycle);
  }
}
