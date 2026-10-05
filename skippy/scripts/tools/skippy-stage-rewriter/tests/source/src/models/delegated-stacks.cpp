struct ggml_tensor {};
struct layer_type {};
struct model_type {
  ggml_tensor *tok_embd;
  int n_layers_per_stack;
  layer_type *layers;
};

struct model_delegated_stacks {
  struct graph {
    const model_type &model;
    graph(const model_type &model);
    ggml_tensor *build_inp_embd(ggml_tensor *);
    ggml_tensor *block(ggml_tensor *cur, int il) const;
    ggml_tensor *build_stack(ggml_tensor *cur, int slot_base) const;
  };
};

ggml_tensor *model_delegated_stacks::graph::build_stack(ggml_tensor *cur,
                                                        int slot_base) const {
  for (int il = 0; il < model.n_layers_per_stack; ++il) {
    const layer_type &layer = model.layers[slot_base + il];
    (void)layer;
    cur = block(cur, slot_base + il);
  }
  return cur;
}

model_delegated_stacks::graph::graph(const model_type &model) : model(model) {
  ggml_tensor *inpL = build_inp_embd(model.tok_embd);
  for (int cycle = 0; cycle < 2; ++cycle) {
    inpL = build_stack(inpL, cycle * model.n_layers_per_stack);
  }
}
