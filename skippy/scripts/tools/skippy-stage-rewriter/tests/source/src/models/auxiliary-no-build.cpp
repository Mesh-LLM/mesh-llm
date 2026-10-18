struct ggml_tensor {};
struct llm_graph_params {};
struct model_type {
  ggml_tensor *tok_embd;
};
struct no_build {};

struct llm_build_base {
  llm_build_base() = default;
  explicit llm_build_base(int params) {(void) params;}
};

// A helper-only graph variant: the empty constructor only constructs the base
// context so a sibling builder (here a combined trunk/MTP head) can reuse its
// helpers. It builds no graph of its own, so it owns no block annotation.
struct model_helper_only {
  struct graph : llm_build_base {
    graph(const model_type &model, const llm_graph_params &params);
    graph(const model_type &model, const llm_graph_params &params, no_build);

    ggml_tensor *build_inp_embd(ggml_tensor *);
    ggml_tensor *build_layer(ggml_tensor *, int);
    void cb(ggml_tensor *, const char *, int);

    int n_layer = 2;
  };
};

model_helper_only::graph::graph(const model_type &, const llm_graph_params &, no_build) :
    llm_build_base(0) {
}

model_helper_only::graph::graph(const model_type &model, const llm_graph_params &) {
  ggml_tensor *inpL = build_inp_embd(model.tok_embd);
  cb(inpL, "model.input_embed", -1);

  for (int il = 0; il < n_layer; ++il) {
    inpL = build_layer(inpL, il);
    cb(inpL, "l_out", il);
  }
}
