struct ggml_tensor {};
struct model_type {};

// Shared graph helpers live on the base context and are defined in another
// translation unit; they are library code, not the model's decoder body.
struct llm_graph_context {
  ggml_tensor *build_norm(ggml_tensor *);
  ggml_tensor *build_attn(ggml_tensor *);
};

struct model_auxiliary_shared {
  struct graph_head : public llm_graph_context {
    graph_head(const model_type &model);
    ggml_tensor *build_decoder_body(const model_type &);
    ggml_tensor *t_h_nextn;
  };
};

model_auxiliary_shared::graph_head::graph_head(const model_type &model) {
  ggml_tensor *cur = build_norm(t_h_nextn);
  cur = build_attn(cur);
  cur = build_decoder_body(model);
  // A repeated call to the same remote helper still describes one remote body.
  cur = build_decoder_body(model);
  (void) cur;
}
