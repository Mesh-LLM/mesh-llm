"""Shared deterministic prompts for non-chat smoke and reference comparisons."""

EMBEDDING_INPUTS = (
    "search_query: distributed GPU inference",
    "search_document: GPUs share one language model over a mesh",
    "search_document: A recipe for tomato soup",
)

RERANK_QUERY = "distributed GPU inference"
RERANK_DOCUMENTS = (
    "GPUs share one language model over a mesh",
    "A recipe for tomato soup",
)

ENCODER_DECODER_PROMPT = "translate English to German: The house is wonderful."
