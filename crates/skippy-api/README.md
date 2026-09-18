# skippy-api

Shared Skippy model preparation. `SingleStageOptions` and a resolved `StageSourceIdentity` produce the same single-stage configuration for standalone and embedded callers. Model-family cache policy and checkpoint quantization/importance-matrix preparation live here. Product hooks, diagnostics and model discovery remain caller concerns.

This initial boundary does not yet own model acquisition, lifecycle orchestration or split graph admission.
