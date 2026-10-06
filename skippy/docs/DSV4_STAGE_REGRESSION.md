# DeepSeek V4 package-stage regression

The native suite registers `skippy_deepseek4_package_stage_replay_{1,2,3}`
against the small seeded four-layer DSV4 fixture. No production weights or
network downloads are required for these tests.

Unlike the no-allocation graph roster or fused-node context checks, these
cases use the public planner and `skippy_model_open_from_parts`. The harness
writes renamed GGUF shards, a metadata carrier, and selected source/terminal
payloads; it supplies the admitted tensor identities, activation bindings and
execution contracts to the runtime. Every interior cut is executed, including
cuts before the fixture's CSA and HCA layers (`[0, 0, 4, 128]`).

Each case prefills two tokens, then decodes two steps. It compares sampled
predictions and the complete vocabulary logits against an ordinary unsplit
reference with the same weights and configuration. The existing ordinary-stage
reference also checks replay logits and owned state. Logit comparisons reject
non-finite values and use `1e-5 * (1 + abs(reference))` tolerance.

Run the full native suite from the workspace root:

```sh
just skippy-native-tests cpu
```

The regression is carried by the existing native replay and DSV4 test-owner
patches (0027 and 0060), not a new terminal patch. It does not change the ABI.

## Limits

This is a small CPU numerical/package-admission contract, not a reproduction
of the reported v0.78.1 CUDA failure. It does not validate multi-GPU allocation,
Mesh coordinator error propagation, large contexts or concurrent lanes. Two
prefill tokens and two decode steps do not cross the HCA compression interval.
Three-stage chains are structurally checked by the existing planner harness;
this regression executes two-stage chains. Passing it must not be reported as
proof that the production 0731 Q4 deployment is fixed.
