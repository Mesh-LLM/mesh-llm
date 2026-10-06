# Mixed prefill/decode A/B

Use `cargo xtool automation waiting-prefix mixed-run --input /absolute/matrix.json --output-directory /absolute/fresh-output` with the current typed mixed matrix schema. `mixed-plan` emits workload/config declarations; it does not execute an arm. Supply exact binary, native-build tree, model pins and reviewed old/new static profiles. The original synthetic shape is rounds8, anchors4, prefills8, anchor blocks8, prefill blocks256, output128/8, delay100ms, stagger5ms, lanes12, batch1024/ubatch256 and adaptive start/step/max256. Both local and split topologies retain separate owned cells.

`comparison.json` retains cells and comparison observations even when terminal cancellation, deadline or signal restoration fails. Such a result has status `mixed_matrix_failed`, `orchestration_completed:false` and a nonzero command exit. A clean terminal result is `mixed_matrix_completed`; its orchestration success does not set `comparison.qualified` or establish model/scheduler qualification. `report.md` states the terminal outcome above retained metrics. Typed final phase capture, full usage, output parity, owned cleanup and model custody remain separate required evidence.

Orchestration completion does not qualify real model behavior, runtime ABI, native scheduling performance or hosted execution.
