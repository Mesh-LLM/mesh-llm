# Native mounted-artifact certification

The native operator selects a complete mounted GGUF roster and delegates to the
existing pinned bootstrap, projector acquisition and native certification worker:

```bash
"$MESH_LLM_AUTOMATION_BIN" automation hf-certify job-worker operator \
  --input operator-input.json --output-directory /results/fresh-certification
```

Use the existing closed certification worker request for the pinned automation
executable, source revisions, explicit tool paths and digests, CPU standalone
Just build, projector identity/acquisition and optional file-credential receipt
export. Set the shared worker/bootstrap timeout to an allowance large enough for
roster observation, bootstrap and certification, for example 7200 seconds for a
planned job. Selection consumes this allowance; the nested worker receives only
the remaining allowance minus its outer cleanup reserve. It needs at least 30
seconds remaining. Tools must already be provisioned; this command does not run
apt, rustup or cargo install.

The operator adds `model_root`, relative `model_pattern`, `expected_parts` and
`mtp_draft`. Its nested `worker.certification` contains mode, projector pin,
layer_count, optional mtp_layer_count and ctx_size. It omits target_parts,
expected_parts and mtp_draft, because those worker fields are produced from the
observed mounted files. For an existing certification request, this data-only
projection demonstrates the operator schema:

```bash
jq --arg root /target --arg pattern 'nested/model-*.gguf' \
   --arg draft /draft/mtp.gguf \
   '{schema_version:1, model_root:$root, model_pattern:$pattern,
     expected_parts:2, mtp_draft:$draft,
     worker:del(.certification.target_parts,
                .certification.expected_parts,.certification.mtp_draft)}' \
   native-worker-input.json > operator-input.json
```

Relative roots and draft paths are resolved in the operator invocation's working
directory. Nested glob patterns and literal metacharacters in the root are
supported. Exactly the expected nonempty roster must match for MTP attachment;
each match and the draft must resolve to a regular GGUF file. Symlink-mounted
regular artifacts are allowed, including targets outside the logical mount root;
canonical paths and SHA-256 pins retain the sorted logical glob order through
worker admission. Canonical target names may have a different lexical order;
that must not change the native --model sequence. Multiple names resolving to one artifact are refused by existing identity
admission. Projector-only mode ignores the model root, pattern, expected count and
draft, as the original operation did.

The command prints the path to `certification-report.json` on success. This file
is the actual native JSON report, rather than a synthesized passing fixture. It
is an observation and must be paired with successful `operator.json` and matching
`worker/native-job-delivery.json`/`worker/native-job.json` receipts. A late signal,
deadline or finish failure makes the operator fail; earlier native observations
and child logs remain available. Outputs must be fresh, with an existing parent;
select another output directory to retry.

Optional receipt export uses an explicit private credential file. The operator
does not forward ambient publication credentials to children. Remote Jobs
submission, model acquisition, real native loader/model acceptance and hosted
receipt publication are separate operations. The mounted-path/operator fixtures
exercise inert local tools and reports and do not certify any real model.
