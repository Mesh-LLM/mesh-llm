#!/usr/bin/env bash
set -euo pipefail

# Deterministically binpack Cargo workspace crates for CI tests. Workspace
# membership comes from cargo metadata so a new crate cannot be omitted by a
# stale hand-maintained list.

usage() {
  echo 'usage: plan-test-batches.sh (--all | --crates-json JSON) [--bins N]' >&2
}

mode=""
crates_json="[]"
bins=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      mode="all"
      shift
      ;;
    --crates-json)
      mode="json"
      crates_json="${2:-}"
      shift 2
      ;;
    --bins)
      bins="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$mode" ]] || ! [[ "$bins" =~ ^[1-9][0-9]*$ ]]; then
  usage
  exit 2
fi

workspace_json=$(cargo metadata --format-version=1 --no-deps | jq -c '
  .workspace_members as $members
  | [.packages[] | select(.id as $id | $members | index($id)) | .name]
')

if [[ "$mode" == "all" ]]; then
  crates_json="$workspace_json"
else
  jq -e 'type == "array" and all(.[]; type == "string")' >/dev/null <<<"$crates_json"
  unknown=$(jq -cn --argjson requested "$crates_json" --argjson workspace "$workspace_json" \
    '$requested - $workspace')
  if [[ "$unknown" != "[]" ]]; then
    echo "test batch request contains non-workspace crates: $unknown" >&2
    exit 2
  fi
fi

python3 - "$bins" "$crates_json" <<'PY'
import json
import sys

bin_count = int(sys.argv[1])
crates = list(dict.fromkeys(json.loads(sys.argv[2])))

# Approximate test/link cost. Unknown crates deliberately receive weight 1 so
# newly added workspace members are scheduled without planner maintenance.
weights = {
    "mesh-llm": 10,
    "mesh-llm-host-runtime": 10,
    "mesh-llm-embedded-runtime": 8,
    "mesh-llm-client": 6,
    "skippy-runtime": 5,
    "skippy-server": 5,
    "model-artifact": 4,
    "model-hf": 4,
    "openai-frontend": 4,
    "skippy-bench": 4,
    "skippy-correctness": 4,
    "skippy-quantize": 4,
    "mesh-llm-api-server": 3,
    "mesh-llm-gpu-bench": 3,
    "mesh-llm-system": 3,
    "skippy-prompt": 3,
}

indexed = [(crate, weights.get(crate, 1), index) for index, crate in enumerate(crates)]
indexed.sort(key=lambda item: (-item[1], item[2], item[0]))

batches = [{"idx": index, "weight": 0, "crates": []} for index in range(bin_count)]
for crate, weight, _ in indexed:
    target = min(batches, key=lambda batch: (batch["weight"], batch["idx"]))
    target["crates"].append(crate)
    target["weight"] += weight

print(json.dumps([batch for batch in batches if batch["crates"]], separators=(",", ":")))
PY
