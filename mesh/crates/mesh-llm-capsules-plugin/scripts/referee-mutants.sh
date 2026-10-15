#!/usr/bin/env bash
# Every referee implementation mutant must fail the referee parity run, and
# fail it on every case vectors/parity/referee/mutants.json lists for it.
# Each mutant is built first, so a compile error never passes as "caught".
#
#   scripts/referee-mutants.sh [feature ...]   (default: every mutant listed)
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
mutants="$root/vectors/parity/referee/mutants.json"
cd "$root"

features=("$@")
if [ ${#features[@]} -eq 0 ]; then
  while IFS= read -r f; do features+=("$f"); done < <(python3 -c 'import json,sys; [print(m["feature"]) for m in json.load(open(sys.argv[1]))["mutants"]]' "$mutants")
fi

status=0
for feature in "${features[@]}"; do
  cargo test --locked -p mesh-llm-capsules-plugin --bin capsules --features "$feature" --no-run -q
  if out=$(cargo test --locked -p mesh-llm-capsules-plugin --bin capsules --features "$feature" -- referee::parity::the_referee_answers_the_corpus_as_expected 2>&1); then
    echo "::error::the referee parity run passed a build with $feature"
    status=1
    continue
  fi
  missing=$(RUN_OUTPUT="$out" python3 - "$mutants" "$feature" <<'EOF'
import json, os, re, sys
mutant = next(m for m in json.load(open(sys.argv[1]))["mutants"] if m["feature"] == sys.argv[2])
reported = set(re.findall(r"^differs: (\S+)$", os.environ["RUN_OUTPUT"], re.M))
print(" ".join(sorted(set(mutant["must_fail"]) - reported)))
EOF
)
  if [ -n "$missing" ]; then
    echo "::error::$feature failed the run, but not on: $missing"
    status=1
  else
    echo "caught: $feature"
  fi
done
exit $status
