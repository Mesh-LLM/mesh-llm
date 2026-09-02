#!/usr/bin/env bash

# Classify one family-certification lane from its terminal evidence. Keep the
# patterns tied to emitted failure messages: command lines are recorded in the
# same log and contain option names such as --startup-timeout-secs, which are
# not evidence that a timeout occurred.
classify_family_outcome() {
  local status="$1"
  local log="$2"
  local note="$3"
  if [[ "$status" == "pass" ]]; then
    printf 'pass\n'
    return
  fi
  if [[ "$status" == "skipped" ]]; then
    printf 'skipped\n'
    return
  fi

  local evidence="$note"
  if [[ -n "$log" && -f "$log" ]]; then
    evidence+=$'\n'
    evidence+="$(tail -n 240 "$log" | sed '/^+ /d')"
  fi
  if grep -Eqi 'timed out|did not become ready|deadline exceeded' <<<"$evidence"; then
    printf 'timeout\n'
  elif grep -Eqi 'unsupported:|not supported for this model architecture|unsupported model architecture' <<<"$evidence"; then
    printf 'unsupported\n'
  elif grep -Eqi 'missing tensor|tensor .* not found|failed to load model|model artifact.*invalid|invalid model artifact' <<<"$evidence"; then
    printf 'model-invalid\n'
  elif grep -Eqi 'mismatch|did not match|matches[=:][[:space:]]*false|token.*different' <<<"$evidence"; then
    printf 'mismatch\n'
  elif grep -Eqi 'no such file or directory|does not exist|requires --|corpus.*(missing|not found)|failed to resolve|command not found|required command not found' <<<"$evidence"; then
    printf 'harness\n'
  else
    printf 'runtime-error\n'
  fi
}
