#!/usr/bin/env bash
# The reviewer's whole check in one command (no API key needed):
#   ./check.sh
# Builds the image, then runs the tests, the retrieval scorecard and the end-to-end
# evaluation with the fake LLM, inside Docker. Prints PASS/FAIL for each step.
set -u
cd "$(dirname "$0")"

run_step() {
  local name="$1"; shift
  echo; echo "=== $name ==="
  if docker compose run --rm app "$@"; then
    results+=("PASS  $name")
  else
    results+=("FAIL  $name"); failed=1
  fi
}

results=(); failed=0
echo "=== building the image (first time: a few minutes) ==="
docker compose build || { echo "FAIL  docker compose build"; exit 1; }

run_step "tests (fake LLM)"                       pytest -q
run_step "retrieval scorecard (90 questions)"     python -m eval.run_retrieval
run_step "end-to-end evaluation (fake LLM)"       python -m eval.run_e2e --offline

echo; echo "=== summary ==="
printf '%s\n' "${results[@]}"
echo
echo "Note: the fake LLM gives canned answers, so the offline pass rate is not the system's score;"
echo "that step checks that every question runs end to end. The real-LLM result (83/90) is in"
echo "eval/results/e2e.jsonl and EVALUATION.md; rerun it with GROQ_API_KEY set: python -m eval.run_e2e"
exit "$failed"
