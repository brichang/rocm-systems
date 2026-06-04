#!/usr/bin/env bash
#
# Tests the "keep the nightly rows we need to build" jq filter in
# .github/workflows/hipfile-ci-toplevel.yml (Precheck job -> "Compute images to
# build" step). Given the nightly candidate matrix and the space-separated set of
# (platform-version) keys NOT yet published (i.e. the ones to build), the filter
# returns exactly those rows.
#
# select_to_build is a verbatim copy of the workflow's final jq. If you edit one,
# mirror the change here and re-run.
#
# Usage: bash projects/hipfile/util/test-build-set-filter.sh
#
set -uo pipefail  # not -e: keep going on failures

select_to_build() {
  local candidate="$1" to_build="$2"  # to_build: space-separated keys to build
  printf '%s' "$candidate" | jq -c --arg tb "$to_build" \
    '($tb | split(" ") | map(select(length > 0))) as $b
     | {include: [.include[]
         | select((.supported_platforms + "-" + .rocm_versions) as $k | $b | index($k))]}'
}

assert_eq() {
  local label="$1" actual="$2" expected="$3"
  local a_norm e_norm
  a_norm=$(printf '%s' "$actual"   | jq -cS '.include |= sort')
  e_norm=$(printf '%s' "$expected" | jq -cS '.include |= sort')
  if [ "$a_norm" = "$e_norm" ]; then
    printf '  PASS  %s\n        -> %s\n' "$label" "$a_norm"
  else
    printf '  FAIL  %s\n        expected: %s\n        got:      %s\n' \
      "$label" "$e_norm" "$a_norm"
  fi
}

echo "Test 1: single nightly row, all published (none to build) -> empty"
assert_eq "  output" \
  "$(select_to_build \
      '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"nightly"}]}' \
      '')" \
  '{"include":[]}'

echo
echo "Test 2: single nightly row to build -> kept"
assert_eq "  output" \
  "$(select_to_build \
      '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"nightly"}]}' \
      'ubuntu-nightly')" \
  '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"nightly"}]}'

echo
echo "Test 3: multiple nightly rows, only the unpublished one is built"
assert_eq "  output" \
  "$(select_to_build \
      '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"nightly"},{"supported_platforms":"rocky","rocm_versions":"nightly"}]}' \
      'rocky-nightly')" \
  '{"include":[{"supported_platforms":"rocky","rocm_versions":"nightly"}]}'

echo
echo "Test 4: multiple nightly rows, all to build -> both kept"
assert_eq "  output" \
  "$(select_to_build \
      '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"nightly"},{"supported_platforms":"rocky","rocm_versions":"nightly"}]}' \
      'ubuntu-nightly rocky-nightly')" \
  '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"nightly"},{"supported_platforms":"rocky","rocm_versions":"nightly"}]}'

echo
echo "Test 5: a to-build key not in the candidate set selects nothing"
assert_eq "  output" \
  "$(select_to_build \
      '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"nightly"}]}' \
      'rocky-nightly')" \
  '{"include":[]}'

echo
echo "Done."
