#!/usr/bin/env bash
#
# Tests the parsing/selection helpers in resolve-rocm-nightly.sh against
# fixtures, so the network-dependent resolver can be validated offline.
#
# Usage: bash projects/hipfile/util/test-resolve-rocm-nightly.sh
#
set -uo pipefail  # not -e: keep going on failures

# shellcheck source=./resolve-rocm-nightly.sh
. "$(dirname "${BASH_SOURCE[0]}")/resolve-rocm-nightly.sh"
set +e  # the sourced resolver enables -e; undo it for the test harness

assert_eq() {
    local label="$1" actual="$2" expected="$3"
    if [ "${actual}" = "${expected}" ]; then
        printf '  PASS  %s\n        -> %s\n' "${label}" "${actual}"
    else
        printf '  FAIL  %s\n        expected: %s\n        got:      %s\n' \
            "${label}" "${expected}" "${actual}"
    fi
}

# Sample /deb/ index, mixing dates and same-date run ids, with href + text
# duplicates as the real S3 listing produces.
read -r -d '' INDEX_FIXTURE <<'EOF'
<a href="20260530-26673017729/">20260530-26673017729/</a>
<a href="20260602-26796000000/">20260602-26796000000/</a>
<a href="20260601-26733368587/">20260601-26733368587/</a>
<a href="20260602-26796279962/">20260602-26796279962/</a>
EOF

# Sample Packages: the versioned -dev7.14 stanza must NOT be matched; only the
# bare amdrocm-runtime-dev meta-package supplies the version.
read -r -d '' PACKAGES_FIXTURE <<'EOF'
Package: amdrocm-runtime-dev7.14
Version: 7.14.0~20260602-26796279962
Architecture: amd64
Filename: pool/main/amdrocm-runtime-dev7.14_7.14.0~20260602-26796279962_amd64.deb

Package: amdrocm-runtime-dev
Version: 7.14.0~20260602-26796279962
Architecture: amd64
Filename: pool/main/amdrocm-runtime-dev_7.14.0~20260602-26796279962_amd64.deb
EOF

# A different pinned version, meta stanza appearing first.
read -r -d '' PACKAGES_FIXTURE_ALT <<'EOF'
Package: amdrocm-runtime-dev
Version: 7.15.1~20260701-30000000001
Architecture: amd64

Package: amdrocm-runtime-dev7.15
Version: 7.15.1~20260701-30000000001
Architecture: amd64
EOF

echo "Test 1: latest build picks the highest date"
assert_eq "  select_latest_build" \
    "$(printf '%s' "${INDEX_FIXTURE}" | select_latest_build)" \
    "20260602-26796279962"

echo
echo "Test 2: same-date tie-break picks the larger numeric run id"
read -r -d '' INDEX_TIE <<'EOF'
<a href="20260602-26796000000/">20260602-26796000000/</a>
<a href="20260602-9999999999/">20260602-9999999999/</a>
<a href="20260602-26796279962/">20260602-26796279962/</a>
EOF
assert_eq "  select_latest_build" \
    "$(printf '%s' "${INDEX_TIE}" | select_latest_build)" \
    "20260602-26796279962"

echo
echo "Test 3: clean version from meta-package (strips ~build, ignores -dev7.14)"
assert_eq "  extract_clean_version" \
    "$(printf '%s' "${PACKAGES_FIXTURE}" | extract_clean_version)" \
    "7.14.0"

echo
echo "Test 4: clean version when meta stanza is first"
assert_eq "  extract_clean_version" \
    "$(printf '%s' "${PACKAGES_FIXTURE_ALT}" | extract_clean_version)" \
    "7.15.1"

echo
echo "Done."
