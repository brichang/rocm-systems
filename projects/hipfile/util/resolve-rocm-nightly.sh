#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Resolves the latest ROCm nightly build and the clean ROCm version it pins.
# Used in CI to target a reproducible nightly when building the hipFile CI image.
#
# ./resolve-rocm-nightly.sh
#
# Emits (GITHUB_OUTPUT form) on stdout:
#   nightly_build=<YYYYMMDD-gha_run_id>
#   nightly_rocm_version=<MAJOR.MINOR.PATCH>

set -euo pipefail

NIGHTLY_DEB_BASE="https://rocm.nightlies.amd.com/deb"

# Reads a /deb/ directory index on stdin and prints the latest build id.
# Build ids are YYYYMMDD-<gha_run_id>; the date dominates, the numeric run id
# breaks ties so same-day builds resolve to a single deterministic match.
select_latest_build() {
    # awk 'NR==1' (rather than head) consumes the whole stream so sort does not
    # receive SIGPIPE under pipefail.
    grep -oE '[0-9]{8}-[0-9]+' \
        | sort -u -t- -k1,1nr -k2,2nr \
        | awk 'NR==1 { print }'
}

# Reads a binary-amd64/Packages file on stdin and prints the clean
# MAJOR.MINOR.PATCH version of the amdrocm-runtime-dev meta-package (the nightly
# Debian version is e.g. 7.14.0~20260602-26796279962; everything from '~' on is
# stripped).
extract_clean_version() {
    # Read the whole file (no early exit) so the upstream feeder does not get
    # SIGPIPE under pipefail; keep the first matching version.
    awk '
        /^Package:/ { in_pkg = ($0 == "Package: amdrocm-runtime-dev") }
        in_pkg && /^Version:/ && ver == "" { ver = $2; sub(/~.*/, "", ver) }
        END { if (ver != "") print ver }
    '
}

main() {
    local index build packages version

    index="$(curl -fsSL "${NIGHTLY_DEB_BASE}/")"
    build="$(printf '%s' "${index}" | select_latest_build)"
    if [ -z "${build}" ]; then
        echo "resolve-rocm-nightly: no nightly build found at ${NIGHTLY_DEB_BASE}/" >&2
        exit 1
    fi

    packages="$(curl -fsSL "${NIGHTLY_DEB_BASE}/${build}/dists/stable/main/binary-amd64/Packages")"
    version="$(printf '%s' "${packages}" | extract_clean_version)"
    if [ -z "${version}" ]; then
        echo "resolve-rocm-nightly: could not determine ROCm version for build ${build}" >&2
        exit 1
    fi

    printf 'nightly_build=%s\n' "${build}"
    printf 'nightly_rocm_version=%s\n' "${version}"
}

# Only run when executed directly; sourcing (e.g. by the test) exposes the
# helper functions without invoking main.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi
