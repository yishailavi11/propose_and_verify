#!/usr/bin/env bash
# Install the pinned third-party source dependency under this checkout.
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cubql_root="$repo_root/external/cuBQL"
revision=20f9db19cdb160e2b83d8f05dfdb0437fe11fe24
mkdir -p "$repo_root/external"
if [[ ! -d "$cubql_root/.git" ]]; then
  git clone https://github.com/NVIDIA/cuBQL.git "$cubql_root"
fi
git -C "$cubql_root" fetch https://github.com/NVIDIA/cuBQL.git "$revision"
git -C "$cubql_root" checkout --detach "$revision"
printf 'export FAST_SYMMETRY_CUBQL_ROOT=%q\n' "$cubql_root"
