"""Shared source fingerprinting, native binary cache validation and atomic builds.

This module only manages artifacts. Numerical code and backend-specific compiler
flags remain next to their CPU/CUDA implementations.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path


def source_fingerprint(directory: Path, extensions: set[str]) -> str:
    """Hash entry files and headers in a fixed order; header edits invalidate caches."""
    digest = hashlib.sha256()
    for member in sorted(directory.iterdir()):
        if member.suffix in extensions:
            digest.update(member.name.encode())
            digest.update(member.read_bytes())
    return digest.hexdigest()


def cached_binary_matches(output: Path, build: dict) -> bool:
    """Reuse a binary only if both its build description and bytes still match."""
    try:
        document = json.loads(output.with_suffix(".json").read_text())
        return (
            document.get("build") == build
            and output.is_file()
            and document.get("library_sha256")
            == hashlib.sha256(output.read_bytes()).hexdigest()
        )
    except (OSError, json.JSONDecodeError):
        return False


def compile_shared_library(source: Path, output: Path, build: dict) -> Path:
    """Compile to a temporary path and atomically install the binary and manifest."""
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.so")
    subprocess.run(
        [build["compiler"], *build["flags"], str(source), "-o", str(temporary)],
        check=True,
    )
    os.replace(temporary, output)
    document = {
        "build": build,
        "library_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    manifest = output.with_suffix(".json")
    temporary_manifest = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
    temporary_manifest.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_manifest, manifest)
    return output
