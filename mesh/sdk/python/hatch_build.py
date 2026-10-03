from __future__ import annotations

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Refuse to publish a pure-Python wheel without its native Mesh bridge."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        if self.target_name != "wheel":
            return

        generated = Path(self.root) / "src" / "meshllm" / "_generated"
        libraries = [
            generated / "libuniffi.dylib",
            generated / "libuniffi.so",
            generated / "uniffi.dll",
        ]
        if not any(library.is_file() for library in libraries):
            raise RuntimeError(
                "MeshLLM native library is missing; run scripts/build-native.sh before building a wheel"
            )

        build_data["pure_python"] = False
        build_data["infer_tag"] = True
