from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Final


FLAT_IMPORT: Final = re.compile(r"^import '([^']+)'\s*(?:\r?\n)?$")
UNSUPPORTED_DIRECTIVE: Final = re.compile(r"^(?:mod(?:\s|$)|import\?)")


@dataclass(frozen=True, slots=True)
class JustfileImportError(Exception):
    path: Path
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


def read_justfile_source(path: Path) -> str:
    return _read_justfile_source(path.resolve(), ())


def _read_justfile_source(path: Path, stack: tuple[Path, ...]) -> str:
    if path in stack:
        cycle = " -> ".join(str(entry) for entry in (*stack, path))
        raise JustfileImportError(path=path, reason=f"import cycle: {cycle}")
    if not path.is_file():
        raise JustfileImportError(path=path, reason="missing import")

    source: list[str] = []
    next_stack = (*stack, path)
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        match = FLAT_IMPORT.fullmatch(line)
        if match is not None:
            imported_path = (path.parent / match.group(1)).resolve()
            source.append(_read_justfile_source(imported_path, next_stack))
        elif UNSUPPORTED_DIRECTIVE.match(line):
            raise JustfileImportError(path=path, reason=f"unsupported directive: {line.strip()}")
        else:
            source.append(line)
    return "".join(source)
