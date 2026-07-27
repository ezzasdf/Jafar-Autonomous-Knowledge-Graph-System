"""
Coding Workspace — persistent file management for Jafar's generated code.

Allows creating, reading, editing, and deleting workspace files:
  .py   Python scripts
  .txt  Text files (logs, readmes, notes)
  .csv  Tabular data
  .json Structured data

State flow:
  list_files() -> file tree
  create(path, content) -> file written
  read(path) -> content or None
  edit(path, old, new) -> patched
  delete(path) -> removed
"""

import csv
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
debug_logger = logging.getLogger(f"{__name__}.debug")

SUPPORTED_EXTENSIONS = {".py", ".txt", ".csv", ".json"}
MAX_FILE_SIZE = 1024 * 100  # 100KB per file


@dataclass
class FileEntry:
    path: str
    size: int
    created_at: float
    modified_at: float
    extension: str
    lines: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "size": self.size,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "extension": self.extension,
            "lines": self.lines,
        }


class CodingWorkspace:
    """Manages project files in an isolated workspace directory."""

    def __init__(self, workspace_dir: Optional[str] = None):
        base = Path(__file__).parent / "workspace"
        self.root = Path(workspace_dir) if workspace_dir else base
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Path resolution & validation
    # ------------------------------------------------------------------ #

    def _resolve(self, rel_path: str) -> Optional[Path]:
        path = (self.root / rel_path).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError:
            return None
        return path

    def _validate(self, path: Path) -> Optional[str]:
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return f"Unsupported extension '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        if path.is_file() and path.stat().st_size > MAX_FILE_SIZE:
            return f"File exceeds max size ({MAX_FILE_SIZE} bytes)"
        return None

    # ------------------------------------------------------------------ #
    #  CRUD
    # ------------------------------------------------------------------ #

    def create(self, rel_path: str, content: str = "") -> Dict[str, Any]:
        path = self._resolve(rel_path)
        if path is None:
            return {"success": False, "error": "Invalid or escaped path"}
        err = self._validate(path)
        if err:
            return {"success": False, "error": err}
        if path.exists():
            return {"success": False, "error": f"File already exists: {rel_path}"}

        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".json" and isinstance(content, str):
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                return {"success": False, "error": f"Invalid JSON: {e}"}
        if path.suffix.lower() == ".csv" and isinstance(content, str):
            try:
                list(csv.reader(io.StringIO(content)))
            except Exception as e:
                return {"success": False, "error": f"Invalid CSV: {e}"}

        path.write_text(content, encoding="utf-8")
        debug_logger.debug("Created workspace file: %s (%d bytes)", rel_path, len(content))
        return {"success": True, "path": rel_path, "size": len(content)}

    def read(self, rel_path: str) -> Dict[str, Any]:
        path = self._resolve(rel_path)
        if path is None:
            return {"success": False, "error": "Invalid or escaped path"}
        if not path.is_file():
            return {"success": False, "error": f"File not found: {rel_path}"}
        try:
            content = path.read_text(encoding="utf-8")
            return {"success": True, "path": rel_path, "content": content, "size": len(content)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def edit(self, rel_path: str, old_string: str, new_string: str) -> Dict[str, Any]:
        read_result = self.read(rel_path)
        if not read_result["success"]:
            return read_result
        content = read_result["content"]
        if old_string not in content:
            return {"success": False, "error": "old_string not found in file content"}
        new_content = content.replace(old_string, new_string, 1)
        return self._write(rel_path, new_content)

    def delete(self, rel_path: str) -> Dict[str, Any]:
        path = self._resolve(rel_path)
        if path is None:
            return {"success": False, "error": "Invalid or escaped path"}
        if not path.exists():
            return {"success": False, "error": f"File not found: {rel_path}"}
        path.unlink()
        debug_logger.debug("Deleted workspace file: %s", rel_path)
        self._cleanup_empty_dirs(path.parent)
        return {"success": True, "path": rel_path}

    def rename(self, rel_path: str, new_rel_path: str) -> Dict[str, Any]:
        path = self._resolve(rel_path)
        new_path = self._resolve(new_rel_path)
        if path is None or new_path is None:
            return {"success": False, "error": "Invalid path"}
        if not path.exists():
            return {"success": False, "error": f"Source not found: {rel_path}"}
        if new_path.exists():
            return {"success": False, "error": f"Target exists: {new_rel_path}"}
        new_path.parent.mkdir(parents=True, exist_ok=True)
        path.rename(new_path)
        return {"success": True, "source": rel_path, "target": new_rel_path}

    def _write(self, rel_path: str, content: str) -> Dict[str, Any]:
        path = self._resolve(rel_path)
        if path is None:
            return {"success": False, "error": "Invalid path"}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"success": True, "path": rel_path, "size": len(content)}

    def write(self, rel_path: str, content: str) -> Dict[str, Any]:
        """Create or overwrite a workspace file."""
        return self._write(rel_path, content)

    # ------------------------------------------------------------------ #
    #  Listing & tree
    # ------------------------------------------------------------------ #

    def list_files(self, prefix: str = "") -> List[Dict[str, Any]]:
        base = self.root / prefix if prefix else self.root
        if not base.is_dir():
            return []
        entries = []
        for f in sorted(base.rglob("*")):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                rel = str(f.relative_to(self.root))
                stat = f.stat()
                entries.append(FileEntry(
                    path=rel,
                    size=stat.st_size,
                    created_at=stat.st_ctime,
                    modified_at=stat.st_mtime,
                    extension=f.suffix.lower(),
                    lines=len(f.read_text(encoding="utf-8").splitlines()) if stat.st_size < MAX_FILE_SIZE else 0,
                ).to_dict())
        return entries

    def tree(self) -> str:
        lines = ["workspace/"]
        self._build_tree(self.root, lines, prefix="")
        return "\n".join(lines)

    def _build_tree(self, directory: Path, lines: List[str], prefix: str):
        items = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            connector = "+-- " if is_last else "|-- "
            if item.is_dir():
                lines.append(f"{prefix}{connector}{item.name}/")
                ext = "    " if is_last else "|   "
                self._build_tree(item, lines, prefix + ext)
            elif item.suffix.lower() in SUPPORTED_EXTENSIONS:
                lines.append(f"{prefix}{connector}{item.name}")

    # ------------------------------------------------------------------ #
    #  Project management
    # ------------------------------------------------------------------ #

    def create_project(self, name: str) -> Dict[str, Any]:
        project_dir = self.root / name
        if project_dir.exists():
            return {"success": False, "error": f"Project already exists: {name}"}
        project_dir.mkdir(parents=True)
        return {"success": True, "project": name}

    def list_projects(self) -> List[str]:
        return sorted(
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    # ------------------------------------------------------------------ #
    #  Utility
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict[str, Any]:
        all_files = self.list_files()
        extensions = {}
        for f in all_files:
            ext = f["extension"]
            extensions[ext] = extensions.get(ext, 0) + 1
        return {
            "total_files": len(all_files),
            "total_size": sum(f["size"] for f in all_files),
            "by_extension": extensions,
            "root": str(self.root),
        }

    @staticmethod
    def _cleanup_empty_dirs(dir_path: Path):
        while dir_path != dir_path.parent:
            try:
                if dir_path.exists() and not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    dir_path = dir_path.parent
                else:
                    break
            except (OSError, PermissionError):
                break
