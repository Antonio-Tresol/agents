"""Built-in tool executors for LocalTools (file I/O, task CRUD)."""

from __future__ import annotations

import json
import uuid
from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.providers import Store


def _validate_path(path: str) -> None:
    """Raise ValueError if path is absolute or escapes via '..'."""
    for p in (PurePosixPath(path), PureWindowsPath(path)):
        if ".." in p.parts:
            raise ValueError(f"Path must not contain '..': {path}")
    if PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute():
        raise ValueError(f"Path must be relative: {path}")


async def read_file(store: Store, prefix: str, path: str) -> str:
    _validate_path(path)
    content = await store.read(f"{prefix}/{path}")
    if content is None:
        return "File not found."
    return content


async def write_file(store: Store, prefix: str, path: str, content: str) -> str:
    _validate_path(path)
    await store.write(f"{prefix}/{path}", content)
    return "Written successfully."


async def _read_tasks(store: Store, prefix: str) -> list[dict]:
    raw = await store.read(f"{prefix}/tasks.json")
    if raw is None:
        return []
    return json.loads(raw)


async def _write_tasks(store: Store, prefix: str, tasks: list[dict]) -> None:
    await store.write(f"{prefix}/tasks.json", json.dumps(tasks))


async def create_task(store: Store, prefix: str, title: str, details: str = "") -> str:
    tasks = await _read_tasks(store, prefix)
    task_id = uuid.uuid4().hex[:12]
    tasks.append(
        {"id": task_id, "title": title, "details": details, "status": "pending"},
    )
    await _write_tasks(store, prefix, tasks)
    return task_id


async def list_tasks(store: Store, prefix: str) -> str:
    tasks = await _read_tasks(store, prefix)
    return json.dumps(tasks)


async def update_task(
    store: Store,
    prefix: str,
    task_id: str,
    status: str | None = None,
    details: str | None = None,
) -> str:
    tasks = await _read_tasks(store, prefix)
    for task in tasks:
        if task["id"] == task_id:
            if status is not None:
                task["status"] = status
            if details is not None:
                task["details"] = details
            await _write_tasks(store, prefix, tasks)
            return "Updated."
    return "Task not found."


async def get_task(store: Store, prefix: str, task_id: str) -> str:
    tasks = await _read_tasks(store, prefix)
    for task in tasks:
        if task["id"] == task_id:
            return json.dumps(task)
    return "Task not found."
