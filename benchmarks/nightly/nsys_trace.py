# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Convert exported Nsight Systems SQLite reports into compact derived artifacts."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any


def write_perfetto_trace(sqlite_path: Path | str, output_path: Path | str, *, measure_frames: int) -> Path:
    """Write a trimmed Perfetto trace JSON for one Nsight Systems SQLite export."""
    sqlite_file = Path(sqlite_path)
    output_file = Path(output_path)

    with sqlite3.connect(sqlite_file) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        strings = {row["id"]: row["value"] for row in cur.execute("SELECT id, value FROM StringIds")}
        window_start, window_end = _resolve_measure_window(cur, measure_frames)
        events = _build_trace_events(cur, strings, window_start, window_end)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    return output_file


def summarize_kernels(sqlite_path: Path | str, *, measure_frames: int) -> dict[str, Any]:
    """Summarize measured-window kernel time by kernel name from an Nsight SQLite export."""
    sqlite_file = Path(sqlite_path)
    with sqlite3.connect(sqlite_file) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        strings = {row["id"]: row["value"] for row in cur.execute("SELECT id, value FROM StringIds")}
        window_start, window_end = _resolve_measure_window(cur, measure_frames)
        kernels_ms, event_count = _aggregate_kernel_ms(cur, strings, window_start, window_end)

    sorted_kernels = dict(sorted(kernels_ms.items(), key=lambda item: item[1], reverse=True))
    total_ms = round(sum(sorted_kernels.values()), 6)
    return {
        "measure_frames": measure_frames,
        "window_start_ns": int(window_start),
        "window_end_ns": int(window_end),
        "window_ms": round(max(window_end - window_start, 0) / 1_000_000.0, 6),
        "kernel_event_count": event_count,
        "kernel_count": len(sorted_kernels),
        "total_kernel_ms": total_ms,
        "kernels": sorted_kernels,
    }


def _resolve_measure_window(cur: sqlite3.Cursor, measure_frames: int) -> tuple[int, int]:
    launches = list(
        cur.execute(
            """
            SELECT r.start, r.end, s.value AS name
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN StringIds s ON s.id = r.nameId
            WHERE s.value LIKE 'cudaGraphLaunch%'
            ORDER BY r.start
            """
        )
    )
    if launches and measure_frames > 0 and len(launches) >= measure_frames:
        window_start = launches[-measure_frames]["start"]
    elif launches:
        window_start = launches[0]["start"]
    else:
        row = cur.execute("SELECT MIN(start) AS start FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
        window_start = row["start"] if row and row["start"] is not None else 0

    row = cur.execute(
        "SELECT MAX(end) AS end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end >= ?",
        (window_start,),
    ).fetchone()
    window_end = row["end"] if row and row["end"] is not None else window_start
    return int(window_start), int(window_end)


def _aggregate_kernel_ms(
    cur: sqlite3.Cursor,
    strings: dict[int, str],
    window_start: int,
    window_end: int,
) -> tuple[dict[str, float], int]:
    kernels_ms: defaultdict[str, float] = defaultdict(float)
    event_count = 0
    for row in cur.execute(
        """
        SELECT start, end, shortName
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE start < ? AND end > ?
        ORDER BY start
        """,
        (window_end, window_start),
    ):
        start = max(int(row["start"]), window_start)
        end = min(int(row["end"]), window_end)
        if end <= start:
            continue
        name = strings.get(row["shortName"], f"kernel_{row['shortName']}")
        kernels_ms[name] += (end - start) / 1_000_000.0
        event_count += 1

    rounded = {name: round(duration_ms, 6) for name, duration_ms in kernels_ms.items()}
    return rounded, event_count


def _build_trace_events(
    cur: sqlite3.Cursor,
    strings: dict[int, str],
    window_start: int,
    window_end: int,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for pid, tid, name in (
        (1, 1, "CUDA Graph Launch API"),
        (1, 2, "CUDA Runtime Summary"),
    ):
        events.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": tid, "args": {"name": name}})

    streams = [
        row["streamId"]
        for row in cur.execute(
            """
            SELECT DISTINCT streamId
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE start < ? AND end > ?
            ORDER BY streamId
            """,
            (window_end, window_start),
        )
    ]
    for stream_id in streams:
        events.append(
            {
                "ph": "M",
                "name": "thread_name",
                "pid": 2,
                "tid": int(stream_id),
                "args": {"name": f"GPU Stream {stream_id}"},
            }
        )

    for row in cur.execute(
        """
        SELECT r.start, r.end, s.value AS name
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON s.id = r.nameId
        WHERE r.start < ? AND r.end > ?
          AND s.value IN ('cudaGraphLaunch_v10000', 'cuCtxSynchronize')
        ORDER BY r.start
        """,
        (window_end, window_start),
    ):
        events.append(
            {
                "name": row["name"],
                "cat": "runtime",
                "ph": "X",
                "ts": row["start"] / 1000.0,
                "dur": max((row["end"] - row["start"]) / 1000.0, 0.001),
                "pid": 1,
                "tid": 1 if "GraphLaunch" in row["name"] else 2,
                "args": {},
            }
        )

    for row in cur.execute(
        """
        SELECT start, end, streamId, shortName
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE start < ? AND end > ?
        ORDER BY start
        """,
        (window_end, window_start),
    ):
        events.append(
            {
                "name": strings.get(row["shortName"], f"kernel_{row['shortName']}"),
                "cat": "kernel",
                "ph": "X",
                "ts": row["start"] / 1000.0,
                "dur": max((row["end"] - row["start"]) / 1000.0, 0.001),
                "pid": 2,
                "tid": int(row["streamId"]),
                "args": {},
            }
        )

    return events
