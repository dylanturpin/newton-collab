# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Convert exported Nsight Systems SQLite reports into compact Perfetto traces."""

from __future__ import annotations

import json
import sqlite3
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
        events = _build_trace_events(cur, strings, window_start, window_end)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    return output_file


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
