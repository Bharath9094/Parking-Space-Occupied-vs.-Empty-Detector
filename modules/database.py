import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from config import DATABASE_PATH


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id TEXT PRIMARY KEY,
                image_name TEXT NOT NULL,
                detected_at TEXT NOT NULL,
                total_spaces INTEGER NOT NULL,
                occupied_spaces INTEGER NOT NULL,
                empty_spaces INTEGER NOT NULL,
                occupancy_percentage REAL NOT NULL,
                availability_percentage REAL NOT NULL,
                output_path TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS parking_layouts (
                name TEXT PRIMARY KEY,
                regions_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


def save_detection(
    image_name: str,
    total_spaces: int,
    occupied_spaces: int,
    output_path: Path,
) -> dict[str, Any]:
    empty_spaces = total_spaces - occupied_spaces
    occupancy = (occupied_spaces / total_spaces * 100) if total_spaces else 0.0
    availability = (empty_spaces / total_spaces * 100) if total_spaces else 0.0
    detected_at = datetime.now().astimezone().isoformat(timespec="seconds")
    detection_id = str(uuid.uuid4())
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO detections (
                id, image_name, detected_at, total_spaces, occupied_spaces,
                empty_spaces, occupancy_percentage, availability_percentage, output_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                detection_id,
                image_name,
                detected_at,
                total_spaces,
                occupied_spaces,
                empty_spaces,
                occupancy,
                availability,
                str(output_path),
            ),
        )
    return {
        "id": detection_id,
        "image_name": image_name,
        "detected_at": detected_at,
        "total_spaces": total_spaces,
        "occupied_spaces": occupied_spaces,
        "empty_spaces": empty_spaces,
        "occupancy_percentage": occupancy,
        "availability_percentage": availability,
        "output_path": str(output_path),
    }


def get_history(limit: int | None = None) -> list[dict[str, Any]]:
    query = "SELECT * FROM detections ORDER BY detected_at DESC"
    parameters: tuple[Any, ...] = ()
    if limit is not None:
        query += " LIMIT ?"
        parameters = (limit,)
    with _connect() as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [dict(row) for row in rows]


def save_layout(name: str, normalized_regions: list[dict[str, float]]) -> None:
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO parking_layouts (name, regions_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                regions_json = excluded.regions_json,
                updated_at = excluded.updated_at
            """,
            (
                name.strip(),
                json.dumps(normalized_regions),
                datetime.now().astimezone().isoformat(timespec="seconds"),
            ),
        )


def get_layout(name: str) -> list[dict[str, float]] | None:
    if not name.strip():
        return None
    with _connect() as connection:
        row = connection.execute(
            "SELECT regions_json FROM parking_layouts WHERE name = ?",
            (name.strip(),),
        ).fetchone()
    if row is None:
        return None
    regions = json.loads(row["regions_json"])
    if not isinstance(regions, list):
        raise ValueError(f"Saved parking layout '{name}' is not valid.")
    return regions


def list_layout_names() -> list[str]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT name FROM parking_layouts ORDER BY name"
        ).fetchall()
    return [str(row["name"]) for row in rows]
