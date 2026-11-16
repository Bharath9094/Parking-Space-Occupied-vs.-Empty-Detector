from typing import Any

import pandas as pd


def history_csv(history: list[dict[str, Any]]) -> bytes:
    columns = [
        "id",
        "image_name",
        "detected_at",
        "total_spaces",
        "occupied_spaces",
        "empty_spaces",
        "occupancy_percentage",
        "availability_percentage",
    ]
    frame = pd.DataFrame(history, columns=columns)
    return frame.to_csv(index=False).encode("utf-8-sig")
