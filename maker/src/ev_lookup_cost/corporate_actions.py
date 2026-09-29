"""Published exchange notices; trade the risk response only after publication."""
import json
from pathlib import Path


def known_actions(day: str, metadata_root: Path) -> list[dict]:
    path = metadata_root / "announcements/index.json"
    if not path.exists():
        return []
    return [r for r in json.loads(path.read_text()) if "error" not in r and r["announce_day"] < day]
