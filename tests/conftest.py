"""Synthetic fixtures only; these are not empirical BovDB results."""

from datetime import date, datetime, time, timedelta, timezone

import pandas as pd
import pytest


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    rows = []
    for day in range(12):
        session = date(2026, 1, 5) + timedelta(days=day)
        for minute in range(8):
            t = datetime.combine(session, time(14, 0), tzinfo=timezone.utc) + timedelta(minutes=minute)
            row = {
                "instrument": "SYNTH",
                "session_date": session.isoformat(),
                "decision_time": t.isoformat(),
                "features_available_at": t.isoformat(),
                "x_return": (minute - 3.5) / 100 + day / 200,
                "x_volume": 10 + day + minute,
            }
            for h in (5, 10, 15):
                row[f"y_{h}m"] = (minute + day + h // 5) % 2
                row[f"label_info_start_{h}m"] = t.isoformat()
                row[f"label_info_end_{h}m"] = (t + timedelta(minutes=h)).isoformat()
                row[f"label_known_at_{h}m"] = (t + timedelta(minutes=h + 1)).isoformat()
            rows.append(row)
    return pd.DataFrame(rows)
