import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HISTORY_PATH = Path("patient_history.csv")
HISTORY_COLUMNS = [
    "timestamp",
    "patient_id",
    "image_name",
    "model",
    "prediction",
    "confidence",
    "probabilities",
]


def read_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    return pd.read_csv(HISTORY_PATH)


def record_prediction(patient_id, image_name, model, prediction, confidence, probabilities):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "patient_id": patient_id or "Not provided",
        "image_name": image_name,
        "model": model,
        "prediction": prediction,
        "confidence": round(float(confidence), 6),
        "probabilities": json.dumps(probabilities),
    }
    history = read_history()
    history = pd.concat([history, pd.DataFrame([record])], ignore_index=True)
    history.to_csv(HISTORY_PATH, index=False)
    return record
