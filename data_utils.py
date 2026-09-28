from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "SMSSpamCollection"


def load_data() -> pd.DataFrame:
    """Load the UCI SMS Spam Collection with columns: label, message."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Run `python download_dataset.py` first."
        )
    df = pd.read_csv(DATA_FILE, sep="\t", header=None, names=["label", "message"], encoding="utf-8")
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])].reset_index(drop=True)
    return df
