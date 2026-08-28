"""Download the CDC Diabetes Health Indicators dataset."""

import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/lburenkov/Diabetes/main/diabetes_012_health_indicators_BRFSS2015.csv"

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT_FILE = RAW_DIR / "diabetes_012_health_indicators_BRFSS2015.csv"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_FILE.exists():
        print(
            f"Already exists: {OUT_FILE} "
            f"({OUT_FILE.stat().st_size / 1e6:.1f} MB)"
        )
        return

    print(f"Downloading to {OUT_FILE} ...")
    urllib.request.urlretrieve(URL, OUT_FILE)
    print(f"Done. Size: {OUT_FILE.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()