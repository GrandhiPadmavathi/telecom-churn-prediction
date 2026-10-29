"""Download the public IBM Telco Customer Churn CSV into data/raw/."""

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from churn.config import DATA_URL, SETTINGS  # noqa: E402


def main() -> None:
    path = SETTINGS.paths.raw_csv
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    resp = requests.get(DATA_URL, timeout=60)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    print(f"saved {len(resp.content):,} bytes to {path}")


if __name__ == "__main__":
    main()
