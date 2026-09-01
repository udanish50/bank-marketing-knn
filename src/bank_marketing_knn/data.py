"""Dataset loading utilities for the UCI Bank Marketing archive."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


@dataclass(frozen=True)
class DatasetChoice:
    """Configuration for one CSV stored in the UCI archive."""

    inner_zip: str
    csv_candidates: tuple[str, ...]


DATASETS: dict[str, DatasetChoice] = {
    "bank": DatasetChoice("bank.zip", ("bank.csv", "bank/bank.csv")),
    "bank-full": DatasetChoice("bank.zip", ("bank-full.csv", "bank/bank-full.csv")),
    "bank-additional": DatasetChoice(
        "bank-additional.zip",
        ("bank-additional/bank-additional.csv", "bank-additional.csv"),
    ),
    "bank-additional-full": DatasetChoice(
        "bank-additional.zip",
        ("bank-additional/bank-additional-full.csv", "bank-additional-full.csv"),
    ),
}


def load_bank_marketing_csv(dataset_zip: str | Path, dataset: str = "bank") -> pd.DataFrame:
    """Load a Bank Marketing CSV from the original nested UCI zip archive.

    Parameters
    ----------
    dataset_zip:
        Path to ``bank+marketing.zip``.
    dataset:
        One of ``bank``, ``bank-full``, ``bank-additional``, or
        ``bank-additional-full``.
    """

    dataset_zip = Path(dataset_zip)
    if dataset not in DATASETS:
        valid = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset '{dataset}'. Valid choices: {valid}")
    if not dataset_zip.exists():
        raise FileNotFoundError(
            f"Dataset archive not found: {dataset_zip}. "
            "Place bank+marketing.zip in data/raw/ or pass --dataset-zip."
        )

    choice = DATASETS[dataset]
    with ZipFile(dataset_zip) as outer_zip:
        inner_bytes = outer_zip.read(choice.inner_zip)

    with ZipFile(BytesIO(inner_bytes)) as inner_zip:
        for csv_name in choice.csv_candidates:
            if csv_name in inner_zip.namelist():
                with inner_zip.open(csv_name) as csv_file:
                    return pd.read_csv(csv_file, sep=";")

    candidates = ", ".join(choice.csv_candidates)
    raise FileNotFoundError(f"Could not find any of these CSV files: {candidates}")
