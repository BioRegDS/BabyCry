#!/usr/bin/env python3
"""
Prepare the public infant-cry datasets used in the manuscript.

Datasets
--------
1. Donate-a-Cry
   hungry      -> hunger
   discomfort  -> discomfort
   tired       -> tired

   The app-instance UUID embedded in the filename is retained as group_id.

2. Dunstan Tier 1
   neh -> hunger
   heh -> discomfort
   owh -> tired

Notes
-----
- Audio is extracted without resampling or re-encoding.
- SHA-256 hashes and basic audio metadata are recorded.
- Exact within- and cross-dataset duplicates are checked.
- Expected dataset counts are validated strictly.
- Only datasets used in the manuscript are prepared here.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import soundfile as sf


# ============================================================
# Constants
# ============================================================

DONATE_LABEL_MAP = {
    "hungry": "hunger",
    "discomfort": "discomfort",
    "tired": "tired",
}

DUNSTAN_LABEL_MAP = {
    "neh": "hunger",
    "heh": "discomfort",
    "owh": "tired",
}

EXPECTED_COUNTS = {
    "donate": {
        "hunger": 382,
        "discomfort": 27,
        "tired": 24,
    },
    "dunstan": {
        "hunger": 72,
        "discomfort": 57,
        "tired": 76,
    },
}

UUID_RE = re.compile(
    r"^("
    r"[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}"
    r")-"
)


# ============================================================
# Utilities
# ============================================================

def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate SHA-256 hash."""

    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(chunk_size),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def audio_info(
    path: Path,
) -> dict:
    """Read basic audio metadata."""

    try:

        info = sf.info(path)

        return {
            "sampling_rate_hz":
                int(info.samplerate),

            "channels":
                int(info.channels),

            "frames":
                int(info.frames),

            "duration_sec":
                float(info.duration),

            "format":
                str(info.format),

            "subtype":
                str(info.subtype),

            "read_error":
                "",
        }

    except Exception as exc:

        return {
            "sampling_rate_hz": None,
            "channels": None,
            "frames": None,
            "duration_sec": None,
            "format": "",
            "subtype": "",
            "read_error": repr(exc),
        }


def safe_extract(
    zf: zipfile.ZipFile,
    member: str,
    destination: Path,
) -> None:
    """
    Extract one member to a controlled destination.

    ZIP directory structure is not recreated, which keeps the output
    deterministic and avoids path traversal.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zf.open(member) as src:
        with destination.open("wb") as dst:
            shutil.copyfileobj(
                src,
                dst,
            )


def unique_destination(
    folder: Path,
    filename: str,
) -> Path:
    """
    Prevent accidental overwriting if ZIP members share a basename.
    """

    destination = (
        folder
        / filename
    )

    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix

    i = 2

    while True:

        candidate = (
            folder
            / f"{stem}__{i}{suffix}"
        )

        if not candidate.exists():
            return candidate

        i += 1


def relative_to_project(
    path: Path,
    project: Path,
) -> str:
    """
    Store a project-relative path for reproducibility.
    """

    return str(
        path.resolve().relative_to(
            project.resolve()
        )
    )


# ============================================================
# Donate-a-Cry
# ============================================================

def prepare_donate(
    zip_path: Path,
    project: Path,
    output_root: Path,
) -> list[dict]:

    rows = []

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as zf:

        for member in zf.namelist():

            if not member.lower().endswith(
                ".wav"
            ):
                continue

            parts = Path(
                member
            ).parts

            lower_parts = [
                x.lower()
                for x in parts
            ]

            try:

                idx = lower_parts.index(
                    "donateacry_corpus_cleaned_and_updated_data"
                )

            except ValueError:

                continue

            if idx + 1 >= len(parts):
                continue

            original_label = (
                parts[idx + 1]
                .strip()
                .lower()
            )

            if original_label not in DONATE_LABEL_MAP:
                continue

            label = (
                DONATE_LABEL_MAP[
                    original_label
                ]
            )

            filename = (
                Path(member).name
            )

            destination = unique_destination(
                output_root
                / "donate"
                / "audio"
                / label,
                filename,
            )

            safe_extract(
                zf,
                member,
                destination,
            )

            match = UUID_RE.match(
                destination.name
            )

            group_id = (
                match.group(1).lower()
                if match
                else ""
            )

            info = audio_info(
                destination
            )

            rows.append(
                {
                    "dataset":
                        "donate",

                    "label":
                        label,

                    "original_label":
                        original_label,

                    "group_id":
                        group_id,

                    "filename":
                        destination.name,

                    "audio_relpath":
                        relative_to_project(
                            destination,
                            project,
                        ),

                    "source_zip_member":
                        member,

                    "sha256":
                        sha256_file(
                            destination
                        ),

                    **info,
                }
            )

    return rows


# ============================================================
# Dunstan Tier 1
# ============================================================

def is_dunstan_337_member(
    member: str,
) -> bool:
    """
    Detect the Dunstan 337 section while tolerating variable whitespace.
    """

    for part in Path(
        member
    ).parts:

        normalized = " ".join(
            part
            .strip()
            .lower()
            .split()
        )

        if (
            normalized.startswith(
                "dunstan"
            )
            and "337"
            in normalized
        ):
            return True

    return False


def prepare_dunstan(
    zip_path: Path,
    project: Path,
    output_root: Path,
) -> list[dict]:

    rows = []

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as zf:

        for member in zf.namelist():

            if not member.lower().endswith(
                ".wav"
            ):
                continue

            if not is_dunstan_337_member(
                member
            ):
                continue

            parts = Path(
                member
            ).parts

            if len(parts) < 2:
                continue

            original_label = (
                parts[-2]
                .strip()
                .lower()
            )

            if original_label not in DUNSTAN_LABEL_MAP:
                continue

            label = (
                DUNSTAN_LABEL_MAP[
                    original_label
                ]
            )

            filename = (
                Path(member).name
            )

            destination = unique_destination(
                output_root
                / "dunstan"
                / "audio"
                / label,
                filename,
            )

            safe_extract(
                zf,
                member,
                destination,
            )

            info = audio_info(
                destination
            )

            rows.append(
                {
                    "dataset":
                        "dunstan",

                    "label":
                        label,

                    "original_label":
                        original_label,

                    "group_id":
                        "",

                    "filename":
                        destination.name,

                    "audio_relpath":
                        relative_to_project(
                            destination,
                            project,
                        ),

                    "source_zip_member":
                        member,

                    "sha256":
                        sha256_file(
                            destination
                        ),

                    **info,
                }
            )

    return rows


# ============================================================
# QC
# ============================================================

def validate_expected_counts(
    metadata: pd.DataFrame,
) -> None:

    print(
        "\n===== EXPECTED COUNT CHECK ====="
    )

    errors = []

    for dataset, expected in (
        EXPECTED_COUNTS.items()
    ):

        subset = metadata[
            metadata[
                "dataset"
            ]
            == dataset
        ]

        observed = (
            subset[
                "label"
            ]
            .value_counts()
            .to_dict()
        )

        print(
            f"\n{dataset}"
        )

        for (
            label,
            expected_n,
        ) in expected.items():

            observed_n = int(
                observed.get(
                    label,
                    0,
                )
            )

            status = (
                "OK"
                if observed_n
                == expected_n
                else "ERROR"
            )

            print(
                f"  {label:12s} "
                f"observed={observed_n:4d} "
                f"expected={expected_n:4d} "
                f"{status}"
            )

            if (
                observed_n
                != expected_n
            ):

                errors.append(
                    (
                        dataset,
                        label,
                        observed_n,
                        expected_n,
                    )
                )

    if errors:

        raise RuntimeError(
            "Dataset count validation failed:\n"
            + "\n".join(
                str(x)
                for x in errors
            )
        )


def validate_readability(
    metadata: pd.DataFrame,
) -> None:

    bad = metadata[
        metadata[
            "read_error"
        ].fillna("") != ""
    ]

    print(
        "\n===== AUDIO READABILITY ====="
    )

    print(
        "Total audio files:",
        len(metadata),
    )

    print(
        "Read errors:",
        len(bad),
    )

    if len(bad) > 0:

        print(
            bad[
                [
                    "dataset",
                    "filename",
                    "read_error",
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Unreadable audio files detected."
        )


def duplicate_tables(
    metadata: pd.DataFrame,
):

    duplicated = metadata[
        metadata.duplicated(
            "sha256",
            keep=False,
        )
    ].copy()

    within = []
    cross = []

    for sha, g in duplicated.groupby(
        "sha256"
    ):

        datasets = sorted(
            g[
                "dataset"
            ].unique()
        )

        record = {
            "sha256":
                sha,

            "n_files":
                len(g),

            "datasets":
                ";".join(
                    datasets
                ),

            "labels":
                ";".join(
                    sorted(
                        g[
                            "label"
                        ].unique()
                    )
                ),

            "files":
                ";".join(
                    g[
                        "audio_relpath"
                    ].astype(str)
                ),
        }

        if len(datasets) > 1:

            cross.append(
                record
            )

        else:

            within.append(
                record
            )

    columns = [
        "sha256",
        "n_files",
        "datasets",
        "labels",
        "files",
    ]

    return (
        pd.DataFrame(
            within,
            columns=columns,
        ),
        pd.DataFrame(
            cross,
            columns=columns,
        ),
    )


def build_summary(
    metadata: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        dataset,
        label,
    ), g in metadata.groupby(
        [
            "dataset",
            "label",
        ]
    ):

        duration = pd.to_numeric(
            g[
                "duration_sec"
            ],
            errors="coerce",
        )

        sr_counts = Counter(
            g[
                "sampling_rate_hz"
            ]
            .dropna()
            .astype(int)
        )

        rows.append(
            {
                "dataset":
                    dataset,

                "label":
                    label,

                "n_files":
                    len(g),

                "n_unique_sha256":
                    g[
                        "sha256"
                    ].nunique(),

                "n_groups":
                    (
                        g[
                            "group_id"
                        ]
                        .replace(
                            "",
                            pd.NA,
                        )
                        .dropna()
                        .nunique()
                    ),

                "duration_min_sec":
                    float(
                        duration.min()
                    ),

                "duration_median_sec":
                    float(
                        duration.median()
                    ),

                "duration_max_sec":
                    float(
                        duration.max()
                    ),

                "sampling_rates":
                    ";".join(
                        f"{sr}:{n}"
                        for sr, n
                        in sorted(
                            sr_counts.items()
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        formatter_class=
        argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--project",
        type=Path,
        default=Path(
            str(Path(__file__).resolve().parents[2])
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete generated Donate/Dunstan outputs before recreating them.",
    )

    args = parser.parse_args()

    project = (
        args.project
        .resolve()
    )

    raw_root = (
        project
        / "data"
        / "raw"
    )

    donate_zip = (
        raw_root
        / "donateacry-corpus-master.zip"
    )

    dunstan_zip = (
        raw_root
        / "InfantCry-DBL.zip"
    )

    for path in [
        donate_zip,
        dunstan_zip,
    ]:

        if not path.exists():

            sys.exit(
                f"ERROR: required raw ZIP not found:\n{path}"
            )

    processed_root = (
        project
        / "data"
        / "processed"
    )

    metadata_root = (
        project
        / "data"
        / "metadata"
    )

    processed_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Protect against accidental mixing with previous output
    # --------------------------------------------------------

    for dataset in [
        "donate",
        "dunstan",
    ]:

        output_dir = (
            processed_root
            / dataset
        )

        if (
            output_dir.exists()
            and any(
                output_dir.iterdir()
            )
        ):

            if args.force:

                shutil.rmtree(
                    output_dir
                )

            else:

                sys.exit(
                    "ERROR: output directory is not empty:\n"
                    f"{output_dir}\n\n"
                    "Use --force only if you intentionally want to recreate it."
                )

    # --------------------------------------------------------
    # Preparation
    # --------------------------------------------------------

    print(
        "Preparing Donate-a-Cry..."
    )

    donate_rows = prepare_donate(
        donate_zip,
        project,
        processed_root,
    )

    print(
        "Selected:",
        len(donate_rows),
    )

    print(
        "\nPreparing Dunstan Tier 1..."
    )

    dunstan_rows = prepare_dunstan(
        dunstan_zip,
        project,
        processed_root,
    )

    print(
        "Selected:",
        len(dunstan_rows),
    )

    metadata = pd.DataFrame(
        donate_rows
        + dunstan_rows
    )

    # --------------------------------------------------------
    # QC
    # --------------------------------------------------------

    validate_expected_counts(
        metadata
    )

    validate_readability(
        metadata
    )

    (
        within_duplicates,
        cross_duplicates,
    ) = duplicate_tables(
        metadata
    )

    if len(
        cross_duplicates
    ) > 0:

        raise RuntimeError(
            "Exact duplicate audio files were detected across datasets."
        )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    for dataset in [
        "donate",
        "dunstan",
    ]:

        subset = metadata[
            metadata[
                "dataset"
            ]
            == dataset
        ].copy()

        subset.to_csv(
            processed_root
            / dataset
            / "metadata.csv",
            index=False,
        )

    metadata.to_csv(
        metadata_root
        / "donate_dunstan_master.csv",
        index=False,
    )

    summary = build_summary(
        metadata
    )

    summary.to_csv(
        metadata_root
        / "dataset_summary.csv",
        index=False,
    )

    within_duplicates.to_csv(
        metadata_root
        / "exact_duplicates_within_dataset.csv",
        index=False,
    )

    cross_duplicates.to_csv(
        metadata_root
        / "exact_duplicates_cross_dataset.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print(
        "\n===== DATASET SUMMARY ====="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nExact duplicate groups within datasets:",
        len(
            within_duplicates
        ),
    )

    print(
        "Exact duplicate groups across datasets:",
        len(
            cross_duplicates
        ),
    )

    print(
        "\n01_prepare_datasets.py completed successfully."
    )


if __name__ == "__main__":
    main()
