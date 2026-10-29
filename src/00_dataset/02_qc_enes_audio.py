#!/usr/bin/env python3
"""
Quality control of the raw Enes Baby Cries audio archive.

This script performs AUDIO-LEVEL QC only.

Checks
------
- Number of files in the ZIP archive
- Audio readability
- Sampling rate
- Number of channels
- Duration
- Audio format / subtype
- SHA-256 exact duplicates

Important
---------
No cry-reason labels are inferred from filenames.
Label construction is handled separately by
03_prepare_enes_labels.py using the official metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".ogg",
    ".aif",
    ".aiff",
}


EXPECTED_AUDIO_FILES = 78094
EXPECTED_SR = 22050
EXPECTED_CHANNELS = 1


def sha256_bytes(
    data: bytes,
) -> str:

    h = hashlib.sha256()
    h.update(data)

    return h.hexdigest()


def inspect_audio_bytes(
    raw: bytes,
) -> dict:
    """
    Read audio metadata directly from bytes.

    No temporary audio files are generated.
    """

    try:

        with io.BytesIO(
            raw
        ) as bio:

            info = sf.info(
                bio
            )

        return {
            "sampling_rate_hz":
                int(
                    info.samplerate
                ),

            "channels":
                int(
                    info.channels
                ),

            "frames":
                int(
                    info.frames
                ),

            "duration_sec":
                float(
                    info.duration
                ),

            "format":
                str(
                    info.format
                ),

            "subtype":
                str(
                    info.subtype
                ),

            "read_error":
                "",
        }

    except Exception as exc:

        return {
            "sampling_rate_hz":
                np.nan,

            "channels":
                np.nan,

            "frames":
                np.nan,

            "duration_sec":
                np.nan,

            "format":
                "",

            "subtype":
                "",

            "read_error":
                repr(
                    exc
                ),
        }


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

    args = parser.parse_args()

    project = (
        args.project
        .resolve()
    )

    zip_path = (
        project
        / "data"
        / "raw"
        / "audio_EnesBabyCries1B_cries.zip"
    )

    if not zip_path.exists():

        raise FileNotFoundError(
            f"Enes audio ZIP not found:\n"
            f"{zip_path}"
        )

    outdir = (
        project
        / "data"
        / "metadata"
        / "enes_audio_qc"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Enes ZIP:",
        zip_path
    )

    print(
        "ZIP size (GB):",
        round(
            zip_path.stat().st_size
            / 1024**3,
            3,
        ),
    )

    # ========================================================
    # ZIP manifest
    # ========================================================

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as zf:

        members = [
            info
            for info
            in zf.infolist()
            if not info.is_dir()
        ]

        manifest_rows = []

        for info in members:

            manifest_rows.append(
                {
                    "zip_member":
                        info.filename,

                    "filename":
                        Path(
                            info.filename
                        ).name,

                    "extension":
                        Path(
                            info.filename
                        )
                        .suffix
                        .lower(),

                    "compressed_size":
                        int(
                            info.compress_size
                        ),

                    "uncompressed_size":
                        int(
                            info.file_size
                        ),
                }
            )

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest.to_csv(
        outdir
        / "zip_manifest.csv",
        index=False,
    )

    audio_manifest = manifest[
        manifest[
            "extension"
        ].isin(
            AUDIO_EXTENSIONS
        )
    ].copy()

    audio_manifest = (
        audio_manifest
        .reset_index(
            drop=True
        )
    )

    print(
        "\nFiles in ZIP:",
        len(
            manifest
        ),
    )

    print(
        "Recognized audio files:",
        len(
            audio_manifest
        ),
    )

    print(
        "\nExtensions:"
    )

    print(
        manifest[
            "extension"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    # ========================================================
    # Audio QC
    # ========================================================

    rows = []

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as zf:

        for i, row in (
            audio_manifest
            .iterrows()
        ):

            member = row[
                "zip_member"
            ]

            try:

                raw = zf.read(
                    member
                )

                info = (
                    inspect_audio_bytes(
                        raw
                    )
                )

                digest = (
                    sha256_bytes(
                        raw
                    )
                )

            except Exception as exc:

                info = {
                    "sampling_rate_hz":
                        np.nan,

                    "channels":
                        np.nan,

                    "frames":
                        np.nan,

                    "duration_sec":
                        np.nan,

                    "format":
                        "",

                    "subtype":
                        "",

                    "read_error":
                        repr(
                            exc
                        ),
                }

                digest = ""

            rec = row.to_dict()

            rec.update(
                info
            )

            rec[
                "sha256"
            ] = digest

            rows.append(
                rec
            )

            if (
                (i + 1)
                % 1000
                == 0
                or i + 1
                == len(
                    audio_manifest
                )
            ):

                print(
                    f"QC "
                    f"{i+1}/"
                    f"{len(audio_manifest)}",
                    flush=True,
                )

    qc = pd.DataFrame(
        rows
    )

    qc.to_csv(
        outdir
        / "audio_qc_master.csv",
        index=False,
    )

    # ========================================================
    # Duplicate QC
    # ========================================================

    valid_hash = qc[
        qc[
            "sha256"
        ]
        .astype(str)
        .str.len()
        > 0
    ].copy()

    duplicate_rows = valid_hash[
        valid_hash.duplicated(
            "sha256",
            keep=False,
        )
    ].copy()

    duplicate_rows = (
        duplicate_rows
        .sort_values(
            "sha256"
        )
    )

    duplicate_rows.to_csv(
        outdir
        / "exact_duplicate_files.csv",
        index=False,
    )

    n_unique_hash = (
        valid_hash[
            "sha256"
        ].nunique()
    )

    duplicate_excess = (
        len(
            valid_hash
        )
        - n_unique_hash
    )

    # ========================================================
    # Summary
    # ========================================================

    n_read_ok = int(
        qc[
            "read_error"
        ]
        .fillna("")
        .eq("")
        .sum()
    )

    n_read_error = int(
        len(qc)
        - n_read_ok
    )

    summary = {
        "zip_path":
            str(
                zip_path.relative_to(
                    project
                )
            ),

        "zip_size_gb":
            float(
                zip_path.stat().st_size
                / 1024**3
            ),

        "n_files_total":
            int(
                len(
                    manifest
                )
            ),

        "n_audio_recognized":
            int(
                len(
                    audio_manifest
                )
            ),

        "n_audio_read_ok":
            n_read_ok,

        "n_audio_read_error":
            n_read_error,

        "n_unique_sha256":
            int(
                n_unique_hash
            ),

        "n_duplicate_files_excess":
            int(
                duplicate_excess
            ),

        "sampling_rates":
            {
                str(int(k)):
                    int(v)
                for k, v
                in qc[
                    "sampling_rate_hz"
                ]
                .dropna()
                .value_counts()
                .sort_index()
                .items()
            },

        "channels":
            {
                str(int(k)):
                    int(v)
                for k, v
                in qc[
                    "channels"
                ]
                .dropna()
                .value_counts()
                .sort_index()
                .items()
            },
    }

    # ========================================================
    # Strict validation
    # ========================================================

    validation_errors = []

    if (
        len(
            audio_manifest
        )
        != EXPECTED_AUDIO_FILES
    ):

        validation_errors.append(
            "Unexpected number of "
            f"audio files: "
            f"{len(audio_manifest)} "
            f"(expected "
            f"{EXPECTED_AUDIO_FILES})"
        )

    if n_read_error != 0:

        validation_errors.append(
            f"{n_read_error} "
            "audio files could not "
            "be read."
        )

    if duplicate_excess != 0:

        validation_errors.append(
            f"{duplicate_excess} "
            "exact duplicate audio "
            "files detected."
        )

    observed_sr = set(
        qc[
            "sampling_rate_hz"
        ]
        .dropna()
        .astype(int)
        .unique()
    )

    if observed_sr != {
        EXPECTED_SR
    }:

        validation_errors.append(
            "Unexpected sampling "
            f"rates: "
            f"{sorted(observed_sr)}"
        )

    observed_channels = set(
        qc[
            "channels"
        ]
        .dropna()
        .astype(int)
        .unique()
    )

    if observed_channels != {
        EXPECTED_CHANNELS
    }:

        validation_errors.append(
            "Unexpected channel "
            f"counts: "
            f"{sorted(observed_channels)}"
        )

    summary[
        "strict_validation_passed"
    ] = (
        len(
            validation_errors
        )
        == 0
    )

    summary[
        "validation_errors"
    ] = validation_errors

    with (
        outdir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # Console report
    # ========================================================

    print(
        "\n===== ENES AUDIO QC ====="
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "\n===== DURATION ====="
    )

    print(
        qc[
            "duration_sec"
        ]
        .describe()
        .to_string()
    )

    print(
        "\nExact duplicate excess:",
        duplicate_excess,
    )

    if validation_errors:

        print(
            "\n===== VALIDATION FAILED ====="
        )

        for error in validation_errors:

            print(
                "-",
                error,
            )

        raise RuntimeError(
            "Enes raw-audio QC failed."
        )

    print(
        "\n02_qc_enes_audio.py "
        "completed successfully."
    )


if __name__ == "__main__":
    main()
