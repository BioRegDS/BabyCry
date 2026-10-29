#!/usr/bin/env python3
"""
Extract identical variable-duration handcrafted acoustic features
from Donate-a-Cry and Enes.

Manuscript analysis subset
--------------------------
Donate-a-Cry:
    hunger
    discomfort

Enes:
    high-confidence hunger
    high-confidence discomfort

Audio harmonization
-------------------
- mono
- resample to 16 kHz
- 4 kHz low-pass filter
- original clip duration retained
- no crop
- no zero-padding
- no peak normalization

Both datasets are passed through exactly the same feature extractor.

Feature families
----------------
- Energy (RMS)
- Zero crossing rate
- Spectral descriptors
- Spectral contrast
- MFCC
- Delta MFCC
- Delta-delta MFCC
- Chroma
- Pitch / voicing
- Temporal onset
- Tempo

Expected acoustic feature count: 618
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt


# ============================================================
# Constants
# ============================================================

TARGET_SR = 16000
LOWPASS_HZ = 4000.0

N_FFT = 1024
HOP_LENGTH = 256
N_MFCC = 20

EXPECTED_FEATURES = 618

EXPECTED_DONATE = {
    "hunger": 382,
    "discomfort": 27,
}

EXPECTED_ENES = {
    "hunger": 9891,
    "discomfort": 9246,
}

EXPECTED_ENES_INFANTS = 24
EXPECTED_ENES_SESSIONS = 378


# ============================================================
# Statistics
# ============================================================

def summary_stats(
    prefix: str,
    values,
) -> dict:

    x = np.asarray(
        values,
        dtype=float,
    ).ravel()

    x = x[
        np.isfinite(x)
    ]

    names = [
        "mean",
        "std",
        "median",
        "p25",
        "p75",
        "min",
        "max",
    ]

    if len(x) == 0:

        return {
            f"{prefix}_{name}": np.nan
            for name in names
        }

    values = [
        np.mean(x),
        np.std(x),
        np.median(x),
        np.percentile(x, 25),
        np.percentile(x, 75),
        np.min(x),
        np.max(x),
    ]

    return {
        f"{prefix}_{name}": float(value)
        for name, value
        in zip(
            names,
            values,
        )
    }


# ============================================================
# Audio preprocessing
# ============================================================

def to_mono(
    y: np.ndarray,
) -> np.ndarray:

    if y.ndim == 1:
        return y.astype(
            np.float32
        )

    return (
        np.mean(
            y,
            axis=1,
        )
        .astype(
            np.float32
        )
    )


def resample_audio(
    y: np.ndarray,
    sr: int,
) -> np.ndarray:

    if sr == TARGET_SR:
        return y.astype(
            np.float32
        )

    return (
        librosa.resample(
            y,
            orig_sr=sr,
            target_sr=TARGET_SR,
        )
        .astype(
            np.float32
        )
    )


def lowpass_filter(
    y: np.ndarray,
) -> np.ndarray:

    sos = butter(
        8,
        LOWPASS_HZ
        / (
            TARGET_SR
            / 2
        ),
        btype="lowpass",
        output="sos",
    )

    return (
        sosfiltfilt(
            sos,
            y,
        )
        .astype(
            np.float32
        )
    )


def preprocess_audio(
    y: np.ndarray,
    sr: int,
) -> np.ndarray:
    """
    Common Donate/Enes preprocessing.
    """

    y = to_mono(
        y
    )

    y = resample_audio(
        y,
        sr,
    )

    y = lowpass_filter(
        y
    )

    return y


# ============================================================
# Acoustic feature extraction
# ============================================================

def safe_delta(
    mfcc: np.ndarray,
    order: int,
) -> np.ndarray:

    n_frames = (
        mfcc.shape[1]
    )

    width = min(
        9,
        (
            n_frames
            if n_frames % 2
            else n_frames - 1
        ),
    )

    if width < 3:

        return np.full_like(
            mfcc,
            np.nan,
            dtype=float,
        )

    try:

        return librosa.feature.delta(
            mfcc,
            order=order,
            width=width,
            mode="nearest",
        )

    except Exception:

        return np.full_like(
            mfcc,
            np.nan,
            dtype=float,
        )


def extract_features(
    y: np.ndarray,
) -> dict:
    """
    Extract the exact same 618 acoustic features from both datasets.
    """

    out = {}

    # --------------------------------------------------------
    # Energy
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=y,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    out.update(
        summary_stats(
            "rms",
            rms,
        )
    )

    # --------------------------------------------------------
    # Zero-crossing
    # --------------------------------------------------------

    zcr = (
        librosa.feature
        .zero_crossing_rate(
            y,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
        )[0]
    )

    out.update(
        summary_stats(
            "zcr",
            zcr,
        )
    )

    # --------------------------------------------------------
    # Spectral
    # --------------------------------------------------------

    centroid = (
        librosa.feature
        .spectral_centroid(
            y=y,
            sr=TARGET_SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )[0]
    )

    bandwidth = (
        librosa.feature
        .spectral_bandwidth(
            y=y,
            sr=TARGET_SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )[0]
    )

    rolloff = (
        librosa.feature
        .spectral_rolloff(
            y=y,
            sr=TARGET_SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            roll_percent=0.85,
        )[0]
    )

    flatness = (
        librosa.feature
        .spectral_flatness(
            y=y,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )[0]
    )

    out.update(
        summary_stats(
            "spectral_centroid",
            centroid,
        )
    )

    out.update(
        summary_stats(
            "spectral_bandwidth",
            bandwidth,
        )
    )

    out.update(
        summary_stats(
            "spectral_rolloff85",
            rolloff,
        )
    )

    out.update(
        summary_stats(
            "spectral_flatness",
            flatness,
        )
    )

    # --------------------------------------------------------
    # Spectral contrast
    # --------------------------------------------------------

    try:

        contrast = (
            librosa.feature
            .spectral_contrast(
                y=y,
                sr=TARGET_SR,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
            )
        )

        for i in range(
            contrast.shape[0]
        ):

            out.update(
                summary_stats(
                    f"spectral_contrast_{i+1}",
                    contrast[i],
                )
            )

    except Exception:

        for i in range(7):

            out.update(
                summary_stats(
                    f"spectral_contrast_{i+1}",
                    [],
                )
            )

    # --------------------------------------------------------
    # MFCC / Delta / Delta-delta
    # --------------------------------------------------------

    mfcc = (
        librosa.feature
        .mfcc(
            y=y,
            sr=TARGET_SR,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )
    )

    delta1 = safe_delta(
        mfcc,
        order=1,
    )

    delta2 = safe_delta(
        mfcc,
        order=2,
    )

    for i in range(
        N_MFCC
    ):

        out.update(
            summary_stats(
                f"mfcc_{i+1}",
                mfcc[i],
            )
        )

        out.update(
            summary_stats(
                f"dmfcc_{i+1}",
                delta1[i],
            )
        )

        out.update(
            summary_stats(
                f"ddmfcc_{i+1}",
                delta2[i],
            )
        )

    # --------------------------------------------------------
    # Chroma
    # --------------------------------------------------------

    try:

        chroma = (
            librosa.feature
            .chroma_stft(
                y=y,
                sr=TARGET_SR,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
            )
        )

        for i in range(12):

            out.update(
                summary_stats(
                    f"chroma_{i+1}",
                    chroma[i],
                )
            )

    except Exception:

        for i in range(12):

            out.update(
                summary_stats(
                    f"chroma_{i+1}",
                    [],
                )
            )

    # --------------------------------------------------------
    # Pitch / voicing
    # --------------------------------------------------------

    try:

        (
            f0,
            voiced_flag,
            voiced_prob,
        ) = librosa.pyin(
            y,
            fmin=60,
            fmax=1000,
            sr=TARGET_SR,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
        )

        out.update(
            summary_stats(
                "f0_hz",
                (
                    f0
                    if f0 is not None
                    else []
                ),
            )
        )

        out[
            "voiced_fraction"
        ] = (
            float(
                np.nanmean(
                    voiced_flag
                )
            )
            if voiced_flag
            is not None
            else np.nan
        )

        out.update(
            summary_stats(
                "voiced_prob",
                (
                    voiced_prob
                    if voiced_prob
                    is not None
                    else []
                ),
            )
        )

    except Exception:

        out.update(
            summary_stats(
                "f0_hz",
                [],
            )
        )

        out[
            "voiced_fraction"
        ] = np.nan

        out.update(
            summary_stats(
                "voiced_prob",
                [],
            )
        )

    # --------------------------------------------------------
    # Temporal / onset
    # --------------------------------------------------------

    try:

        onset = (
            librosa.onset
            .onset_strength(
                y=y,
                sr=TARGET_SR,
                hop_length=HOP_LENGTH,
            )
        )

        out.update(
            summary_stats(
                "onset_strength",
                onset,
            )
        )

        tempo, _ = (
            librosa.beat
            .beat_track(
                onset_envelope=onset,
                sr=TARGET_SR,
                hop_length=HOP_LENGTH,
            )
        )

        out[
            "tempo_bpm"
        ] = float(
            np.asarray(
                tempo
            ).ravel()[0]
        )

    except Exception:

        out.update(
            summary_stats(
                "onset_strength",
                [],
            )
        )

        out[
            "tempo_bpm"
        ] = np.nan

    return out


# ============================================================
# Donate
# ============================================================

def extract_donate(
    project: Path,
) -> tuple[
    pd.DataFrame,
    list[dict],
]:

    metadata_path = (
        project
        / "data"
        / "processed"
        / "donate"
        / "metadata.csv"
    )

    if not metadata_path.exists():

        raise FileNotFoundError(
            "Donate metadata not found:\n"
            f"{metadata_path}\n"
            "Run 01_prepare_datasets.py first."
        )

    metadata = pd.read_csv(
        metadata_path
    )

    metadata = metadata[
        metadata[
            "label"
        ].isin(
            [
                "hunger",
                "discomfort",
            ]
        )
    ].copy()

    metadata = (
        metadata
        .reset_index(
            drop=True
        )
    )

    rows = []
    errors = []

    for i, row in (
        metadata
        .iterrows()
    ):

        try:

            audio_path = (
                project
                / row[
                    "audio_relpath"
                ]
            )

            y, sr = sf.read(
                audio_path,
                always_2d=False,
            )

            original_duration = (
                len(y)
                / int(sr)
            )

            y = preprocess_audio(
                y,
                int(sr),
            )

            features = extract_features(
                y
            )

            rec = {
                "dataset":
                    "donate",

                "file":
                    row[
                        "filename"
                    ],

                "group_id":
                    row[
                        "group_id"
                    ],

                "label":
                    row[
                        "label"
                    ],

                "original_sr":
                    int(sr),

                "original_duration_sec":
                    float(
                        original_duration
                    ),

                "analysis_duration_sec":
                    float(
                        len(y)
                        / TARGET_SR
                    ),
            }

            rec.update(
                features
            )

            rows.append(
                rec
            )

        except Exception as exc:

            errors.append(
                {
                    "dataset":
                        "donate",

                    "file":
                        row.get(
                            "filename",
                            "",
                        ),

                    "error":
                        repr(
                            exc
                        ),
                }
            )

        if (
            (i + 1) % 50
            == 0
            or i + 1
            == len(metadata)
        ):

            print(
                f"Donate "
                f"{i+1}/"
                f"{len(metadata)} "
                f"ok={len(rows)} "
                f"err={len(errors)}",
                flush=True,
            )

    return (
        pd.DataFrame(
            rows
        ),
        errors,
    )


# ============================================================
# Enes
# ============================================================

def extract_enes(
    project: Path,
) -> tuple[
    pd.DataFrame,
    list[dict],
]:

    metadata_path = (
        project
        / "data"
        / "processed"
        / "enes"
        / "hunger_discomfort_metadata.csv"
    )

    zip_path = (
        project
        / "data"
        / "raw"
        / "audio_EnesBabyCries1B_cries.zip"
    )

    if not metadata_path.exists():

        raise FileNotFoundError(
            "Enes analysis metadata not found:\n"
            f"{metadata_path}\n"
            "Run 03_prepare_enes_labels.py first."
        )

    if not zip_path.exists():

        raise FileNotFoundError(
            "Enes audio ZIP not found:\n"
            f"{zip_path}"
        )

    metadata = pd.read_csv(
        metadata_path
    )

    metadata = metadata[
        metadata[
            "label"
        ].isin(
            [
                "hunger",
                "discomfort",
            ]
        )
    ].copy()

    metadata = (
        metadata
        .reset_index(
            drop=True
        )
    )

    rows = []
    errors = []

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as zf:

        basename_map = {}

        duplicate_basenames = set()

        for member in zf.namelist():

            if member.endswith("/"):
                continue

            basename = (
                Path(
                    member
                ).name
            )

            if basename in basename_map:

                duplicate_basenames.add(
                    basename
                )

            else:

                basename_map[
                    basename
                ] = member

        if duplicate_basenames:

            raise RuntimeError(
                "Enes ZIP contains duplicate "
                "basenames; filename-only matching "
                "is not safe.\n"
                f"Examples: "
                f"{sorted(duplicate_basenames)[:10]}"
            )

        missing_files = sorted(
            set(
                metadata[
                    "file"
                ].astype(str)
            )
            - set(
                basename_map
            )
        )

        if missing_files:

            raise RuntimeError(
                "Metadata files not found "
                "inside Enes ZIP.\n"
                f"N missing: "
                f"{len(missing_files)}\n"
                f"Examples: "
                f"{missing_files[:10]}"
            )

        for i, row in (
            metadata
            .iterrows()
        ):

            filename = str(
                row[
                    "file"
                ]
            )

            try:

                raw = zf.read(
                    basename_map[
                        filename
                    ]
                )

                y, sr = sf.read(
                    io.BytesIO(
                        raw
                    ),
                    always_2d=False,
                )

                original_duration = (
                    len(y)
                    / int(sr)
                )

                y = preprocess_audio(
                    y,
                    int(sr),
                )

                features = (
                    extract_features(
                        y
                    )
                )

                rec = {
                    "dataset":
                        "enes",

                    "file":
                        filename,

                    "file_seq_S":
                        row[
                            "file_seq_S"
                        ],

                    "baby":
                        row[
                            "baby"
                        ],

                    "sex":
                        row[
                            "sex"
                        ],

                    "age_month":
                        row[
                            "age_month"
                        ],

                    "label":
                        row[
                            "label"
                        ],

                    "original_sr":
                        int(sr),

                    "original_duration_sec":
                        float(
                            original_duration
                        ),

                    "analysis_duration_sec":
                        float(
                            len(y)
                            / TARGET_SR
                        ),
                }

                rec.update(
                    features
                )

                rows.append(
                    rec
                )

            except Exception as exc:

                errors.append(
                    {
                        "dataset":
                            "enes",

                        "file":
                            filename,

                        "error":
                            repr(
                                exc
                            ),
                    }
                )

            if (
                (i + 1)
                % 500
                == 0
                or i + 1
                == len(metadata)
            ):

                print(
                    f"Enes "
                    f"{i+1}/"
                    f"{len(metadata)} "
                    f"ok={len(rows)} "
                    f"err={len(errors)}",
                    flush=True,
                )

    return (
        pd.DataFrame(
            rows
        ),
        errors,
    )


# ============================================================
# QC
# ============================================================

METADATA_COLUMNS = {
    "dataset",
    "file",
    "group_id",
    "file_seq_S",
    "baby",
    "sex",
    "age_month",
    "label",
    "original_sr",
    "original_duration_sec",
    "analysis_duration_sec",
}


def acoustic_feature_columns(
    df: pd.DataFrame,
) -> list[str]:

    return [
        c
        for c in df.columns
        if c not in METADATA_COLUMNS
    ]


def validate_counts(
    name: str,
    df: pd.DataFrame,
    expected: dict,
    errors: list[str],
):

    observed = (
        df[
            "label"
        ]
        .value_counts()
        .to_dict()
    )

    print(
        f"\n{name}"
    )

    for label, n_expected in (
        expected.items()
    ):

        n_observed = int(
            observed.get(
                label,
                0,
            )
        )

        status = (
            "OK"
            if n_observed
            == n_expected
            else "ERROR"
        )

        print(
            f"  {label:12s} "
            f"observed={n_observed:5d} "
            f"expected={n_expected:5d} "
            f"{status}"
        )

        if (
            n_observed
            != n_expected
        ):

            errors.append(
                f"{name} {label}: "
                f"{n_observed} != "
                f"{n_expected}"
            )


def validate_features(
    donate: pd.DataFrame,
    enes: pd.DataFrame,
    errors: list[str],
):

    donate_features = (
        acoustic_feature_columns(
            donate
        )
    )

    enes_features = (
        acoustic_feature_columns(
            enes
        )
    )

    print(
        "\n===== FEATURE VALIDATION ====="
    )

    print(
        "Donate acoustic features:",
        len(
            donate_features
        ),
    )

    print(
        "Enes acoustic features:",
        len(
            enes_features
        ),
    )

    if set(
        donate_features
    ) != set(
        enes_features
    ):

        errors.append(
            "Donate and Enes feature "
            "column sets differ."
        )

    if len(
        donate_features
    ) != EXPECTED_FEATURES:

        errors.append(
            "Unexpected Donate feature count: "
            f"{len(donate_features)} "
            f"(expected "
            f"{EXPECTED_FEATURES})"
        )

    if len(
        enes_features
    ) != EXPECTED_FEATURES:

        errors.append(
            "Unexpected Enes feature count: "
            f"{len(enes_features)} "
            f"(expected "
            f"{EXPECTED_FEATURES})"
        )

    forbidden = {
        "y",
        "target",
        "dataset_id",
        "sampling_rate_hz",
        "channels",
    }

    leaked = (
        forbidden
        & set(
            donate_features
        )
    ) | (
        forbidden
        & set(
            enes_features
        )
    )

    if leaked:

        errors.append(
            "Forbidden metadata/target "
            "columns detected among features: "
            f"{sorted(leaked)}"
        )

    return sorted(
        set(
            donate_features
        )
        & set(
            enes_features
        )
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

    args = parser.parse_args()

    project = (
        args.project
        .resolve()
    )

    outdir = (
        project
        / "data"
        / "processed"
        / "features_variable"
    )

    qcdir = (
        project
        / "data"
        / "metadata"
        / "feature_extraction"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    qcdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "===== VARIABLE-DURATION FEATURE EXTRACTION ====="
    )

    print(
        "\nAudio preprocessing:"
    )

    print(
        f"  target SR: "
        f"{TARGET_SR} Hz"
    )

    print(
        f"  low-pass: "
        f"{LOWPASS_HZ} Hz"
    )

    print(
        "  crop/pad: none"
    )

    print(
        "  peak normalization: none"
    )

    # --------------------------------------------------------
    # Donate
    # --------------------------------------------------------

    print(
        "\nExtracting Donate..."
    )

    (
        donate,
        donate_errors,
    ) = extract_donate(
        project
    )

    # --------------------------------------------------------
    # Enes
    # --------------------------------------------------------

    print(
        "\nExtracting Enes..."
    )

    (
        enes,
        enes_errors,
    ) = extract_enes(
        project
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    errors = []

    print(
        "\n===== COUNT VALIDATION ====="
    )

    validate_counts(
        "Donate",
        donate,
        EXPECTED_DONATE,
        errors,
    )

    validate_counts(
        "Enes",
        enes,
        EXPECTED_ENES,
        errors,
    )

    if donate_errors:

        errors.append(
            f"Donate feature extraction "
            f"errors: "
            f"{len(donate_errors)}"
        )

    if enes_errors:

        errors.append(
            f"Enes feature extraction "
            f"errors: "
            f"{len(enes_errors)}"
        )

    feature_columns = (
        validate_features(
            donate,
            enes,
            errors,
        )
    )

    if (
        enes[
            "baby"
        ].nunique()
        != EXPECTED_ENES_INFANTS
    ):

        errors.append(
            "Unexpected number "
            "of Enes infants."
        )

    if (
        enes[
            "file_seq_S"
        ].nunique()
        != EXPECTED_ENES_SESSIONS
    ):

        errors.append(
            "Unexpected number "
            "of Enes sessions."
        )

    # --------------------------------------------------------
    # Missing-feature QC
    # --------------------------------------------------------

    missing_rows = []

    for dataset_name, frame in [
        (
            "donate",
            donate,
        ),
        (
            "enes",
            enes,
        ),
    ]:

        for feature in (
            feature_columns
        ):

            n_missing = int(
                frame[
                    feature
                ]
                .isna()
                .sum()
            )

            if n_missing > 0:

                missing_rows.append(
                    {
                        "dataset":
                            dataset_name,

                        "feature":
                            feature,

                        "n_missing":
                            n_missing,

                        "fraction_missing":
                            n_missing
                            / len(
                                frame
                            ),
                    }
                )

    missing_df = pd.DataFrame(
        missing_rows,
        columns=[
            "dataset",
            "feature",
            "n_missing",
            "fraction_missing",
        ],
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    donate.to_csv(
        outdir
        / "donate_hunger_discomfort.csv",
        index=False,
    )

    enes.to_csv(
        outdir
        / "enes_hunger_discomfort.csv",
        index=False,
    )

    pd.DataFrame(
        donate_errors
    ).to_csv(
        qcdir
        / "donate_errors.csv",
        index=False,
    )

    pd.DataFrame(
        enes_errors
    ).to_csv(
        qcdir
        / "enes_errors.csv",
        index=False,
    )

    missing_df.to_csv(
        qcdir
        / "missing_features.csv",
        index=False,
    )

    feature_manifest = pd.DataFrame(
        {
            "feature":
                feature_columns,
        }
    )

    feature_manifest[
        "family"
    ] = (
        feature_manifest[
            "feature"
        ]
        .map(
            lambda x:
                (
                    "DeltaDeltaMFCC"
                    if x.startswith(
                        "ddmfcc_"
                    )
                    else
                    "DeltaMFCC"
                    if x.startswith(
                        "dmfcc_"
                    )
                    else
                    "MFCC"
                    if x.startswith(
                        "mfcc_"
                    )
                    else
                    "Chroma"
                    if x.startswith(
                        "chroma_"
                    )
                    else
                    "Pitch_Voicing"
                    if (
                        x.startswith(
                            "f0_"
                        )
                        or x.startswith(
                            "voiced_"
                        )
                    )
                    else
                    "Energy"
                    if x.startswith(
                        "rms_"
                    )
                    else
                    "Temporal_Onset"
                    if (
                        x.startswith(
                            "onset_"
                        )
                        or x
                        == "tempo_bpm"
                    )
                    else
                    "Spectral"
                    if (
                        x.startswith(
                            "spectral_"
                        )
                        or x.startswith(
                            "zcr_"
                        )
                    )
                    else
                    "Other"
                )
        )
    )

    feature_manifest.to_csv(
        qcdir
        / "feature_manifest.csv",
        index=False,
    )

    summary = {
        "target_sr":
            TARGET_SR,

        "lowpass_hz":
            LOWPASS_HZ,

        "n_fft":
            N_FFT,

        "hop_length":
            HOP_LENGTH,

        "n_acoustic_features":
            len(
                feature_columns
            ),

        "donate_rows":
            int(
                len(
                    donate
                )
            ),

        "donate_errors":
            int(
                len(
                    donate_errors
                )
            ),

        "enes_rows":
            int(
                len(
                    enes
                )
            ),

        "enes_errors":
            int(
                len(
                    enes_errors
                )
            ),

        "enes_infants":
            int(
                enes[
                    "baby"
                ].nunique()
            ),

        "enes_sessions":
            int(
                enes[
                    "file_seq_S"
                ].nunique()
            ),

        "strict_validation_passed":
            len(
                errors
            )
            == 0,

        "validation_errors":
            errors,
    }

    with (
        qcdir
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

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print(
        "\n===== FEATURE EXTRACTION SUMMARY ====="
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "\nMissing-value feature entries:",
        len(
            missing_df
        ),
    )

    if len(
        missing_df
    ):

        print(
            "\nTop missing features:"
        )

        print(
            missing_df
            .sort_values(
                "fraction_missing",
                ascending=False,
            )
            .head(20)
            .to_string(
                index=False
            )
        )

    if errors:

        print(
            "\n===== VALIDATION FAILED ====="
        )

        for error in errors:

            print(
                "-",
                error
            )

        raise RuntimeError(
            "Variable-duration feature extraction "
            "failed strict validation."
        )

    print(
        "\n04_extract_acoustic_features.py "
        "completed successfully."
    )

    print(
        "\nSaved:"
    )

    print(
        outdir
        / "donate_hunger_discomfort.csv"
    )

    print(
        outdir
        / "enes_hunger_discomfort.csv"
    )


if __name__ == "__main__":
    main()
