#!/usr/bin/env python3
"""
Prepare fixed-duration 0.25-second handcrafted acoustic features
for Donate-a-Cry and Enes.

Purpose
-------
Sensitivity analysis controlling clip duration / segmentation length.

Both datasets are processed identically:

    mono
    -> resample to 16 kHz
    -> 4 kHz low-pass filter
    -> exact center 0.25-second window
    -> no zero-padding
    -> identical feature extraction

Two feature conditions are generated:

1. original
2. cmvn
   Cepstral mean and variance normalization is applied to each MFCC
   coefficient within each clip before computing MFCC summary statistics
   and delta / delta-delta MFCCs.

Expected feature count: 617
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

WINDOW_SEC = 0.25
WINDOW_SAMPLES = int(
    TARGET_SR * WINDOW_SEC
)

N_FFT = 512
HOP_LENGTH = 128
N_MFCC = 20

EXPECTED_FEATURES = 617

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
# Summary statistics
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

    vals = [
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
        for name, value in zip(
            names,
            vals,
        )
    }


# ============================================================
# Audio processing
# ============================================================

def to_mono(
    y: np.ndarray,
) -> np.ndarray:

    if y.ndim > 1:

        y = np.mean(
            y,
            axis=1,
        )

    return y.astype(
        np.float32
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
            TARGET_SR / 2
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


def center_window(
    y: np.ndarray,
):

    if len(y) < WINDOW_SAMPLES:

        return None

    start = (
        len(y)
        - WINDOW_SAMPLES
    ) // 2

    end = (
        start
        + WINDOW_SAMPLES
    )

    window = y[
        start:end
    ]

    if len(window) != WINDOW_SAMPLES:

        raise RuntimeError(
            "Unexpected fixed-window length."
        )

    return window


# ============================================================
# Features
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


def cmvn_mfcc(
    mfcc: np.ndarray,
) -> np.ndarray:
    """
    Per-clip, per-coefficient cepstral mean and variance normalization.
    """

    x = (
        mfcc
        - mfcc.mean(
            axis=1,
            keepdims=True,
        )
    )

    sd = x.std(
        axis=1,
        keepdims=True,
    )

    sd = np.where(
        sd < 1e-8,
        1.0,
        sd,
    )

    return x / sd


def extract_features(
    y: np.ndarray,
    use_cmvn: bool,
) -> dict:

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
    # Zero crossing rate
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
    # Spectral descriptors
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
    # MFCC
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

    if use_cmvn:

        mfcc = cmvn_mfcc(
            mfcc
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
    # Temporal onset
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

    except Exception:

        out.update(
            summary_stats(
                "onset_strength",
                [],
            )
        )

    # NOTE:
    # tempo_bpm is intentionally not included in fixed-0.25-s features.

    return out


# ============================================================
# Donate
# ============================================================

def prepare_donate(
    project: Path,
):

    metadata_path = (
        project
        / "data"
        / "processed"
        / "donate"
        / "metadata.csv"
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
    ].reset_index(
        drop=True
    )

    original_rows = []
    cmvn_rows = []
    errors = []
    skipped_short = []

    for i, row in (
        metadata.iterrows()
    ):

        try:

            path = (
                project
                / row[
                    "audio_relpath"
                ]
            )

            y, sr = sf.read(
                path,
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

            window = center_window(
                y
            )

            if window is None:

                skipped_short.append(
                    {
                        "dataset":
                            "donate",

                        "file":
                            row[
                                "filename"
                            ],

                        "duration_sec":
                            original_duration,
                    }
                )

                continue

            base = {
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
                    WINDOW_SEC,
            }

            rec = dict(
                base
            )

            rec.update(
                extract_features(
                    window,
                    use_cmvn=False,
                )
            )

            original_rows.append(
                rec
            )

            rec = dict(
                base
            )

            rec.update(
                extract_features(
                    window,
                    use_cmvn=True,
                )
            )

            cmvn_rows.append(
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
            (i + 1) % 50 == 0
            or i + 1
            == len(metadata)
        ):

            print(
                f"Donate "
                f"{i+1}/"
                f"{len(metadata)} "
                f"retained="
                f"{len(original_rows)} "
                f"short="
                f"{len(skipped_short)} "
                f"errors="
                f"{len(errors)}",
                flush=True,
            )

    return (
        pd.DataFrame(
            original_rows
        ),
        pd.DataFrame(
            cmvn_rows
        ),
        skipped_short,
        errors,
    )


# ============================================================
# Enes
# ============================================================

def prepare_enes(
    project: Path,
):

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
    ].reset_index(
        drop=True
    )

    original_rows = []
    cmvn_rows = []
    errors = []
    skipped_short = []

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as zf:

        basename_map = {}

        duplicate_basenames = set()

        for member in (
            zf.namelist()
        ):

            if member.endswith("/"):
                continue

            basename = (
                Path(member).name
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
                "Duplicate basenames detected "
                "inside Enes ZIP."
            )

        missing = sorted(
            set(
                metadata[
                    "file"
                ].astype(str)
            )
            - set(
                basename_map
            )
        )

        if missing:

            raise RuntimeError(
                f"{len(missing)} Enes metadata "
                "files are missing from ZIP."
            )

        for i, row in (
            metadata.iterrows()
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

                window = center_window(
                    y
                )

                if window is None:

                    skipped_short.append(
                        {
                            "dataset":
                                "enes",

                            "file":
                                filename,

                            "baby":
                                row[
                                    "baby"
                                ],

                            "file_seq_S":
                                row[
                                    "file_seq_S"
                                ],

                            "label":
                                row[
                                    "label"
                                ],

                            "duration_sec":
                                original_duration,
                        }
                    )

                    continue

                base = {
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
                        WINDOW_SEC,
                }

                rec = dict(
                    base
                )

                rec.update(
                    extract_features(
                        window,
                        use_cmvn=False,
                    )
                )

                original_rows.append(
                    rec
                )

                rec = dict(
                    base
                )

                rec.update(
                    extract_features(
                        window,
                        use_cmvn=True,
                    )
                )

                cmvn_rows.append(
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
                (i + 1) % 500
                == 0
                or i + 1
                == len(metadata)
            ):

                print(
                    f"Enes "
                    f"{i+1}/"
                    f"{len(metadata)} "
                    f"retained="
                    f"{len(original_rows)} "
                    f"short="
                    f"{len(skipped_short)} "
                    f"errors="
                    f"{len(errors)}",
                    flush=True,
                )

    return (
        pd.DataFrame(
            original_rows
        ),
        pd.DataFrame(
            cmvn_rows
        ),
        skipped_short,
        errors,
    )


# ============================================================
# Validation
# ============================================================

META_COLUMNS = {
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


def feature_columns(
    df: pd.DataFrame,
) -> list[str]:

    return [
        c
        for c in df.columns
        if c not in META_COLUMNS
    ]


def validate_counts(
    name,
    df,
    expected,
    errors,
):

    observed = (
        df[
            "label"
        ]
        .value_counts()
        .to_dict()
    )

    for label, expected_n in (
        expected.items()
    ):

        observed_n = int(
            observed.get(
                label,
                0,
            )
        )

        print(
            f"{name:8s} "
            f"{label:12s} "
            f"observed={observed_n:5d} "
            f"expected={expected_n:5d} "
            f"{'OK' if observed_n == expected_n else 'ERROR'}"
        )

        if observed_n != expected_n:

            errors.append(
                f"{name} {label}: "
                f"{observed_n} != {expected_n}"
            )


def validate_condition_pair(
    original,
    cmvn,
    dataset_name,
    errors,
):

    if len(original) != len(cmvn):

        errors.append(
            f"{dataset_name}: "
            "Original / CMVN row counts differ."
        )

    f_orig = feature_columns(
        original
    )

    f_cmvn = feature_columns(
        cmvn
    )

    if set(f_orig) != set(f_cmvn):

        errors.append(
            f"{dataset_name}: "
            "Original / CMVN feature columns differ."
        )

    if len(f_orig) != EXPECTED_FEATURES:

        errors.append(
            f"{dataset_name}: "
            f"feature count={len(f_orig)}, "
            f"expected={EXPECTED_FEATURES}"
        )

    duration_ok = (
        pd.to_numeric(
            original[
                "analysis_duration_sec"
            ],
            errors="coerce",
        )
        .eq(
            WINDOW_SEC
        )
        .all()
    )

    if not duration_ok:

        errors.append(
            f"{dataset_name}: "
            "not all analysis durations are 0.25 s."
        )

    return f_orig


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
        args.project.resolve()
    )

    outdir = (
        project
        / "data"
        / "processed"
        / "fixed025"
    )

    qcdir = (
        project
        / "data"
        / "metadata"
        / "fixed025"
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
        "===== FIXED 0.25 s DATASET PREPARATION ====="
    )

    print(
        f"Target sampling rate: "
        f"{TARGET_SR} Hz"
    )

    print(
        f"Low-pass cutoff: "
        f"{LOWPASS_HZ} Hz"
    )

    print(
        f"Window: "
        f"{WINDOW_SEC} s "
        f"({WINDOW_SAMPLES} samples)"
    )

    print(
        "Padding: none"
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    print(
        "\nPreparing Donate..."
    )

    (
        donate_original,
        donate_cmvn,
        donate_short,
        donate_errors,
    ) = prepare_donate(
        project
    )

    print(
        "\nPreparing Enes..."
    )

    (
        enes_original,
        enes_cmvn,
        enes_short,
        enes_errors,
    ) = prepare_enes(
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
        donate_original,
        EXPECTED_DONATE,
        errors,
    )

    validate_counts(
        "Enes",
        enes_original,
        EXPECTED_ENES,
        errors,
    )

    donate_features = (
        validate_condition_pair(
            donate_original,
            donate_cmvn,
            "Donate",
            errors,
        )
    )

    enes_features = (
        validate_condition_pair(
            enes_original,
            enes_cmvn,
            "Enes",
            errors,
        )
    )

    if set(
        donate_features
    ) != set(
        enes_features
    ):

        errors.append(
            "Donate and Enes acoustic "
            "feature sets differ."
        )

    if donate_short:

        errors.append(
            f"Donate clips shorter than "
            f"0.25 s: {len(donate_short)}"
        )

    if enes_short:

        errors.append(
            f"Enes clips shorter than "
            f"0.25 s: {len(enes_short)}"
        )

    if donate_errors:

        errors.append(
            f"Donate extraction errors: "
            f"{len(donate_errors)}"
        )

    if enes_errors:

        errors.append(
            f"Enes extraction errors: "
            f"{len(enes_errors)}"
        )

    if (
        enes_original[
            "baby"
        ].nunique()
        != EXPECTED_ENES_INFANTS
    ):

        errors.append(
            "Unexpected Enes infant count."
        )

    if (
        enes_original[
            "file_seq_S"
        ].nunique()
        != EXPECTED_ENES_SESSIONS
    ):

        errors.append(
            "Unexpected Enes session count."
        )

    # --------------------------------------------------------
    # Save datasets
    # --------------------------------------------------------

    donate_original.to_csv(
        outdir
        / "donate_original.csv",
        index=False,
    )

    donate_cmvn.to_csv(
        outdir
        / "donate_cmvn.csv",
        index=False,
    )

    enes_original.to_csv(
        outdir
        / "enes_original.csv",
        index=False,
    )

    enes_cmvn.to_csv(
        outdir
        / "enes_cmvn.csv",
        index=False,
    )

    pd.DataFrame(
        donate_short
    ).to_csv(
        qcdir
        / "donate_skipped_short.csv",
        index=False,
    )

    pd.DataFrame(
        enes_short
    ).to_csv(
        qcdir
        / "enes_skipped_short.csv",
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

    # --------------------------------------------------------
    # Feature manifest
    # --------------------------------------------------------

    feature_manifest = pd.DataFrame(
        {
            "feature":
                sorted(
                    set(
                        donate_features
                    )
                )
        }
    )

    feature_manifest.to_csv(
        qcdir
        / "feature_manifest.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "target_sr":
            TARGET_SR,

        "lowpass_hz":
            LOWPASS_HZ,

        "window_sec":
            WINDOW_SEC,

        "window_samples":
            WINDOW_SAMPLES,

        "n_fft":
            N_FFT,

        "hop_length":
            HOP_LENGTH,

        "n_features":
            len(
                donate_features
            ),

        "donate_retained":
            len(
                donate_original
            ),

        "donate_skipped_short":
            len(
                donate_short
            ),

        "donate_errors":
            len(
                donate_errors
            ),

        "enes_retained":
            len(
                enes_original
            ),

        "enes_skipped_short":
            len(
                enes_short
            ),

        "enes_errors":
            len(
                enes_errors
            ),

        "enes_infants":
            int(
                enes_original[
                    "baby"
                ].nunique()
            ),

        "enes_sessions":
            int(
                enes_original[
                    "file_seq_S"
                ].nunique()
            ),

        "strict_validation_passed":
            len(errors) == 0,

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

    print(
        "\n===== FIXED 0.25 s SUMMARY ====="
    )

    print(
        json.dumps(
            summary,
            indent=2,
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
            "Fixed 0.25-second dataset "
            "preparation failed."
        )

    print(
        "\n05_prepare_fixed025.py "
        "completed successfully."
    )

    print(
        "\nSaved:"
    )

    for filename in [
        "donate_original.csv",
        "donate_cmvn.csv",
        "enes_original.csv",
        "enes_cmvn.csv",
    ]:

        print(
            outdir
            / filename
        )


if __name__ == "__main__":
    main()
