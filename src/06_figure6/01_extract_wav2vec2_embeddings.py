#!/usr/bin/env python3
"""
Figure 6:
Extract variable-duration wav2vec2 embeddings from Donate-a-Cry and Enes.

Primary analysis subset
-----------------------
Donate-a-Cry:
    hunger      382
    discomfort   27
    total       409

Enes:
    hunger      9891
    discomfort  9246
    total      19137

Representation
--------------
facebook/wav2vec2-base-960h
Last hidden state -> masked mean pooling over time
Expected embedding dimension: 768

Audio handling
--------------
- Uses original / variable-duration audio
- Resamples to 16 kHz as required by wav2vec2
- Mono conversion if necessary
- No fixed-duration cropping
- No 4-kHz low-pass filtering for the primary representation comparison

The script:
- automatically resolves audio from filesystem paths or ZIP members
- runs in batches
- uses CUDA when available
- saves checkpoints
- supports resume
- validates expected row counts and embedding dimension

Outputs
-------
data/processed/wav2vec2_variable/
    donate_embeddings.csv
    enes_embeddings.csv
    extraction_summary.json

results/06_figure6/
    (created later by analysis scripts)
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from scipy.signal import resample_poly
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2Model,
)


# ============================================================
# Constants
# ============================================================

MODEL_NAME = "facebook/wav2vec2-base-960h"

TARGET_SR = 16000

EXPECTED_DONATE = 409
EXPECTED_ENES = 19137
EXPECTED_DIM = 768

LABELS = {
    "hunger",
    "discomfort",
}


# ============================================================
# Paths / audio resolver
# ============================================================

def find_existing_path(
    value,
    project: Path,
):
    """
    Resolve a metadata path into an existing local file.
    """

    if pd.isna(value):
        return None

    value = str(value)

    candidates = [
        Path(value),
        project / value,
        project / "data" / value,
        project / "data" / "raw" / value,
        project / "data" / "processed" / value,
    ]

    for path in candidates:

        if path.exists() and path.is_file():
            return path.resolve()

    return None


def candidate_path_columns(df):
    """
    Metadata columns likely to contain an audio path.
    """

    preferred = [
        "source_audio_resolved",
        "audio_path",
        "audio_relpath",
        "path",
        "filepath",
        "file_path",
        "model_audio_path",
        "file",
    ]

    return [
        c
        for c in preferred
        if c in df.columns
    ]


class ZipAudioReader:
    """
    Cache ZipFile handles and resolve ZIP members.
    """

    def __init__(self):
        self.handles = {}

    def get_zip(self, path: Path):

        key = str(path.resolve())

        if key not in self.handles:
            self.handles[key] = zipfile.ZipFile(
                path,
                "r",
            )

        return self.handles[key]

    def close(self):

        for handle in self.handles.values():
            handle.close()

        self.handles = {}


# ============================================================
# Audio loading
# ============================================================

def to_mono(audio):

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.ndim == 1:
        return audio

    if audio.ndim == 2:
        return audio.mean(
            axis=1
        ).astype(
            np.float32
        )

    raise ValueError(
        f"Unexpected audio shape: {audio.shape}"
    )


def resample_audio(
    audio,
    original_sr,
    target_sr=TARGET_SR,
):

    if original_sr == target_sr:
        return audio.astype(
            np.float32
        )

    gcd = np.gcd(
        int(original_sr),
        int(target_sr),
    )

    up = target_sr // gcd
    down = original_sr // gcd

    output = resample_poly(
        audio,
        up,
        down,
    )

    return output.astype(
        np.float32
    )


def read_soundfile_bytes(
    data: bytes,
):

    audio, sr = sf.read(
        io.BytesIO(data),
        dtype="float32",
        always_2d=False,
    )

    return (
        to_mono(audio),
        int(sr),
    )


def read_soundfile_path(
    path: Path,
):

    audio, sr = sf.read(
        path,
        dtype="float32",
        always_2d=False,
    )

    return (
        to_mono(audio),
        int(sr),
    )


# ============================================================
# ZIP member resolution
# ============================================================

def locate_zip_member(
    row,
    zip_path: Path,
    zip_reader: ZipAudioReader,
):
    """
    Resolve metadata row to ZIP member.

    Preference:
    1. source_zip_member
    2. exact 'file'
    3. basename match

    Basename matching is used only if unique.
    """

    zf = zip_reader.get_zip(
        zip_path
    )

    names = zf.namelist()

    if (
        "source_zip_member"
        in row.index
        and pd.notna(
            row[
                "source_zip_member"
            ]
        )
    ):

        member = str(
            row[
                "source_zip_member"
            ]
        )

        if member in names:
            return member

    if (
        "file"
        in row.index
        and pd.notna(
            row["file"]
        )
    ):

        file_value = str(
            row["file"]
        )

        if file_value in names:
            return file_value

        target_name = Path(
            file_value
        ).name

        matches = [
            n
            for n in names
            if Path(n).name
            == target_name
        ]

        if len(matches) == 1:
            return matches[0]

    raise FileNotFoundError(
        "Could not uniquely resolve ZIP member "
        f"for row file={row.get('file', None)}"
    )


# ============================================================
# Dataset-specific loading
# ============================================================

def load_audio_from_row(
    row,
    dataset,
    project,
    zip_reader,
    donate_zip,
    enes_zip,
):

    # --------------------------------------------------------
    # First try explicit filesystem paths
    # --------------------------------------------------------

    for col in [
        "source_audio_resolved",
        "audio_path",
        "audio_relpath",
        "path",
        "filepath",
        "file_path",
        "file",
    ]:

        if (
            col in row.index
            and pd.notna(
                row[col]
            )
        ):

            resolved = find_existing_path(
                row[col],
                project,
            )

            if resolved is not None:

                audio, sr = (
                    read_soundfile_path(
                        resolved
                    )
                )

                return (
                    audio,
                    sr,
                    f"path:{resolved}",
                )

    # --------------------------------------------------------
    # ZIP fallback
    # --------------------------------------------------------

    zip_path = (
        donate_zip
        if dataset == "donate"
        else enes_zip
    )

    member = locate_zip_member(
        row,
        zip_path,
        zip_reader,
    )

    zf = zip_reader.get_zip(
        zip_path
    )

    data = zf.read(
        member
    )

    audio, sr = read_soundfile_bytes(
        data
    )

    return (
        audio,
        sr,
        f"zip:{zip_path.name}:{member}",
    )


# ============================================================
# Embedding model
# ============================================================

def masked_mean_pool(
    hidden_state,
    attention_mask,
):
    """
    Pool wav2vec2 hidden representations while ignoring padding.

    hidden_state:
        [batch, time_hidden, hidden_dim]

    attention_mask:
        [batch, time_input]
    """

    if attention_mask is None:

        return hidden_state.mean(
            dim=1
        )

    # Convert input-level attention mask to output feature length.
    output_lengths = (
        model._get_feat_extract_output_lengths(
            attention_mask.sum(
                dim=1
            )
        )
    )

    batch_size = hidden_state.shape[0]
    hidden_length = hidden_state.shape[1]

    mask = torch.zeros(
        (
            batch_size,
            hidden_length,
        ),
        dtype=hidden_state.dtype,
        device=hidden_state.device,
    )

    for i, length in enumerate(
        output_lengths
    ):

        length = int(
            length.item()
        )

        if length > 0:
            mask[
                i,
                :min(
                    length,
                    hidden_length,
                )
            ] = 1.0

    denominator = (
        mask.sum(
            dim=1,
            keepdim=True,
        )
        .clamp(
            min=1.0
        )
    )

    pooled = (
        hidden_state
        * mask.unsqueeze(
            -1
        )
    ).sum(
        dim=1
    ) / denominator

    return pooled


# model is assigned inside main()
model = None


# ============================================================
# Batch extraction
# ============================================================

def extract_batch(
    audio_list,
    feature_extractor,
    device,
):

    inputs = feature_extractor(
        audio_list,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )

    input_values = (
        inputs[
            "input_values"
        ]
        .to(device)
    )

    attention_mask = (
        inputs[
            "attention_mask"
        ]
        .to(device)
    )

    with torch.inference_mode():

        outputs = model(
            input_values=input_values,
            attention_mask=attention_mask,
        )

        pooled = masked_mean_pool(
            outputs.last_hidden_state,
            attention_mask,
        )

    return (
        pooled
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )


# ============================================================
# Dataset extraction
# ============================================================

def extract_dataset(
    df,
    dataset,
    project,
    output_path,
    feature_extractor,
    device,
    batch_size,
    checkpoint_every,
    donate_zip,
    enes_zip,
):

    print(
        "\n========================================"
    )
    print(
        f"Extracting: {dataset}"
    )
    print(
        "Rows:",
        len(df)
    )
    print(
        "========================================"
    )

    zip_reader = ZipAudioReader()

    completed = {}

    if output_path.exists():

        old = pd.read_csv(
            output_path
        )

        if (
            "file"
            in old.columns
        ):

            completed = {
                str(row["file"]):
                    row
                for _, row in old.iterrows()
            }

            print(
                "Resume:",
                len(completed),
                "rows already saved",
            )

    rows = []

    if completed:

        rows.extend(
            [
                dict(x)
                for x in completed.values()
            ]
        )

    embedding_columns = [
        f"w2v_{i:03d}"
        for i in range(
            EXPECTED_DIM
        )
    ]

    pending_audio = []
    pending_metadata = []

    n_errors = 0
    n_processed_new = 0

    def flush_batch():

        nonlocal pending_audio, pending_metadata, n_processed_new

        if not pending_audio:
            return

        embeddings = extract_batch(
            pending_audio,
            feature_extractor,
            device,
        )

        if (
            embeddings.ndim != 2
            or embeddings.shape[1]
            != EXPECTED_DIM
        ):

            raise RuntimeError(
                "Unexpected wav2vec2 embedding shape: "
                f"{embeddings.shape}"
            )

        for metadata, embedding in zip(
            pending_metadata,
            embeddings,
        ):

            output_row = dict(
                metadata
            )

            for column, value in zip(
                embedding_columns,
                embedding,
            ):

                output_row[
                    column
                ] = float(
                    value
                )

            rows.append(
                output_row
            )

            n_processed_new += 1

        pending_audio = []
        pending_metadata = []

    for idx, row in df.iterrows():

        file_id = str(
            row.get(
                "file",
                idx,
            )
        )

        if file_id in completed:
            continue

        try:

            audio, original_sr, source = (
                load_audio_from_row(
                    row,
                    dataset,
                    project,
                    zip_reader,
                    donate_zip,
                    enes_zip,
                )
            )

            original_duration = (
                len(audio)
                / original_sr
            )

            audio = resample_audio(
                audio,
                original_sr,
                TARGET_SR,
            )

            # wav2vec2 needs non-empty input
            if len(audio) < 400:

                raise ValueError(
                    f"Audio too short after resampling: "
                    f"{len(audio)} samples"
                )

            metadata = {
                "dataset":
                    dataset,

                "file":
                    file_id,

                "label":
                    str(
                        row[
                            "label"
                        ]
                    ),

                "original_sr":
                    int(
                        original_sr
                    ),

                "duration_sec":
                    float(
                        original_duration
                    ),

                "audio_source":
                    source,
            }

            for col in [
                "group_id",
                "file_seq_S",
                "baby",
                "sex",
                "age_month",
            ]:

                if col in row.index:
                    metadata[
                        col
                    ] = row[
                        col
                    ]

            pending_audio.append(
                audio
            )

            pending_metadata.append(
                metadata
            )

            if len(
                pending_audio
            ) >= batch_size:

                flush_batch()

        except Exception as exc:

            n_errors += 1

            print(
                "\nERROR",
                dataset,
                file_id,
                repr(exc),
                flush=True,
            )

        n_seen = (
            idx + 1
        )

        if (
            n_seen % 250
            == 0
        ):

            print(
                f"{n_seen}/{len(df)} "
                f"new={n_processed_new} "
                f"errors={n_errors}",
                flush=True,
            )

        if (
            checkpoint_every > 0
            and n_processed_new > 0
            and n_processed_new
            % checkpoint_every
            == 0
        ):

            flush_batch()

            pd.DataFrame(
                rows
            ).to_csv(
                output_path,
                index=False,
            )

            print(
                "Checkpoint:",
                len(rows),
                output_path,
                flush=True,
            )

    flush_batch()

    out = pd.DataFrame(
        rows
    )

    if (
        "file"
        in out.columns
    ):

        out = (
            out
            .drop_duplicates(
                subset=[
                    "file"
                ],
                keep="last",
            )
        )

    out.to_csv(
        output_path,
        index=False,
    )

    zip_reader.close()

    print(
        "\nFinished:",
        dataset
    )
    print(
        "Rows saved:",
        len(out)
    )
    print(
        "Errors:",
        n_errors
    )

    return (
        out,
        n_errors,
    )


# ============================================================
# Metadata preparation
# ============================================================

def load_donate_metadata(
    project,
):

    path = (
        project
        / "data"
        / "processed"
        / "donate"
        / "metadata.csv"
    )

    df = pd.read_csv(
        path
    )

    df = df[
        df[
            "label"
        ].isin(
            LABELS
        )
    ].copy()

    return (
        df
        .reset_index(
            drop=True
        )
    )


def load_enes_metadata(
    project,
):

    path = (
        project
        / "data"
        / "processed"
        / "enes"
        / "hunger_discomfort_metadata.csv"
    )

    df = pd.read_csv(
        path
    )

    df = df[
        df[
            "label"
        ].isin(
            LABELS
        )
    ].copy()

    return (
        df
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Validation
# ============================================================

def validate_output(
    df,
    dataset,
    expected_rows,
):

    problems = []

    if len(df) != expected_rows:

        problems.append(
            f"{dataset}: "
            f"rows={len(df)}, "
            f"expected={expected_rows}"
        )

    if (
        df["file"]
        .nunique()
        != len(df)
    ):

        problems.append(
            f"{dataset}: duplicate file IDs"
        )

    expected_labels = (
        {
            "hunger": 382,
            "discomfort": 27,
        }
        if dataset == "donate"
        else
        {
            "hunger": 9891,
            "discomfort": 9246,
        }
    )

    observed = (
        df[
            "label"
        ]
        .value_counts()
        .to_dict()
    )

    for label, n_expected in (
        expected_labels.items()
    ):

        if int(
            observed.get(
                label,
                0,
            )
        ) != n_expected:

            problems.append(
                f"{dataset}/{label}: "
                f"{observed.get(label, 0)} "
                f"!= {n_expected}"
            )

    embedding_cols = [
        c
        for c in df.columns
        if c.startswith(
            "w2v_"
        )
    ]

    if len(
        embedding_cols
    ) != EXPECTED_DIM:

        problems.append(
            f"{dataset}: "
            f"{len(embedding_cols)} embedding dims "
            f"!= {EXPECTED_DIM}"
        )

    if embedding_cols:

        values = (
            df[
                embedding_cols
            ]
            .to_numpy(
                dtype=float
            )
        )

        if not np.isfinite(
            values
        ).all():

            problems.append(
                f"{dataset}: NaN/Inf in embeddings"
            )

    return problems


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project",
        type=Path,
        default=Path(
            str(Path(__file__).resolve().parents[2])
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
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
        / "wav2vec2_variable"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    donate_zip = (
        project
        / "data"
        / "raw"
        / "donateacry-corpus-master.zip"
    )

    enes_zip = (
        project
        / "data"
        / "raw"
        / "audio_EnesBabyCries1B_cries.zip"
    )

    if not donate_zip.exists():

        raise FileNotFoundError(
            donate_zip
        )

    if not enes_zip.exists():

        raise FileNotFoundError(
            enes_zip
        )

    device = torch.device(
        "cpu"
        if args.cpu
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    print(
        "Device:",
        device
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "Loading:",
        MODEL_NAME,
        flush=True,
    )

    feature_extractor = (
        AutoFeatureExtractor
        .from_pretrained(
            MODEL_NAME
        )
    )

    global model

    model = (
        Wav2Vec2Model
        .from_pretrained(
            MODEL_NAME
        )
        .to(
            device
        )
    )

    model.eval()

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    donate = load_donate_metadata(
        project
    )

    enes = load_enes_metadata(
        project
    )

    print(
        "\n===== INPUT ====="
    )

    print(
        "Donate:",
        len(donate)
    )

    print(
        donate[
            "label"
        ].value_counts()
    )

    print(
        "\nEnes:",
        len(enes)
    )

    print(
        enes[
            "label"
        ].value_counts()
    )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    donate_out, donate_errors = (
        extract_dataset(
            donate,
            "donate",
            project,
            outdir
            / "donate_embeddings.csv",
            feature_extractor,
            device,
            args.batch_size,
            args.checkpoint_every,
            donate_zip,
            enes_zip,
        )
    )

    enes_out, enes_errors = (
        extract_dataset(
            enes,
            "enes",
            project,
            outdir
            / "enes_embeddings.csv",
            feature_extractor,
            device,
            args.batch_size,
            args.checkpoint_every,
            donate_zip,
            enes_zip,
        )
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    problems = []

    problems += validate_output(
        donate_out,
        "donate",
        EXPECTED_DONATE,
    )

    problems += validate_output(
        enes_out,
        "enes",
        EXPECTED_ENES,
    )

    summary = {
        "model":
            MODEL_NAME,

        "target_sampling_rate":
            TARGET_SR,

        "pooling":
            "masked_mean_last_hidden_state",

        "embedding_dimension":
            EXPECTED_DIM,

        "donate_rows":
            len(
                donate_out
            ),

        "enes_rows":
            len(
                enes_out
            ),

        "donate_errors":
            donate_errors,

        "enes_errors":
            enes_errors,

        "device":
            str(
                device
            ),

        "validation_passed":
            len(
                problems
            )
            == 0,

        "validation_errors":
            problems,
    }

    with (
        outdir
        / "extraction_summary.json"
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
        "\n===== WAV2VEC2 EXTRACTION SUMMARY ====="
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    if problems:

        raise RuntimeError(
            "wav2vec2 extraction validation failed:\n"
            + "\n".join(
                problems
            )
        )

    print(
        "\nWav2vec2 extraction completed successfully."
    )


if __name__ == "__main__":
    main()
