#!/usr/bin/env python3
"""Compare wav2vec2 embeddings with different padding/masking strategies."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoFeatureExtractor, Wav2Vec2Model


MODEL_NAME = "facebook/wav2vec2-base-960h"


def load_original(path):
    spec = importlib.util.spec_from_file_location(
        "wav2vec2_original", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cosine(a, b):
    return float(
        np.dot(a, b) /
        (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    )


def embed_single(audio, fe, model, device):
    inputs = fe(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=False,
        return_attention_mask=False,
    )

    with torch.inference_mode():
        hidden = model(
            input_values=inputs["input_values"].to(device)
        ).last_hidden_state

    return hidden.mean(dim=1).cpu().numpy()[0]


def embed_batch(audio_list, fe, model, device, use_mask):
    inputs = fe(
        audio_list,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )

    values = inputs["input_values"].to(device)
    mask = inputs["attention_mask"].to(device)

    kwargs = {"input_values": values}
    if use_mask:
        kwargs["attention_mask"] = mask

    with torch.inference_mode():
        hidden = model(**kwargs).last_hidden_state

        lengths = model._get_feat_extract_output_lengths(
            mask.sum(dim=1)
        )

        pooled = []
        for i, length in enumerate(lengths):
            n = int(length.item())
            pooled.append(hidden[i, :n].mean(dim=0))

    return torch.stack(pooled).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--n", type=int, default=12)
    args = parser.parse_args()

    project = args.project.resolve()
    original = load_original(
        project / "src/06_figure6/01_extract_wav2vec2_embeddings.py"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    fe = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    zip_reader = original.ZipAudioReader()

    audio_list = []
    metadata = []

    try:
        for dataset in ["donate", "enes"]:
            if dataset == "donate":
                df = original.load_donate_metadata(project)
            else:
                df = original.load_enes_metadata(project)

            subset = df.sample(
                n=min(args.n // 2, len(df)),
                random_state=42,
            )

            for _, row in subset.iterrows():
                audio, sr, source = original.load_audio_from_row(
                    row,
                    dataset,
                    project,
                    zip_reader,
                    project / "data/raw/donateacry-corpus-master.zip",
                    project / "data/raw/audio_EnesBabyCries1B_cries.zip",
                )

                audio = original.resample_audio(audio, sr, 16000)

                if len(audio) < 400:
                    continue

                audio_list.append(audio)
                metadata.append({
                    "dataset": dataset,
                    "file": str(
                        row.get(
                            "file",
                            row.get("source_zip_member", f"{dataset}_{len(metadata)}")
                        )
                    ),
                    "duration_sec": len(audio) / 16000,
                })
    finally:
        zip_reader.close()

    if not audio_list:
        raise RuntimeError("No valid audio loaded")

    single = np.stack([
        embed_single(a, fe, model, device)
        for a in audio_list
    ])

    batch_mask = embed_batch(
        audio_list, fe, model, device, use_mask=True
    )

    batch_no_mask = embed_batch(
        audio_list, fe, model, device, use_mask=False
    )

    rows = []

    for i, meta in enumerate(metadata):
        for method, embeddings in [
            ("batch_with_mask", batch_mask),
            ("batch_without_mask", batch_no_mask),
        ]:
            rows.append({
                **meta,
                "method": method,
                "cosine_similarity": cosine(
                    single[i], embeddings[i]
                ),
                "mae": float(
                    np.mean(np.abs(single[i] - embeddings[i]))
                ),
            })

    result = pd.DataFrame(rows)

    outdir = project / "results/06_figure6/batch_effect_qc"
    outdir.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        outdir / "batch_effect_comparison.csv",
        index=False,
    )

    print("\n=== BATCH EFFECT SUMMARY ===")
    print(
        result.groupby("method")[
            ["cosine_similarity", "mae"]
        ].agg(["mean", "min", "max"])
        .to_string()
    )

    print("\n=== INDIVIDUAL RESULTS ===")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
