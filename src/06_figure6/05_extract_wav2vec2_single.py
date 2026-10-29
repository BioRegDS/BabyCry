#!/usr/bin/env python3
"""Extract wav2vec2 embeddings one audio at a time, without padding."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2Model


def load_original(path):
    spec = importlib.util.spec_from_file_location(
        "wav2vec2_original", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--dataset",
        choices=["donate", "enes", "both"],
        default="donate",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=100
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    project = args.project.resolve()

    original = load_original(
        project / "src/06_figure6/01_extract_wav2vec2_embeddings.py"
    )

    device = torch.device(
        "cpu" if args.cpu else
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device, flush=True)

    fe = AutoFeatureExtractor.from_pretrained(
        original.MODEL_NAME
    )

    model = Wav2Vec2Model.from_pretrained(
        original.MODEL_NAME
    ).to(device)
    model.eval()

    # Replace the original batch extraction function.
    # No padding and no attention mask.
    def extract_single(audio_list, feature_extractor, device):
        if len(audio_list) != 1:
            raise RuntimeError(
                "Single-audio extraction requires batch_size=1"
            )

        audio = audio_list[0]

        inputs = feature_extractor(
            audio,
            sampling_rate=original.TARGET_SR,
            return_tensors="pt",
            padding=False,
            return_attention_mask=False,
        )

        with torch.inference_mode():
            hidden = model(
                input_values=inputs["input_values"].to(device)
            ).last_hidden_state

            embedding = hidden.mean(dim=1)

        result = embedding.cpu().numpy().astype(np.float32)

        if result.shape != (1, original.EXPECTED_DIM):
            raise RuntimeError(
                f"Unexpected embedding shape: {result.shape}"
            )

        if not np.isfinite(result).all():
            raise RuntimeError("Non-finite embedding detected")

        return result

    original.extract_batch = extract_single

    outdir = project / "data/processed/wav2vec2_single"
    outdir.mkdir(parents=True, exist_ok=True)

    donate_zip = (
        project / "data/raw/donateacry-corpus-master.zip"
    )
    enes_zip = (
        project / "data/raw/audio_EnesBabyCries1B_cries.zip"
    )

    datasets = (
        ["donate", "enes"]
        if args.dataset == "both"
        else [args.dataset]
    )

    summary = {}

    for dataset in datasets:
        if dataset == "donate":
            df = original.load_donate_metadata(project)
            expected = original.EXPECTED_DONATE
        else:
            df = original.load_enes_metadata(project)
            expected = original.EXPECTED_ENES

        output_path = outdir / f"{dataset}_embeddings.csv"

        output, errors = original.extract_dataset(
            df=df,
            dataset=dataset,
            project=project,
            output_path=output_path,
            feature_extractor=fe,
            device=device,
            batch_size=1,
            checkpoint_every=args.checkpoint_every,
            donate_zip=donate_zip,
            enes_zip=enes_zip,
        )

        problems = original.validate_output(
            output, dataset, expected
        )

        summary[dataset] = {
            "rows": len(output),
            "expected": expected,
            "errors": errors,
            "validation_errors": problems,
            "output": str(output_path),
        }

        print(
            json.dumps(
                summary[dataset],
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )

        if problems or errors:
            raise RuntimeError(
                f"Validation failed for {dataset}: {problems}"
            )

    with (outdir / "extraction_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "model": original.MODEL_NAME,
                "method": "single_audio_no_padding_no_attention_mask",
                "sampling_rate": original.TARGET_SR,
                "pooling": "mean_last_hidden_state",
                "datasets": summary,
            },
            f,
            indent=2,
        )

    print("\nExtraction completed successfully.")


if __name__ == "__main__":
    main()
