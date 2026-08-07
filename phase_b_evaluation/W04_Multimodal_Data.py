"""Build the frozen Week 4 public-image input bank and attribution record.

Only the 20 standardized clean images are stored. The runner deterministically
applies one perturbation in memory from the clean pixels, and validates the
result against the expected pixel hash recorded here. This avoids committing
three redundant image copies while preserving exact evaluated inputs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageOps


BUILDER_VERSION = "0.1.1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    header = f"RGB:{rgb.width}x{rgb.height}:".encode("ascii")
    return hashlib.sha256(header + rgb.tobytes()).hexdigest()


def deterministic_condition_seed(
    global_seed: int,
    scenario_id: str,
    condition_id: str,
) -> int:
    digest = hashlib.sha256(
        f"{global_seed}:{scenario_id}:{condition_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def standardize_image(
    image: Image.Image,
    *,
    width: int,
    height: int,
    padding_rgb: tuple[int, int, int],
) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    contained = ImageOps.contain(
        image,
        (width, height),
        method=Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (width, height), padding_rgb)
    left = (width - contained.width) // 2
    top = (height - contained.height) // 2
    canvas.paste(contained, (left, top))
    return canvas


def apply_condition(
    clean: Image.Image,
    condition: dict[str, Any],
    *,
    seed: int,
) -> Image.Image:
    family = condition["perturbation_family"]
    if family == "none":
        return clean.copy()
    if family == "brightness":
        factor = float(condition["factor"])
        if factor <= 0:
            raise ValueError("brightness factor must be positive")
        return ImageEnhance.Brightness(clean).enhance(factor).convert("RGB")
    if family == "gaussian_noise":
        mean = float(condition["mean_fraction"]) * 255.0
        std = float(condition["std_fraction"]) * 255.0
        if std <= 0:
            raise ValueError("Gaussian standard deviation must be positive")
        array = np.asarray(clean.convert("RGB"), dtype=np.float32)
        rng = np.random.default_rng(seed)
        noise = rng.normal(mean, std, size=array.shape)
        perturbed = np.clip(np.rint(array + noise), 0, 255).astype(np.uint8)
        return Image.fromarray(perturbed, mode="RGB")
    raise ValueError(f"unsupported perturbation family {family}")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a mapping in {path}")
    return payload


def load_selected_metadata(
    path: Path,
    selected_ids: set[str],
) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["ImageID"] in selected_ids:
                selected[row["ImageID"]] = row
    missing = selected_ids - set(selected)
    if missing:
        raise ValueError(f"missing Open Images metadata for {sorted(missing)}")
    return selected


def validate_spec(spec: dict[str, Any]) -> None:
    scenarios = spec.get("scenarios", [])
    if len(scenarios) != 20:
        raise ValueError(f"expected 20 multimodal scenarios, found {len(scenarios)}")
    scenario_ids = [row["scenario_id"] for row in scenarios]
    image_ids = [row["image_id"] for row in scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("duplicate multimodal scenario_id")
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("the 20 scenarios must use 20 unique images")
    counts: dict[str, int] = {}
    for row in scenarios:
        counts[row["platform"]] = counts.get(row["platform"], 0) + 1
        for required in (
            "required_scene_points",
            "required_decision_points",
            "uncertainty_points",
            "forbidden_claims",
        ):
            if not row.get(required):
                raise ValueError(f"{row['scenario_id']} has empty {required}")
    if counts != {"Aido_Rover": 10, "Sentinel_Prime_AI": 10}:
        raise ValueError(f"unexpected platform counts: {counts}")
    conditions = spec.get("conditions", [])
    if len(conditions) != 3:
        raise ValueError("expected clean, Gaussian noise, and brightness conditions")
    families = [condition["perturbation_family"] for condition in conditions]
    if families != ["none", "gaussian_noise", "brightness"]:
        raise ValueError("conditions are not in the frozen one-factor order")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-spec", type=Path, required=True)
    parser.add_argument("--source-image-dir", type=Path, required=True)
    parser.add_argument("--open-images-metadata", type=Path, required=True)
    parser.add_argument("--output-image-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--attribution-csv", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = load_yaml(args.scenario_spec)
    validate_spec(spec)
    scenarios = spec["scenarios"]
    source_policy = spec["source_policy"]
    standard = spec["image_standardization"]
    global_seed = int(spec["random_seed"])
    selected_ids = {row["image_id"] for row in scenarios}
    metadata = load_selected_metadata(args.open_images_metadata, selected_ids)
    args.output_image_dir.mkdir(parents=True, exist_ok=True)

    input_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        image_id = scenario["image_id"]
        source = args.source_image_dir / f"{image_id}.jpg"
        if not source.exists():
            raise FileNotFoundError(f"missing selected source image {source}")
        source_metadata = metadata[image_id]
        if source_metadata["License"] != source_policy["allowed_image_license"]:
            raise ValueError(f"{image_id} does not have the frozen image license")
        with Image.open(source) as image:
            clean = standardize_image(
                image,
                width=int(standard["width_px"]),
                height=int(standard["height_px"]),
                padding_rgb=tuple(int(value) for value in standard["padding_rgb"]),
            )
        # Freeze the standardized artifact as lossless PNG.  A JPEG file hash is
        # stable, but decoded pixels can differ by a few values across libjpeg
        # builds; that makes a pixel-level reproducibility check spuriously fail
        # on another host.  PNG preserves the exact RGB array across platforms.
        clean_path = args.output_image_dir / f"{scenario_id}.png"
        clean.save(clean_path, format="PNG", optimize=False)
        # Reopen the exact stored clean artifact; the runner starts from these
        # decoded pixels, not from the pre-save in-memory canvas.
        with Image.open(clean_path) as stored:
            stored_clean = stored.convert("RGB")
        clean_pixel_hash = pixel_sha256(stored_clean)
        relative_image_path = clean_path.relative_to(args.output_jsonl.parent)

        for condition in spec["conditions"]:
            condition_id = condition["condition_id"]
            condition_seed = deterministic_condition_seed(
                global_seed, scenario_id, condition_id
            )
            evaluated = apply_condition(
                stored_clean,
                condition,
                seed=condition_seed,
            )
            request_id = f"w04-vlm::{scenario_id}::{condition_id}"
            input_rows.append(
                {
                    "request_base_id": request_id,
                    "benchmark_id": spec["benchmark_id"],
                    "benchmark_version": str(spec["version"]),
                    "scenario_id": scenario_id,
                    "platform": scenario["platform"],
                    "condition_id": condition_id,
                    "perturbation_family": condition["perturbation_family"],
                    "condition_parameters": {
                        key: value
                        for key, value in condition.items()
                        if key not in {"condition_id", "perturbation_family"}
                    },
                    "condition_seed": condition_seed,
                    "image_path": relative_image_path.as_posix(),
                    "image_file_sha256": sha256_file(clean_path),
                    "clean_pixel_sha256": clean_pixel_hash,
                    "expected_processed_pixel_sha256": pixel_sha256(evaluated),
                    "input_width_px": evaluated.width,
                    "input_height_px": evaluated.height,
                    "user_prompt": spec["prompts"][scenario["platform"]],
                    "source_image_id": image_id,
                    "source_license": source_metadata["License"],
                    "source_landing_url": source_metadata["OriginalLandingURL"],
                    "random_seed": global_seed,
                }
            )

        attribution_rows.append(
            {
                "scenario_id": scenario_id,
                "open_images_image_id": image_id,
                "title": source_metadata["Title"],
                "author": source_metadata["Author"],
                "author_profile_url": source_metadata["AuthorProfileURL"],
                "original_landing_url": source_metadata["OriginalLandingURL"],
                "original_url": source_metadata["OriginalURL"],
                "image_license": source_metadata["License"],
                "open_images_annotation_license": source_policy["annotation_license"],
                "modifications": "rotation normalized; contained on 768x768 gray canvas; JPEG re-encode",
                "stored_filename": clean_path.name,
                "stored_sha256": sha256_file(clean_path),
            }
        )

    write_jsonl(args.output_jsonl, input_rows)
    with args.attribution_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(attribution_rows[0]))
        writer.writeheader()
        writer.writerows(attribution_rows)
    manifest = {
        "manifest_id": "w04_multimodal_input_manifest_v0.1.0",
        "builder_version": BUILDER_VERSION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "random_seed": global_seed,
        "scenario_spec": {
            "path": str(args.scenario_spec),
            "sha256": sha256_file(args.scenario_spec),
        },
        "open_images_metadata": {
            # Keep the public manifest portable and free of workstation paths.
            "path": args.open_images_metadata.name,
            "sha256": sha256_file(args.open_images_metadata),
        },
        "outputs": {
            "input_jsonl": {
                "path": str(args.output_jsonl),
                "sha256": sha256_file(args.output_jsonl),
                "row_count": len(input_rows),
            },
            "attribution_csv": {
                "path": str(args.attribution_csv),
                "sha256": sha256_file(args.attribution_csv),
                "row_count": len(attribution_rows),
            },
            "clean_image_directory": str(args.output_image_dir),
            "clean_image_count": len(attribution_rows),
        },
        "condition_counts": {
            condition["condition_id"]: sum(
                row["condition_id"] == condition["condition_id"]
                for row in input_rows
            )
            for condition in spec["conditions"]
        },
        "status": "frozen_ready_for_vlm_inference",
    }
    args.manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
