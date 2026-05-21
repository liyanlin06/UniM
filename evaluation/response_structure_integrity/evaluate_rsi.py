#!/usr/bin/env python3
"""Evaluate response structure integrity with StS and LeS scores."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


ALLOWED_MODALITIES = ("image", "document", "audio", "video", "code", "threeD")
MODALITY_PATTERN = re.compile(r"<(" + "|".join(ALLOWED_MODALITIES) + r")\d*>")


def parse_modalities(content: str) -> Counter:
    """Extract allowed modality tags and count them by modality type."""
    if not isinstance(content, str):
        return Counter()
    matches = MODALITY_PATTERN.findall(content)
    return Counter(matches)


def calculate_sts(golden_counts: Counter, response_counts: Counter) -> float:
    """Calculate the strict structure score as the mean per-modality F1."""
    modal_union = set(golden_counts.keys()) | set(response_counts.keys())
    if not modal_union:
        return 1.0

    f1_scores = []
    for modality in modal_union:
        golden_num = golden_counts.get(modality, 0)
        response_num = response_counts.get(modality, 0)
        matched_num = min(golden_num, response_num)

        precision = matched_num / response_num if response_num > 0 else 0.0
        recall = matched_num / golden_num if golden_num > 0 else 0.0

        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = (2 * precision * recall) / (precision + recall)

        f1_scores.append(f1_score)

    return sum(f1_scores) / len(f1_scores)


def calculate_les(golden_counts: Counter, response_counts: Counter) -> float:
    """Calculate the lenient structure score using modality-type overlap."""
    golden_types = set(golden_counts.keys())
    response_types = set(response_counts.keys())

    if not golden_types:
        return 1.0

    overlap_types = golden_types.intersection(response_types)
    return len(overlap_types) / len(golden_types)


def load_multi_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load concatenated multi-line JSON objects or JSON arrays."""
    file_text = Path(file_path).read_text(encoding="utf-8").strip()
    if not file_text:
        return []

    if file_text.startswith("["):
        data = json.loads(file_text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {file_path}, but got {type(data).__name__}.")
        return data

    objects: List[Dict[str, Any]] = []
    buffer = ""
    in_string = False
    escape = False
    brace_count = 0
    object_index = 0

    for line_num, line in enumerate(file_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        buffer += stripped

        for char in stripped:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = not in_string
            elif not in_string:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1

        if brace_count == 0 and not in_string and buffer:
            try:
                obj = json.loads(buffer)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON object #{object_index + 1} ending near line {line_num}: {exc}"
                ) from exc

            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected each JSON object in {file_path} to be a dict, but got {type(obj).__name__}."
                )

            objects.append(obj)
            object_index += 1
            buffer = ""

    return objects


def iter_golden_records(golden_records: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    """Yield only golden records that have an id."""
    for item in golden_records:
        if "id" not in item:
            print(f"[WARN] Skipping a golden record without 'id': {item}")
            continue
        yield item


def build_missing_output_record(golden_item: Dict[str, Any]) -> Dict[str, Any]:
    """Create a placeholder record when a golden item has no matching model output."""
    record = dict(golden_item)
    record["output"] = {
        "modal": {},
        "content": "",
    }
    record["missing_model_output"] = True
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate response structure integrity with StS and LeS scores."
    )
    parser.add_argument(
        "--golden-path",
        "-g",
        required=True,
        help="Path to the golden reference file.",
    )
    parser.add_argument(
        "--model-output-path",
        "-m",
        required=True,
        help="Path to the model output file.",
    )
    parser.add_argument(
        "--eval-results-path",
        "-e",
        required=True,
        help="Path to the output results file.",
    )
    args = parser.parse_args()

    print("[INFO] Loading data...")
    try:
        golden_records = load_multi_json_file(args.golden_path)
        model_output_records = load_multi_json_file(args.model_output_path)
    except (FileNotFoundError, ValueError) as exc:
        print("[ERROR] Failed to load input files.")
        print(exc)
        raise SystemExit(1) from exc

    golden_with_id = list(iter_golden_records(golden_records))
    golden_by_id = {item["id"]: item for item in golden_with_id}
    model_output_by_id = {item["id"]: item for item in model_output_records if "id" in item}

    print(f"[INFO] Loaded {len(golden_with_id)} golden records with valid ids.")
    print(f"[INFO] Loaded {len(model_output_by_id)} model output records with valid ids.")

    output_path = Path(args.eval_results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evaluated_count = 0
    missing_count = 0

    with output_path.open("w", encoding="utf-8") as results_file:
        for golden_item in golden_with_id:
            item_id = golden_item["id"]
            model_output_item = model_output_by_id.get(item_id)

            if model_output_item is None:
                model_output_item = build_missing_output_record(golden_item)
                missing_count += 1

            golden_content = golden_item.get("output", {}).get("content", "")
            response_content = model_output_item.get("output", {}).get("content", "")

            golden_counts = parse_modalities(golden_content)
            response_counts = parse_modalities(response_content)

            sts_score = calculate_sts(golden_counts, response_counts)
            les_score = calculate_les(golden_counts, response_counts)

            output_record = dict(model_output_item)
            output_record["eval_score"] = {
                "StS": round(sts_score, 4),
                "LeS": round(les_score, 4),
            }

            results_file.write(json.dumps(output_record, ensure_ascii=False))
            results_file.write("\n")
            evaluated_count += 1

    unmatched_model_ids = set(model_output_by_id.keys()) - set(golden_by_id.keys())
    if unmatched_model_ids:
        print(
            "[WARN] "
            f"{len(unmatched_model_ids)} model output record(s) have no matching golden id. "
            f"Examples (up to 10): {list(unmatched_model_ids)[:10]}"
        )

    print("[INFO] Evaluation complete.")
    print(f"[INFO] Evaluated records: {evaluated_count}")
    print(f"[INFO] Missing model outputs assigned zero-structure scores via empty content: {missing_count}")
    print(f"[INFO] Results saved to: {output_path}")


if __name__ == "__main__":
    main()
