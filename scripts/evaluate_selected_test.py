"""Evaluate one validation-selected dual-text ALMT checkpoint on test once."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exactly one checkpoint selected by validation metrics. "
            "This script never searches for the best test epoch."
        )
    )
    parser.add_argument(
        "--config_file",
        default="configs/mosi_dual_c4_intensity.yaml",
        help="Training configuration that produced the checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Validation-selected checkpoint. Defaults to "
            "<ckpt_root>/<project_name>/best_validation_model.pth."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Artifact directory. Defaults to the checkpoint directory.",
    )
    parser.add_argument("--gpu_id", type=int, default=None)
    return parser.parse_args()


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate(model, data_loader, metrics_fn, device):
    model.eval()
    predictions = []
    labels = []
    component_predictions = {
        "regression_predictions": [],
        "ordinal_predictions": [],
    }

    with torch.no_grad():
        for data in data_loader:
            model_output = model(
                data["vision"].to(device),
                data["audio"].to(device),
                data["text"].to(device),
                data["text_llm"].to(device),
                return_aux=True,
            )
            label = data["labels"]["M"].view(-1, 1)
            prediction = (
                model_output["prediction"]
                if isinstance(model_output, dict)
                else model_output
            )
            predictions.append(prediction.detach().cpu())
            labels.append(label.cpu())

            if isinstance(model_output, dict):
                output_key_mapping = {
                    "regression_predictions": "regression_prediction",
                    "ordinal_predictions": "ordinal_prediction",
                }
                for result_key, model_key in output_key_mapping.items():
                    if model_key in model_output:
                        component_predictions[result_key].append(
                            model_output[model_key].detach().cpu()
                        )

    prediction = torch.cat(predictions)
    label = torch.cat(labels)
    result = {
        "predictions": prediction,
        "labels": label,
        "results": metrics_fn(prediction, label),
    }
    for key, values in component_predictions.items():
        if values:
            result[key] = torch.cat(values)
    return result


def save_artifacts(result, dataset, checkpoint, config_file, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch = int(checkpoint.get("epoch", -1))
    predictions = result["predictions"].view(-1).numpy()
    labels = result["labels"].view(-1).numpy()

    payload = {
        "predictions": predictions,
        "labels": labels,
        "epoch": np.asarray(epoch, dtype=np.int64),
    }
    for key in ("regression_predictions", "ordinal_predictions"):
        if key in result:
            payload[key] = result[key].view(-1).numpy()

    prediction_path = output_dir / "selected_test_predictions.npz"
    np.savez_compressed(prediction_path, **payload)

    csv_path = output_dir / "selected_test_predictions.csv"
    raw_text_llm = getattr(dataset, "raw_text_llm", None)
    regression_predictions = payload.get("regression_predictions")
    ordinal_predictions = payload.get("ordinal_predictions")
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "index",
                "id",
                "label",
                "prediction",
                "regression_prediction",
                "ordinal_prediction",
                "raw_text",
                "raw_text_llm",
            ]
        )
        for index, (label, prediction) in enumerate(zip(labels, predictions)):
            writer.writerow(
                [
                    index,
                    dataset.ids[index],
                    float(label),
                    float(prediction),
                    (
                        ""
                        if regression_predictions is None
                        else float(regression_predictions[index])
                    ),
                    (
                        ""
                        if ordinal_predictions is None
                        else float(ordinal_predictions[index])
                    ),
                    dataset.raw_text[index],
                    "" if raw_text_llm is None else raw_text_llm[index],
                ]
            )

    report = {
        "evaluation_protocol": (
            "single_test_evaluation_of_validation_selected_checkpoint"
        ),
        "config_file": str(config_file),
        "selected_epoch": epoch,
        "validation_selection": checkpoint.get("selection"),
        "test_sample_count": len(labels),
        "test_results": result["results"],
    }
    report_path = output_dir / "selected_test_results.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(_json_ready(report), file, ensure_ascii=False, indent=2)

    return prediction_path, csv_path, report_path


def main():
    cli_args = parse_args()
    from core.dataset_dual import DualTextMMDataset
    from core.metric import MetricsTop
    from core.utils import dict_to_namespace, setup_seed
    from models.almt_dual import build_model

    config_file = Path(cli_args.config_file)
    with open(config_file, encoding="utf-8") as file:
        args = dict_to_namespace(yaml.load(file, Loader=yaml.FullLoader))

    gpu_id = args.base.gpu_id if cli_args.gpu_id is None else cli_args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(args.base.seed)

    checkpoint_path = (
        Path(cli_args.checkpoint)
        if cli_args.checkpoint is not None
        else Path(args.base.ckpt_root)
        / args.base.project_name
        / "best_validation_model.pth"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Validation-selected checkpoint not found: {checkpoint_path}"
        )
    output_dir = (
        Path(cli_args.output_dir)
        if cli_args.output_dir is not None
        else checkpoint_path.parent
    )

    print(f"Device: {device} ({gpu_id})")
    print(f"Checkpoint: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    selection = checkpoint.get("selection")
    if selection is None:
        print("Warning: checkpoint has no saved validation-selection metadata.")
    else:
        print(
            "Validation selection: "
            f"epoch={selection['selected_epoch']}, "
            f"{selection['primary_metric']}={selection['primary_value']:.6f}"
        )
        if selection["primary_metric"] != "Mult_acc_7":
            print(
                "Warning: this checkpoint was not selected by validation "
                "Mult_acc_7."
            )

    test_dataset = DualTextMMDataset(args, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.base.batch_size,
        num_workers=args.base.num_workers,
        shuffle=False,
    )
    model = build_model(args).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    metrics_fn = MetricsTop().getMetics(args.dataset.datasetName)
    result = evaluate(model, test_loader, metrics_fn, device)
    prediction_path, csv_path, report_path = save_artifacts(
        result=result,
        dataset=test_dataset,
        checkpoint=checkpoint,
        config_file=config_file,
        output_dir=output_dir,
    )

    print("\n---------------- Selected Checkpoint Test ----------------")
    print("Test was evaluated exactly once; no test epoch was selected.")
    print(f"Test Results: {result['results']}")
    print(f"Predictions: {prediction_path}")
    print(f"Readable predictions: {csv_path}")
    print(f"Report: {report_path}")
    print("----------------------------------------------------------")


if __name__ == "__main__":
    main()
