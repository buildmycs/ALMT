"""
Training entry point for the dual-text ALMT variant.
"""

import argparse
import csv
import os

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter

from core.dataset_dual import DualTextMMDataLoader
from core.intensity_objective import build_sentiment_objective
from core.metric import MetricsTop
from core.scheduler import get_scheduler
from core.utils import AverageMeter, dict_to_namespace, results_recorder, setup_seed
from models.almt_dual import build_model


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_file",
    type=str,
    default="configs/mosi_dual_c4.yaml",
)
parser.add_argument("--seed", type=int, default=-1)
parser.add_argument("--gpu_id", type=int, default=-1)
opt = parser.parse_args()
print(opt)

with open(opt.config_file, encoding="utf-8") as file:
    args = yaml.load(file, Loader=yaml.FullLoader)
args = dict_to_namespace(args)
print(args)

seed = args.base.seed if opt.seed == -1 else opt.seed
gpu_id = args.base.gpu_id if opt.gpu_id == -1 else opt.gpu_id

print("-----------------args-----------------")
print(args)
print("-------------------------------------")

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({gpu_id})")


def _update_fusion_meters(model, meters):
    for name, value in model.get_dual_text_stats().items():
        meters.setdefault(name, AverageMeter()).update(value, 1)


def _average_fusion_stats(meters):
    return {name: meter.value_avg for name, meter in meters.items()}


def _save_best_artifacts(
    model,
    optimizer,
    epoch,
    validation_ret,
    test_ret,
    test_dataset,
    save_path,
):
    checkpoint_path = os.path.join(save_path, "best_validation_model.pth")
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "validation_results": validation_ret["results"],
            "test_results": test_ret["results"],
        },
        checkpoint_path,
    )

    predictions = test_ret["predictions"].view(-1).numpy()
    labels = test_ret["labels"].view(-1).numpy()
    prediction_path = os.path.join(save_path, "best_test_predictions.npz")
    prediction_payload = {
        "predictions": predictions,
        "labels": labels,
        "epoch": np.asarray(epoch, dtype=np.int64),
    }
    for key in ("regression_predictions", "ordinal_predictions"):
        if key in test_ret:
            prediction_payload[key] = test_ret[key].view(-1).numpy()
    np.savez_compressed(prediction_path, **prediction_payload)

    csv_path = os.path.join(save_path, "best_test_predictions.csv")
    raw_text_llm = getattr(test_dataset, "raw_text_llm", None)
    regression_predictions = prediction_payload.get("regression_predictions")
    ordinal_predictions = prediction_payload.get("ordinal_predictions")
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
            enhanced_text = "" if raw_text_llm is None else raw_text_llm[index]
            regression_prediction = (
                ""
                if regression_predictions is None
                else float(regression_predictions[index])
            )
            ordinal_prediction = (
                ""
                if ordinal_predictions is None
                else float(ordinal_predictions[index])
            )
            writer.writerow(
                [
                    index,
                    test_dataset.ids[index],
                    float(label),
                    float(prediction),
                    regression_prediction,
                    ordinal_prediction,
                    test_dataset.raw_text[index],
                    enhanced_text,
                ]
            )

    print(f"Saved best checkpoint: {checkpoint_path}")
    print(f"Saved test predictions: {prediction_path}")
    print(f"Saved readable predictions: {csv_path}")


def run_epoch(model, data_loader, loss_fn, metrics_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)

    loss_recorder = AverageMeter()
    loss_meters = {}
    fusion_meters = {}
    predictions, labels = [], []
    component_predictions = {
        "regression_predictions": [],
        "ordinal_predictions": [],
    }

    for data in data_loader:
        visual = data["vision"].to(device)
        audio = data["audio"].to(device)
        text = data["text"].to(device)
        text_llm = data["text_llm"].to(device)
        label = data["labels"]["M"].to(device).view(-1, 1)

        if training:
            optimizer.zero_grad()
            model_output = model(
                visual,
                audio,
                text,
                text_llm,
                return_aux=loss_fn.requires_auxiliary_outputs,
            )
            loss, loss_values = loss_fn(model_output, label)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model_output = model(
                    visual,
                    audio,
                    text,
                    text_llm,
                    return_aux=loss_fn.requires_auxiliary_outputs,
                )
                loss, loss_values = loss_fn(model_output, label)

        output = (
            model_output["prediction"]
            if isinstance(model_output, dict)
            else model_output
        )

        batch_size = visual.size(0)
        loss_recorder.update(loss.item(), batch_size)
        for name, value in loss_values.items():
            loss_meters.setdefault(name, AverageMeter()).update(value, batch_size)
        _update_fusion_meters(model, fusion_meters)
        predictions.append(output.detach().cpu())
        labels.append(label.detach().cpu())
        if isinstance(model_output, dict):
            component_predictions["regression_predictions"].append(
                model_output["regression_prediction"].detach().cpu()
            )
            component_predictions["ordinal_predictions"].append(
                model_output["ordinal_prediction"].detach().cpu()
            )

    prediction = torch.cat(predictions)
    label = torch.cat(labels)
    epoch_result = {
        "results": metrics_fn(prediction, label),
        "loss_recorder": loss_recorder,
        "loss_components": _average_fusion_stats(loss_meters),
        "fusion_stats": _average_fusion_stats(fusion_meters),
        "predictions": prediction,
        "labels": label,
    }
    for name, values in component_predictions.items():
        if values:
            epoch_result[name] = torch.cat(values)
    return epoch_result


def main():
    setup_seed(seed)
    log_path = os.path.join(".", "log", args.base.project_name)
    save_path = os.path.join(args.base.ckpt_root, args.base.project_name)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    model = build_model(args).to(device)
    data_loader = DualTextMMDataLoader(args)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.base.lr,
        weight_decay=args.base.weight_decay,
    )
    scheduler_warmup = get_scheduler(optimizer, args)
    loss_fn = build_sentiment_objective(
        args, data_loader["train"].dataset.labels["M"]
    ).to(device)
    print("-----------------objective-----------------")
    print(loss_fn.describe())
    print("-------------------------------------------")
    metrics_fn = MetricsTop().getMetics(args.dataset.datasetName)

    training_recorder = results_recorder()
    validation_recorder = results_recorder()
    test_recorder = results_recorder()
    writer = SummaryWriter(logdir=log_path)
    best_validation_mae = float("inf")

    for epoch in range(1, args.base.n_epochs + 1):
        loss_fn.set_epoch(epoch)
        training_ret = run_epoch(
            model, data_loader["train"], loss_fn, metrics_fn, optimizer
        )
        validation_ret = run_epoch(
            model, data_loader["valid"], loss_fn, metrics_fn
        )
        test_ret = run_epoch(
            model, data_loader["test"], loss_fn, metrics_fn
        )

        training_recorder.update(training_ret["results"], epoch)
        validation_recorder.update(validation_ret["results"], epoch)
        test_recorder.update(test_ret["results"], epoch)
        best_validation = validation_recorder.get_best_results()
        best_test = test_recorder.get_best_results()

        validation_mae = torch.mean(
            torch.abs(validation_ret["predictions"] - validation_ret["labels"])
        ).item()
        if validation_mae < best_validation_mae:
            best_validation_mae = validation_mae
            _save_best_artifacts(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_ret=validation_ret,
                test_ret=test_ret,
                test_dataset=data_loader["test"].dataset,
                save_path=save_path,
            )

        print(f"\n----------------- Results Epoch {epoch} -----------------")
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]}')
        print(f'Training Results: {training_ret["results"]}')
        print(f'Validation Results: {validation_ret["results"]}')
        print(f'Test Results: {test_ret["results"]}')
        print(f'Training Objective: {training_ret["loss_components"]}')
        if training_ret["fusion_stats"]:
            print(f'Dual-text Fusion: {training_ret["fusion_stats"]}')
        print(
            "Best Validation Results across All Epochs: "
            f'{best_validation["best_results_all_epochs"]}'
        )
        print(
            "Best Validation Results of One Epoch: "
            f'{best_validation["best_results_one_epoch"]}'
        )
        print(
            "Best Test Results across All Epochs: "
            f'{best_test["best_results_all_epochs"]}'
        )
        print(
            "Best Test Results of One Epoch: "
            f'{best_test["best_results_one_epoch"]}'
        )
        print("----------------------------------------------------------\n")

        writer.add_scalar(
            "train/loss", training_ret["loss_recorder"].value_avg, epoch
        )
        writer.add_scalar(
            "valid/loss", validation_ret["loss_recorder"].value_avg, epoch
        )
        writer.add_scalar(
            "test/loss", test_ret["loss_recorder"].value_avg, epoch
        )
        for name, value in training_ret["fusion_stats"].items():
            writer.add_scalar(f"dual_text/{name}", value, epoch)
        for name, value in training_ret["loss_components"].items():
            writer.add_scalar(f"objective/{name}", value, epoch)

        ordinal_thresholds = model.get_ordinal_thresholds()
        if ordinal_thresholds is not None:
            for index, value in enumerate(ordinal_thresholds.tolist()):
                writer.add_scalar(f"ordinal/threshold_{index}", value, epoch)

        scheduler_warmup.step()

    writer.close()


if __name__ == "__main__":
    main()
