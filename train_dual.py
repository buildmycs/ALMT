"""
Training entry point for the dual-text ALMT variant.
"""

import argparse
import os

import torch
import yaml
from tensorboardX import SummaryWriter

from core.dataset_dual import DualTextMMDataLoader
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


def run_epoch(model, data_loader, loss_fn, metrics_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)

    loss_recorder = AverageMeter()
    fusion_meters = {}
    predictions, labels = [], []

    for data in data_loader:
        visual = data["vision"].to(device)
        audio = data["audio"].to(device)
        text = data["text"].to(device)
        text_llm = data["text_llm"].to(device)
        label = data["labels"]["M"].to(device).view(-1, 1)

        if training:
            optimizer.zero_grad()
            output = model(visual, audio, text, text_llm)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                output = model(visual, audio, text, text_llm)
                loss = loss_fn(output, label)

        batch_size = visual.size(0)
        loss_recorder.update(loss.item(), batch_size)
        _update_fusion_meters(model, fusion_meters)
        predictions.append(output.detach().cpu())
        labels.append(label.detach().cpu())

    prediction = torch.cat(predictions)
    label = torch.cat(labels)
    return {
        "results": metrics_fn(prediction, label),
        "loss_recorder": loss_recorder,
        "fusion_stats": _average_fusion_stats(fusion_meters),
    }


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
    loss_fn = torch.nn.MSELoss()
    metrics_fn = MetricsTop().getMetics(args.dataset.datasetName)

    training_recorder = results_recorder()
    validation_recorder = results_recorder()
    test_recorder = results_recorder()
    writer = SummaryWriter(logdir=log_path)

    for epoch in range(1, args.base.n_epochs + 1):
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

        print(f"\n----------------- Results Epoch {epoch} -----------------")
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]}')
        print(f'Training Results: {training_ret["results"]}')
        print(f'Validation Results: {validation_ret["results"]}')
        print(f'Test Results: {test_ret["results"]}')
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

        scheduler_warmup.step()

    writer.close()


if __name__ == "__main__":
    main()
