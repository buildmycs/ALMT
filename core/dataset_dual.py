"""
Dataset loader for the raw-text + LLM-enhanced-text ALMT variant.
"""

import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class DualTextMMDataset(Dataset):
    def __init__(self, args, mode="train"):
        self.mode = mode
        self.args = args.dataset
        self._load_data()

    def _load_data(self):
        with open(self.args.dataPath, "rb") as file:
            data = pickle.load(file)

        split = data[self.mode]
        required_keys = (
            "text_bert",
            "text_bert_llm",
            "vision",
            "audio",
            "raw_text",
            "id",
            self.args.train_mode + "_labels",
        )
        missing_keys = [key for key in required_keys if key not in split]
        if missing_keys:
            raise KeyError(
                f"{self.args.dataPath} split '{self.mode}' is missing keys: "
                f"{missing_keys}. Use scripts/build_dual_text_pkl.py first."
            )

        self.text = split["text_bert"].astype(np.float32)
        self.text_llm = split["text_bert_llm"].astype(np.float32)
        if self.text.shape != self.text_llm.shape:
            raise ValueError(
                f"{self.mode}: text_bert shape {self.text.shape} does not match "
                f"text_bert_llm shape {self.text_llm.shape}"
            )
        if self.text.ndim != 3 or self.text.shape[1] != 3:
            raise ValueError(
                f"{self.mode}: expected text tensors shaped (N, 3, L), "
                f"got {self.text.shape}"
            )

        self.vision = split["vision"].astype(np.float32)
        self.audio = split["audio"].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.raw_text = split["raw_text"]
        self.raw_text_llm = split.get("raw_text_llm")
        self.ids = split["id"]
        self.labels = {
            "M": split[self.args.train_mode + "_labels"].astype(np.float32)
        }

        if self.args.datasetName == "sims":
            for modality in "TAV":
                self.labels[modality] = split[
                    self.args.train_mode + "_labels_" + modality
                ]

        sample_count = len(self.labels["M"])
        named_arrays = {
            "text_bert": self.text,
            "text_bert_llm": self.text_llm,
            "vision": self.vision,
            "audio": self.audio,
            "id": self.ids,
        }
        mismatched = {
            name: len(value)
            for name, value in named_arrays.items()
            if len(value) != sample_count
        }
        if mismatched:
            raise ValueError(
                f"{self.mode}: arrays do not match label count {sample_count}: "
                f"{mismatched}"
            )

        print(f"----------------- {self.args.datasetName} {self.mode} -----------------")
        print(f"Original language shape: {self.text.shape}")
        print(f"Enhanced language shape: {self.text_llm.shape}")
        print(f"Vision shape: {self.vision.shape}")
        print(f"Audio shape: {self.audio.shape}")
        print("-------------------------------------------------------------------------")

    def __len__(self):
        return len(self.labels["M"])

    def __getitem__(self, index):
        sample = {
            "raw_text": self.raw_text[index],
            "text": torch.from_numpy(self.text[index]),
            "text_llm": torch.from_numpy(self.text_llm[index]),
            "audio": torch.from_numpy(self.audio[index]),
            "vision": torch.from_numpy(self.vision[index]),
            "index": index,
            "id": self.ids[index],
            "labels": {
                key: torch.from_numpy(value[index].reshape(-1))
                for key, value in self.labels.items()
            },
        }
        if self.raw_text_llm is not None:
            sample["raw_text_llm"] = self.raw_text_llm[index]
        return sample


def _build_split_generators(seed):
    """Create isolated RNG streams so evaluation cannot perturb train order."""
    base_seed = int(seed)
    split_seeds = {
        "train": base_seed,
        "valid": base_seed + 10_000,
        "test": base_seed + 20_000,
    }
    generators = {}
    for split, split_seed in split_seeds.items():
        generator = torch.Generator()
        generator.manual_seed(split_seed)
        generators[split] = generator
    return generators, split_seeds


def DualTextMMDataLoader(args, seed=None):
    loader_seed = args.base.seed if seed is None else seed
    generators, split_seeds = _build_split_generators(loader_seed)
    print(
        "DataLoader generator seeds: "
        + ", ".join(
            f"{split}={split_seeds[split]}"
            for split in ("train", "valid", "test")
        )
    )
    datasets = {
        split: DualTextMMDataset(args, mode=split)
        for split in ("train", "valid", "test")
    }
    return {
        split: DataLoader(
            dataset,
            batch_size=args.base.batch_size,
            num_workers=args.base.num_workers,
            shuffle=(split == "train"),
            generator=generators[split],
        )
        for split, dataset in datasets.items()
    }
