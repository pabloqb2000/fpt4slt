# coding: utf-8
"""
Data module
"""
from torchtext import data
from torchtext.data import Field, RawField
from typing import List, Tuple
import pickle
import gzip
import json
import torch
from torch import Tensor
import numpy as np
from tqdm import tqdm

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)
    
class ReducedDataset(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))
    
    def __init__(self, old_dataset, idxs):
        super().__init__([old_dataset.examples[i] for i in idxs], old_dataset.fields)

class ProbTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        cfg,
        path: str,
        fields: Tuple[RawField, RawField, Field, Field, Field],
        keep_only=None,
        **kwargs
    ):
        print(f"loading {path}")
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("signer", fields[1]),
                ("sgn", fields[2]),
                ("gls", fields[3]),
                ("txt", fields[4]),
            ]

        if not isinstance(path, list):
            path = [path]

        examples = []
        for annotation_file in path:
            annotations = load_json(annotation_file)
            glosses = annotations["glosses"]
            texts = annotations["translations"]
            logits = np.load(cfg.get("data_path", "") + annotations["sequence_logits"].split("/")[-1])
            for i in range(len(glosses)):
                examples.append(
                    data.Example.fromlist(
                        [
                            f"sign_prob_{i}",
                            "unk",
                            Tensor(logits[i,:,:]).float(),
                            glosses[i].strip(),
                            texts[i].strip(),
                        ],
                        fields,
                    )
                )
                # logits = np.delete(logits, 0, axis=0)
        super().__init__(examples, fields, **kwargs)


class SignTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[RawField, RawField, Field, Field, Field],
        keep_only=None,
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            keep_only: If not `None`, only the sample with this name will be kept.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("signer", fields[1]),
                ("sgn", fields[2]),
                ("gls", fields[3]),
                ("txt", fields[4]),
            ]

        if not isinstance(path, list):
            path = [path]

        samples = {}
        for annotation_file in path:
            tmp = load_dataset_file(annotation_file)
            for s in tmp:
                seq_id = s["name"]
                if keep_only is not None:
                    if seq_id not in keep_only:
                        continue
                if seq_id in samples:
                    assert samples[seq_id]["name"] == s["name"]
                    assert samples[seq_id]["signer"] == s["signer"]
                    assert samples[seq_id]["gloss"] == s["gloss"]
                    assert samples[seq_id]["text"] == s["text"]
                    samples[seq_id]["sign"] = torch.cat(
                        [samples[seq_id]["sign"].float(), s["sign"].float()], axis=1
                    )
                else:
                    samples[seq_id] = {
                        "name": s["name"],
                        "signer": s["signer"],
                        "gloss": s["gloss"],
                        "text": s["text"],
                        "sign": s["sign"],
                    }

        examples = []
        for s in samples:
            sample = samples[s]
            examples.append(
                data.Example.fromlist(
                    [
                        sample["name"],
                        sample["signer"],
                        # This is for numerical stability
                        (sample["sign"] + 1e-8).float(),
                        sample["gloss"].strip(),
                        sample["text"].strip(),
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)
