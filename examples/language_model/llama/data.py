# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import io
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import paddle

from paddlenlp.transformers.tokenizer_utils_base import PretrainedTokenizerBase

IGNORE_INDEX = -100


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.
    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def reader(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        json_lines = jload(f)
        for json_line in json_lines:
            yield json_line


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def convert_example(example, tokenizer, data_args):
    """
    Convert an example into necessary features.
    """
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    # NOTE: Almost the same functionality as HuggingFace's prepare_train_features function. The main difference is
    # that HugggingFace uses ArrowTable as basic data structure, while we use list of dictionary instead.

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

    if example.get("input", "") != "":
        input_seq = prompt_input.format_map(example)
    else:
        input_seq = prompt_no_input.format_map(example)

    output_seq = example["output"]

    source_tokenized = tokenizer(
        input_seq,
        return_tensors="pd",
        max_length=data_args.src_length,
        truncation=True,
    )

    source_input_ids_len = (
        source_tokenized["input_ids"].not_equal(paddle.to_tensor(tokenizer.pad_token_id)).sum().item()
    )

    example_tokenized = tokenizer(
        input_seq + output_seq,
        return_tensors="pd",
        max_length=data_args.src_length + data_args.tgt_length,
        truncation=True,
    )

    input_ids = example_tokenized["input_ids"][0]
    labels = copy.deepcopy(input_ids)
    labels[:source_input_ids_len] = IGNORE_INDEX

    return dict(
        input_ids=input_ids,
        labels=labels,
    )


def left_padding(inputs, pad_id):
    max_length = 0
    for ids in inputs:
        max_length = max(max_length, len(ids))

    def extend_max_lenth(value, max_length, to_pad_id):
        return [to_pad_id] * (max_length - len(value)) + value

    def extend_filed(values, max_length, to_pad_id):
        res = []
        for value in values:
            res.append(extend_max_lenth(value.tolist(), max_length, to_pad_id))
        return res

    res = extend_filed(inputs, max_length, pad_id)
    return paddle.to_tensor(res)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PretrainedTokenizerBase

    def __call__(self, features: List[Dict]) -> Dict[str, paddle.Tensor]:

        input_ids, labels = tuple([feature[key] for feature in features] for key in ("input_ids", "labels"))
        input_ids = left_padding(input_ids, pad_id=self.tokenizer.pad_token_id)
        labels = left_padding(labels, pad_id=IGNORE_INDEX)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.not_equal(paddle.to_tensor(self.tokenizer.pad_token_id)),
        )
