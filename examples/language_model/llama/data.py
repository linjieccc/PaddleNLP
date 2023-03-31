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
from dataclasses import dataclass
from typing import Dict, List

import paddle

from paddlenlp.data import Pad
from paddlenlp.transformers.tokenizer_utils_base import PretrainedTokenizerBase

IGNORE_INDEX = -100


def convert_example(example, tokenizer, data_args):
    """
    Convert an example into necessary features.
    """
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    # NOTE: Almost the same functionality as HuggingFace's prepare_train_features function. The main difference is
    # that HugggingFace uses ArrowTable as basic data structure, while we use list of dictionary instead.
    context = example["context"]
    question = example["question"]
    try:
        answer = example["answers"][0]
    except:
        print(example["context"])
        print(example["question"])
        print(example["answers"])
        print(example["answer_starts"])
        print(example["is_impossible"])

    input_seq = f"answer: {answer} context: {context} </s>"
    output_seq = f"question: {question} </s>"

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


def pad_sequence(inputs, pad_index=0):
    sequences = [inp.numpy() for inp in inputs]
    outputs = Pad(pad_val=pad_index)(sequences)
    output_tensor = paddle.to_tensor(outputs)
    return output_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PretrainedTokenizerBase

    def __call__(self, features: List[Dict]) -> Dict[str, paddle.Tensor]:

        input_ids, labels = tuple([feature[key] for feature in features] for key in ("input_ids", "labels"))
        input_ids = pad_sequence(input_ids, pad_index=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, pad_index=IGNORE_INDEX)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.not_equal(paddle.to_tensor(self.tokenizer.pad_token_id)),
        )
