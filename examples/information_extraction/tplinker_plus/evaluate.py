# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import json
import os
from tqdm import tqdm

import paddle
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoTokenizer, AutoModel
from paddlenlp.utils.log import logger

from utils import postprocess, create_dataloader, reader, get_label_dict, DedupList
from metric import get_eval
from model import TPLinkerPlus


@paddle.no_grad()
def evaluate(model, dataloader, label_dict, task_type="relation_extraction"):
    model.eval()
    all_preds = ([], []) if task_type in [
        "opinion_extraction", "relation_extraction"
    ] else []
    for batch in tqdm(dataloader, desc="Evaluating: ", leave=False):
        input_ids, attention_masks, offset_mappings, texts = batch
        logits = model(input_ids, attention_masks)
        batch_outputs = postprocess(logits, offset_mappings, texts,
                                    input_ids.shape[1], label_dict, task_type)
        if isinstance(batch_outputs, tuple):
            all_preds[0].extend(batch_outputs[0])  # Entity output
            all_preds[1].extend(batch_outputs[1])  # Relation output
        else:
            all_preds.extend(batch_outputs)
    eval_results = get_eval(all_preds, dataloader.dataset.raw_data, task_type)
    model.train()
    return eval_results


def do_eval():
    label_dict = get_label_dict(args.task_type, args.label_dict_path)
    num_tags = len(label_dict["id2tag"])

    tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
    encoder = AutoModel.from_pretrained("ernie-3.0-base-zh")
    model = TPLinkerPlus(encoder, num_tags, shaking_type="cln")
    state_dict = paddle.load(
        os.path.join(args.model_path, "model_state.pdparams"))
    model.set_dict(state_dict)

    test_ds = load_dataset(reader, data_path=args.test_path, lazy=False)

    test_dataloader = create_dataloader(test_ds,
                                        tokenizer,
                                        max_seq_len=args.max_seq_len,
                                        batch_size=args.batch_size,
                                        label_dict=label_dict,
                                        mode="test",
                                        task_type=args.task_type)

    eval_result = evaluate(model,
                           test_dataloader,
                           label_dict,
                           task_type=args.task_type)
    logger.info("Evaluation precision: " + str(eval_result))


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default=None, help="The path of saved model that you want to load.")
    parser.add_argument("--test_path", type=str, default=None, help="The path of test set.")
    parser.add_argument("--label_dict_path", default="./ner_data/label_dict.json", type=str, help="The file path of the labels dictionary.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_len", type=int, default=128, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--task_type", choices=['relation_extraction', 'event_extraction', 'entity_extraction', 'opinion_extraction'], default="relation_extraction", type=str, help="Select the training task type.")

    args = parser.parse_args()
    # yapf: enable

    do_eval()
