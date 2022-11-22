# coding=utf-8
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

import os
import math
import random
from tqdm import tqdm

from .log import logger
from .doc_parser import DocParser


class Convertor(object):
    """Convertor to convert data export from annotation platform"""

    def __init__(self,
                 label_studio_file,
                 negative_ratio=5,
                 prompt_prefix="情感倾向",
                 options=["正向", "负向"],
                 separator="##",
                 layout_analysis=False,
                 expand_to_a4_size=True,
                 schema_lang="ch",
                 anno_type="text"):
        """Init Data Convertor"""
        self.negative_ratio = negative_ratio
        self.prompt_prefix = prompt_prefix
        self.options = options
        self.separator = separator
        self.layout_analysis = layout_analysis
        self.expand_to_a4_size = expand_to_a4_size
        self.schema_lang = schema_lang
        self.anno_type = anno_type
        self.label_studio_file = label_studio_file
        self.ignore_list = ["属性值", "object"]

    def process_text_tag(self, line, task_type="ext"):
        items = {}
        items['text'] = line['data']['text']
        if task_type == "ext":
            items['entities'] = []
            items['relations'] = []
            result_list = line['annotations'][0]['result']
            for a in result_list:
                if a['type'] == "labels":
                    items['entities'].append({
                        "id": a['id'],
                        "start_offset": a['value']['start'],
                        "end_offset": a['value']['end'],
                        "label": a['value']['labels'][0]
                    })
                else:
                    items['relations'].append({
                        "id":
                        a['from_id'] + "-" + a['to_id'],
                        "from_id":
                        a['from_id'],
                        "to_id":
                        a['to_id'],
                        "type":
                        a['labels'][0]
                    })
        elif task_type == "cls":
            items['label'] = line['annotations'][0]['result'][0]['value'][
                'choices']
        return items

    def process_image_tag(self, line, task_type="ext"):

        def _io1(box1, box2):
            """calc intersection over box1 area"""
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            if x2 <= x1 or y2 <= y1:
                return 0.0
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            return (x2 - x1) * (y2 - y1) * 1.0 / box1_area

        def _find_segment_in_box(layouts, box, threshold=0.5):
            positions = []
            global_offset = 0
            for seg in layouts:
                sbox = seg[0]
                text_len = len(seg[1])
                if text_len == 0:
                    continue
                char_w = (sbox[2] - sbox[0]) * 1.0 / text_len
                for i in range(len(seg[1])):
                    cbox = [
                        sbox[0] + i * char_w, sbox[1],
                        sbox[0] + (i + 1) * char_w, sbox[3]
                    ]
                    c_covered = _io1(cbox, box)
                    if c_covered >= threshold:
                        positions.append(global_offset)
                    elif cbox[2] == min(cbox[2], box[2]) and cbox[0] == max(cbox[0], box[0]) \
                            and cbox[1] < box[1] and cbox[3] > box[3]:
                        # all covered on x-axis
                        if c_covered > 0.5:
                            positions.append(global_offset)
                    global_offset += 1
            offsets = []
            if not positions:
                return offsets
            spos = positions[0]
            for i in range(1, len(positions)):
                if positions[i] != positions[i - 1] + 1:
                    offsets.append((spos, positions[i - 1] + 1))
                    spos = positions[i]
            offsets.append((spos, positions[-1] + 1))
            return offsets

        items = {}
        img_file = os.path.basename(line['data']['image'])
        p = img_file.find("-")
        img_file = img_file[p + 1:]

        img_path = os.path.join(
            "/".join(self.label_studio_file.split('/')[:-1]), "images",
            img_file)
        if not os.path.exists(img_path):
            logger.warning(
                "Image file %s not exist in %s" %
                (img_file,
                 "/".join(self.label_studio_file.split('/')[:-1]) + "images"))
            return None
        logger.info("Parsing image file %s ..." % (img_file))
        doc_parser = DocParser(layout_analysis=self.layout_analysis)

        parsed_doc = doc_parser.parse({'doc': img_path})
        img_w, img_h = parsed_doc['img_w'], parsed_doc['img_h']

        text = ''
        boxes = []
        for segment in parsed_doc['layout']:
            box = doc_parser._normalize_box(segment[0], [img_w, img_h],
                                            [1000, 1000])
            text += segment[1]
            boxes.extend([box] * len(segment[1]))
        assert len(text) == len(
            boxes), "len of text is not equal to len of bbox"
        items['text'] = text
        items['bbox'] = boxes
        items['image'] = parsed_doc['image']

        if task_type == "ext":
            items['entities'] = []
            items['relations'] = []

            result_list = line['annotations'][0]['result']
            ent_ids = []
            for e in result_list:
                if e['type'] != 'rectanglelabels':
                    continue
                assert img_w == e['original_width'] and img_h == e[
                    'original_height'], "image size not match"
                box = [
                    e['value']['x'] * 0.01 * img_w,
                    e['value']['y'] * 0.01 * img_h,
                    (e['value']['x'] + e['value']['width']) * 0.01 * img_w,
                    (e['value']['y'] + e['value']['height']) * 0.01 * img_h
                ]
                offsets = _find_segment_in_box(parsed_doc['layout'], box)
                if len(offsets) > 0:
                    items['entities'].append({
                        'id':
                        e['id'],
                        'start_offset':
                        offsets[0][0],
                        'end_offset':
                        offsets[0][1],
                        'label':
                        e['value']['rectanglelabels'][0]
                    })
                    ent_ids.append(e['id'])
            for r in result_list:
                if r['type'] != 'relation':
                    continue
                if r['from_id'] in ent_ids and r['to_id'] in ent_ids:
                    items['relations'].append({
                        'id':
                        r['from_id'] + '-' + r['to_id'],
                        'from_id':
                        r['from_id'],
                        'to_id':
                        r['to_id'],
                        'type':
                        r['labels'][0]
                    })
        else:
            items['label'] = line['annotations'][0]['result'][0]['value'][
                'choices']
        return items

    def convert_cls_examples(self, raw_examples):
        """
        Convert labeled data for classification task.
        """
        examples = []
        logger.info(f"Converting annotation data...")
        with tqdm(total=len(raw_examples)) as pbar:
            for line in raw_examples:
                if self.anno_type == "text":
                    items = self.process_text_tag(line, task_type="cls")
                    image, boxes = None, None
                elif self.anno_type == "image":
                    items = self.process_image_tag(line, task_type="cls")
                    if items is None:
                        continue
                    image, boxes = items['image'], items['bbox']
                else:
                    raise ValueError(
                        "The type of annotation should be text or image")
                text, labels = items["text"], items["label"]
                example = self.generate_cls_example(text, labels,
                                                    self.prompt_prefix,
                                                    self.options, image, boxes)
                examples.append(example)
        return examples

    def convert_ext_examples(self, raw_examples, is_train=True):
        """
        Convert labeled data for extraction task.
        """

        def _sep_cls_label(label, separator):
            label_list = label.split(separator)
            if len(label_list) == 1:
                return label_list[0], None
            return label_list[0], label_list[1:]

        texts = []
        # {"content": "", "result_list": [], "prompt": "X"}
        entity_examples = []
        # {"content": "", "result_list": [], "prompt": "X的Y"}
        relation_examples = []
        # {"content": "", "result_list": [], "prompt": "X的情感倾向[正向，负向]"}
        entity_cls_examples = []

        # Entity label set: ["时间", "地点", ... ]
        entity_label_set = []
        # Entity name set: ["2月8日上午", "北京", ... ]
        entity_name_set = []
        # Predicate set: ["歌手", "所属专辑", ... ]
        predicate_set = []

        # List[List[str]]
        # List of entity prompt for each example
        entity_prompt_list = []
        # List of relation prompt for each example
        relation_prompt_list = []
        # Golden subject label for each example
        subject_golden_list = []
        # List of inverse relation for each example
        inverse_relation_list = []
        # List of predicate for each example
        predicate_list = []

        if self.anno_type == "text":
            images, boxes_list = None, None
        else:
            images, boxes_list = [], []

        logger.info(f"Converting annotation data...")
        with tqdm(total=len(raw_examples)) as pbar:
            for line in raw_examples:

                if self.anno_type == "text":
                    items = self.process_text_tag(line, task_type="ext")
                    image, boxes = None, None
                elif self.anno_type == "image":
                    items = self.process_image_tag(line, task_type="ext")
                    if items is None:
                        continue
                    image, boxes = items['image'], items['bbox']
                    images.append(image)
                    boxes_list.append(boxes)
                else:
                    raise ValueError(
                        "The type of annotation should be text or image")

                text, relations, entities = items["text"], items[
                    "relations"], items["entities"]
                texts.append(text)

                entity_example = []
                entity_prompt = []
                entity_example_map = {}
                entity_map = {}  # id to entity name
                for entity in entities:
                    entity_name = text[
                        entity["start_offset"]:entity["end_offset"]]
                    entity_map[entity["id"]] = {
                        "name": entity_name,
                        "start": entity["start_offset"],
                        "end": entity["end_offset"]
                    }
                    if entity["label"] in self.ignore_list:
                        continue

                    entity_label, entity_cls_label = _sep_cls_label(
                        entity["label"], self.separator)

                    # Define the prompt prefix for entity-level classification
                    # xxx + "的" + 情感倾向 -> Chinese
                    # Sentiment classification + " of " + xxx -> English
                    if self.schema_lang == "ch":
                        entity_cls_prompt_prefix = entity_name + "的" + self.prompt_prefix
                    else:
                        entity_cls_prompt_prefix = self.prompt_prefix + " of " + entity_name
                    if entity_cls_label is not None:
                        entity_cls_example = self.generate_cls_example(
                            text, entity_cls_label, entity_cls_prompt_prefix,
                            self.options, image, boxes)

                        entity_cls_examples.append(entity_cls_example)

                    result = {
                        "text": entity_name,
                        "start": entity["start_offset"],
                        "end": entity["end_offset"]
                    }
                    if entity_label not in entity_example_map.keys():
                        entity_example_map[entity_label] = {
                            "content": text,
                            "result_list": [result],
                            "prompt": entity_label
                        }
                        if self.anno_type == "image":
                            entity_example_map[entity_label]['image'] = image
                            entity_example_map[entity_label]['bbox'] = boxes
                    else:
                        entity_example_map[entity_label]["result_list"].append(
                            result)

                    if entity_label not in entity_label_set and entity_label != "观点词":
                        entity_label_set.append(entity_label)
                    if entity_name not in entity_name_set:
                        entity_name_set.append(entity_name)
                    entity_prompt.append(entity_label)

                for v in entity_example_map.values():
                    entity_example.append(v)

                entity_examples.append(entity_example)
                entity_prompt_list.append(entity_prompt)

                subject_golden = []  # Golden entity inputs
                relation_example = []
                relation_prompt = []
                relation_example_map = {}
                inverse_relation = []
                predicates = []
                for relation in relations:
                    predicate = relation["type"]
                    subject_id = relation["from_id"]
                    object_id = relation["to_id"]
                    # The relation prompt is constructed as follows:
                    # subject + "的" + predicate -> Chinese
                    # predicate + " of " + subject -> English
                    if self.schema_lang == "ch":
                        prompt = entity_map[subject_id]["name"] + "的" + predicate
                        inverse_negative = entity_map[object_id][
                            "name"] + "的" + predicate
                    else:
                        prompt = predicate + " of " + entity_map[subject_id][
                            "name"]
                        inverse_negative = predicate + " of " + entity_map[
                            object_id]["name"]

                    if entity_map[subject_id]["name"] not in subject_golden:
                        subject_golden.append(entity_map[subject_id]["name"])
                    result = {
                        "text": entity_map[object_id]["name"],
                        "start": entity_map[object_id]["start"],
                        "end": entity_map[object_id]["end"]
                    }

                    inverse_relation.append(inverse_negative)
                    predicates.append(predicate)

                    if prompt not in relation_example_map.keys():
                        relation_example_map[prompt] = {
                            "content": text,
                            "result_list": [result],
                            "prompt": prompt
                        }
                        if self.anno_type == "image":
                            relation_example_map[prompt]['image'] = image
                            relation_example_map[prompt]['bbox'] = boxes
                    else:
                        relation_example_map[prompt]["result_list"].append(
                            result)

                    if predicate not in predicate_set:
                        predicate_set.append(predicate)
                    relation_prompt.append(prompt)

                for v in relation_example_map.values():
                    relation_example.append(v)

                relation_examples.append(relation_example)
                relation_prompt_list.append(relation_prompt)
                subject_golden_list.append(subject_golden)
                inverse_relation_list.append(inverse_relation)
                predicate_list.append(predicates)
                pbar.update(1)

        logger.info(f"Adding negative samples for first stage prompt...")
        positive_examples, negative_examples = self.add_entity_negative_example(
            entity_examples, texts, entity_prompt_list, entity_label_set,
            images, boxes_list)
        if len(positive_examples) == 0:
            all_entity_examples = []
        else:
            all_entity_examples = positive_examples + negative_examples

        all_relation_examples = []
        if len(predicate_set) != 0:
            logger.info(f"Adding negative samples for second stage prompt...")
            if is_train:

                positive_examples = []
                negative_examples = []
                per_n_ratio = self.negative_ratio // 3

                with tqdm(total=len(texts)) as pbar:
                    for i, text in enumerate(texts):
                        negative_example = []
                        collects = []
                        num_positive = len(relation_examples[i])

                        # 1. inverse_relation_list
                        redundants1 = inverse_relation_list[i]

                        # 2. entity_name_set ^ subject_golden_list[i]
                        redundants2 = []
                        if len(predicate_list[i]) != 0:
                            nonentity_list = list(
                                set(entity_name_set)
                                ^ set(subject_golden_list[i]))
                            nonentity_list.sort()

                            if self.schema_lang == "ch":
                                redundants2 = [
                                    nonentity + "的" +
                                    predicate_list[i][random.randrange(
                                        len(predicate_list[i]))]
                                    for nonentity in nonentity_list
                                ]
                            else:
                                redundants2 = [
                                    predicate_list[i][random.randrange(
                                        len(predicate_list[i]))] + " of " +
                                    nonentity for nonentity in nonentity_list
                                ]

                        # 3. entity_label_set ^ entity_prompt_list[i]
                        redundants3 = []
                        if len(subject_golden_list[i]) != 0:
                            non_ent_label_list = list(
                                set(entity_label_set)
                                ^ set(entity_prompt_list[i]))
                            non_ent_label_list.sort()

                            if self.schema_lang == "ch":
                                redundants3 = [
                                    subject_golden_list[i][random.randrange(
                                        len(subject_golden_list[i]))] + "的" +
                                    non_ent_label
                                    for non_ent_label in non_ent_label_list
                                ]
                            else:
                                redundants3 = [
                                    non_ent_label + " of " +
                                    subject_golden_list[i][random.randrange(
                                        len(subject_golden_list[i]))]
                                    for non_ent_label in non_ent_label_list
                                ]

                        redundants_list = [
                            redundants1, redundants2, redundants3
                        ]

                        for redundants in redundants_list:
                            if self.anno_type == "text":
                                added, rest = self.add_relation_negative_example(
                                    redundants,
                                    texts[i],
                                    num_positive,
                                    per_n_ratio,
                                )
                            else:
                                added, rest = self.add_relation_negative_example(
                                    redundants, texts[i], num_positive,
                                    per_n_ratio, images[i], boxes_list[i])
                            negative_example.extend(added)
                            collects.extend(rest)

                        num_sup = num_positive * self.negative_ratio - len(
                            negative_example)
                        if num_sup > 0 and collects:
                            if num_sup > len(collects):
                                idxs = [k for k in range(len(collects))]
                            else:
                                idxs = random.sample(range(0, len(collects)),
                                                     num_sup)
                            for idx in idxs:
                                negative_example.append(collects[idx])

                        positive_examples.extend(relation_examples[i])
                        negative_examples.extend(negative_example)
                        pbar.update(1)
                all_relation_examples = positive_examples + negative_examples
            else:
                relation_examples = self.add_full_negative_example(
                    relation_examples, texts, relation_prompt_list,
                    predicate_set, subject_golden_list)
                all_relation_examples = [
                    r for relation_example in relation_examples
                    for r in relation_example
                ]
        return all_entity_examples + all_relation_examples + entity_cls_examples

    def generate_cls_example(self,
                             text,
                             labels,
                             prompt_prefix,
                             options,
                             image=None,
                             boxes=None):
        random.shuffle(self.options)
        cls_options = ",".join(self.options)
        prompt = prompt_prefix + "[" + cls_options + "]"

        result_list = []
        example = {
            "content": text,
            "result_list": result_list,
            "prompt": prompt
        }
        if image and boxes:
            example['image'] = image
            example['bbox'] = boxes
        for label in labels:
            start = prompt.rfind(label) - len(prompt) - 1
            end = start + len(label)
            result = {"text": label, "start": start, "end": end}
            example["result_list"].append(result)
        return example

    def add_full_negative_example(self,
                                  examples,
                                  texts,
                                  relation_prompt_list,
                                  predicate_set,
                                  subject_golden_list,
                                  images=None,
                                  boxes_list=None):
        with tqdm(total=len(relation_prompt_list)) as pbar:
            for i, relation_prompt in enumerate(relation_prompt_list):
                negative_sample = []
                for subject in subject_golden_list[i]:
                    for predicate in predicate_set:
                        # The relation prompt is constructed as follows:
                        # subject + "的" + predicate -> Chinese
                        # predicate + " of " + subject -> English
                        if self.schema_lang == "ch":
                            prompt = subject + "的" + predicate
                        else:
                            prompt = predicate + " of " + subject
                        if prompt not in relation_prompt:
                            negative_result = {
                                "content": texts[i],
                                "result_list": [],
                                "prompt": prompt
                            }
                            if images and boxes_list:
                                negative_result['image'] = images[i]
                                negative_result['bbox'] = boxes_list[i]
                            negative_sample.append(negative_result)
                examples[i].extend(negative_sample)
                pbar.update(1)
        return examples

    def add_entity_negative_example(self,
                                    examples,
                                    texts,
                                    prompts,
                                    label_set,
                                    images=None,
                                    boxes_list=None):
        negative_examples = []
        positive_examples = []
        with tqdm(total=len(prompts)) as pbar:
            for i, prompt in enumerate(prompts):
                redundants = list(set(label_set) ^ set(prompt))
                redundants.sort()

                num_positive = len(examples[i])
                if num_positive != 0:
                    actual_ratio = math.ceil(len(redundants) / num_positive)
                else:
                    # Set num_positive to 1 for text without positive example
                    num_positive, actual_ratio = 1, 0

                if actual_ratio <= self.negative_ratio or self.negative_ratio == -1:
                    idxs = [k for k in range(len(redundants))]
                else:
                    idxs = random.sample(range(0, len(redundants)),
                                         self.negative_ratio * num_positive)

                for idx in idxs:
                    negative_result = {
                        "content": texts[i],
                        "result_list": [],
                        "prompt": redundants[idx]
                    }
                    if images and boxes_list:
                        negative_result['image'] = images[i]
                        negative_result['bbox'] = boxes_list[i]
                    negative_examples.append(negative_result)
                positive_examples.extend(examples[i])
                pbar.update(1)
        return positive_examples, negative_examples

    def add_relation_negative_example(self,
                                      redundants,
                                      text,
                                      num_positive,
                                      ratio,
                                      image=None,
                                      boxes=None):
        added_example = []
        rest_example = []

        if num_positive != 0:
            actual_ratio = math.ceil(len(redundants) / num_positive)
        else:
            # Set num_positive to 1 for text without positive example
            num_positive, actual_ratio = 1, 0

        all_idxs = [k for k in range(len(redundants))]
        if actual_ratio <= ratio or ratio == -1:
            idxs = all_idxs
            rest_idxs = []
        else:
            idxs = random.sample(range(0, len(redundants)),
                                 ratio * num_positive)
            rest_idxs = list(set(all_idxs) ^ set(idxs))

        for idx in idxs:
            negative_result = {
                "content": text,
                "result_list": [],
                "prompt": redundants[idx]
            }
            if image and boxes:
                negative_result['image'] = image
                negative_result['bbox'] = boxes
            added_example.append(negative_result)

        for rest_idx in rest_idxs:
            negative_result = {
                "content": text,
                "result_list": [],
                "prompt": redundants[rest_idx]
            }
            if image and boxes:
                negative_result['image'] = image
                negative_result['bbox'] = boxes
            rest_example.append(negative_result)

        return added_example, rest_example