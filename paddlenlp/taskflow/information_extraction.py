# coding:utf-8
# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import re
import os
import json
import imghdr
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import paddle
from ..datasets import load_dataset
from ..transformers import AutoTokenizer, AutoModel
from ..layers import GlobalPointerForEntityExtraction, GPLinkerForRelationExtraction
from ..utils.image_utils import (
    DocParser,
    ResizeImage,
    Permute,
    NormalizeImage,
)
from .models import UIE, UIEM, UIEX
from .task import Task
from .utils import (get_span, get_id_and_prob, get_bool_ids_greater_than,
                    dbc2sbc, gp_decode, map_offset, SchemaTree, DataCollatorGP)

usage = r"""
            from paddlenlp import Taskflow

            # Entity Extraction
            schema = ['时间', '选手', '赛事名称'] # Define the schema for entity extraction
            ie = Taskflow('information_extraction', schema=schema)
            ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")
            '''
            [{'时间': [{'text': '2月8日上午', 'start': 0, 'end': 6, 'probability': 0.9857378532924486}], '选手': [{'text': '谷爱凌', 'start': 28, 'end': 31, 'probability': 0.8981548639781138}], '赛事名称': [{'text': '北京冬奥会自由式滑雪女子大跳台决赛', 'start': 6, 'end': 23, 'probability': 0.8503089953268272}]}]
            '''

            # Relation Extraction
            schema = [{"歌曲名称":["歌手", "所属专辑"]}] # Define the schema for relation extraction
            ie.set_schema(schema) # Reset schema
            ie("《告别了》是孙耀威在专辑爱的故事里面的歌曲")
            '''
            [{'歌曲名称': [{'text': '告别了', 'start': 1, 'end': 4, 'probability': 0.6296155977145546, 'relations': {'歌手': [{'text': '孙耀威', 'start': 6, 'end': 9, 'probability': 0.9988381005599081}], '所属专辑': [{'text': '爱的故事', 'start': 12, 'end': 16, 'probability': 0.9968462078543183}]}}, {'text': '爱的故事', 'start': 12, 'end': 16, 'probability': 0.2816869478191606, 'relations': {'歌手': [{'text': '孙耀威', 'start': 6, 'end': 9, 'probability': 0.9951415104192272}]}}]}]
            '''

            # Event Extraction
            schema = [{'地震触发词': ['地震强度', '时间', '震中位置', '震源深度']}] # Define the schema for event extraction
            ie.set_schema(schema) # Reset schema
            ie('中国地震台网正式测定：5月16日06时08分在云南临沧市凤庆县(北纬24.34度，东经99.98度)发生3.5级地震，震源深度10千米。')
            '''
            [{'地震触发词': [{'text': '地震', 'start': 56, 'end': 58, 'probability': 0.9977425555988333, 'relations': {'地震强度': [{'text': '3.5级', 'start': 52, 'end': 56, 'probability': 0.998080217831891}], '时间': [{'text': '5月16日06时08分', 'start': 11, 'end': 22, 'probability': 0.9853299772936026}], '震中位置': [{'text': '云南临沧市凤庆县(北纬24.34度，东经99.98度)', 'start': 23, 'end': 50, 'probability': 0.7874012889740385}], '震源深度': [{'text': '10千米', 'start': 63, 'end': 67, 'probability': 0.9937974422968665}]}}]}]
            '''

            # Opinion Extraction
            schema = [{'评价维度': ['观点词', '情感倾向[正向，负向]']}] # Define the schema for opinion extraction
            ie.set_schema(schema) # Reset schema
            ie("地址不错，服务一般，设施陈旧")
            '''
            [{'评价维度': [{'text': '地址', 'start': 0, 'end': 2, 'probability': 0.9888139270606509, 'relations': {'观点词': [{'text': '不错', 'start': 2, 'end': 4, 'probability': 0.9927847072459528}], '情感倾向[正向，负向]': [{'text': '正向', 'probability': 0.998228967796706}]}}, {'text': '设施', 'start': 10, 'end': 12, 'probability': 0.9588297379365116, 'relations': {'观点词': [{'text': '陈旧', 'start': 12, 'end': 14, 'probability': 0.9286753967902683}], '情感倾向[正向，负向]': [{'text': '负向', 'probability': 0.9949389795770394}]}}, {'text': '服务', 'start': 5, 'end': 7, 'probability': 0.9592857070501211, 'relations': {'观点词': [{'text': '一般', 'start': 7, 'end': 9, 'probability': 0.9949359182521675}], '情感倾向[正向，负向]': [{'text': '负向', 'probability': 0.9952498258302498}]}}]}]
            '''

            # Sentence-level Sentiment Classification
            schema = ['情感倾向[正向，负向]'] # Define the schema for sentence-level sentiment classification
            ie.set_schema(schema) # Reset schema
            ie('这个产品用起来真的很流畅，我非常喜欢')
            '''
            [{'情感倾向[正向，负向]': [{'text': '正向', 'probability': 0.9990024058203417}]}]
            '''

            # English Model
            schema = [{'Person': ['Company', 'Position']}]
            ie_en = Taskflow('information_extraction', schema=schema, model='uie-base-en')
            ie_en('In 1997, Steve was excited to become the CEO of Apple.')
            '''
            [{'Person': [{'text': 'Steve', 'start': 9, 'end': 14, 'probability': 0.999631971804547, 'relations': {'Company': [{'text': 'Apple', 'start': 48, 'end': 53, 'probability': 0.9960158209451642}], 'Position': [{'text': 'CEO', 'start': 41, 'end': 44, 'probability': 0.8871063806420736}]}}]}]
            '''

            schema = ['Sentiment classification [negative, positive]']
            ie_en.set_schema(schema)
            ie_en('I am sorry but this is the worst film I have ever seen in my life.')
            '''
            [{'Sentiment classification [negative, positive]': [{'text': 'negative', 'probability': 0.9998415771287057}]}]
            '''

            # Multilingual Model
            schema = [{'Person': ['Company', 'Position']}]
            ie_m = Taskflow('information_extraction', schema=schema, model='uie-m-base', schema_lang="en")
            ie_m('In 1997, Steve was excited to become the CEO of Apple.')
            '''
            [{'Person': [{'text': 'Steve', 'start': 9, 'end': 14, 'probability': 0.9998436034905893, 'relations': {'Company': [{'text': 'Apple', 'start': 48, 'end': 53, 'probability': 0.9842775467359672}], 'Position': [{'text': 'CEO', 'start': 41, 'end': 44, 'probability': 0.9628799853543271}]}}]}]
            '''
         """

resize_func = ResizeImage(target_size=224, interp=1)
norm_func = NormalizeImage(is_channel_first=False,
                           mean=[123.675, 116.280, 103.530],
                           std=[58.395, 57.120, 57.375])
permute_func = Permute(to_bgr=False)
doc_parser = DocParser()


class UIETask(Task):
    """
    Universal Information Extraction Task. 
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 
    """

    resource_files_names = {
        "model_state": "model_state.pdparams",
        "model_config": "model_config.json",
        "vocab_file": "vocab.txt",
        "special_tokens_map": "special_tokens_map.json",
        "tokenizer_config": "tokenizer_config.json"
    }
    # vocab.txt/special_tokens_map.json/tokenizer_config.json are common to the default model.
    resource_files_urls = {
        "uie-base": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_v1.0/model_state.pdparams",
                "aeca0ed2ccf003f4e9c6160363327c9b"
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/model_config.json",
                "a36c185bfc17a83b6cfef6f98b29c909"
            ],
            "vocab_file": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
                "1c1c1f4fd93c5bed3b4eebec4de976a8"
            ],
            "special_tokens_map": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec"
            ],
            "tokenizer_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
                "59acb0ce78e79180a2491dfd8382b28c"
            ]
        },
        "uie-medium": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medium_v1.0/model_state.pdparams",
                "15874e4e76d05bc6de64cc69717f172e"
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medium/model_config.json",
                "6f1ee399398d4f218450fbbf5f212b15"
            ],
            "vocab_file": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
                "1c1c1f4fd93c5bed3b4eebec4de976a8"
            ],
            "special_tokens_map": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec"
            ],
            "tokenizer_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
                "59acb0ce78e79180a2491dfd8382b28c"
            ]
        },
        "uie-mini": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_mini_v1.0/model_state.pdparams",
                "f7b493aae84be3c107a6b4ada660ce2e"
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_mini/model_config.json",
                "9229ce0a9d599de4602c97324747682f"
            ],
            "vocab_file": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
                "1c1c1f4fd93c5bed3b4eebec4de976a8"
            ],
            "special_tokens_map": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec"
            ],
            "tokenizer_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
                "59acb0ce78e79180a2491dfd8382b28c"
            ]
        },
        "uie-micro": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_micro_v1.0/model_state.pdparams",
                "80baf49c7f853ab31ac67802104f3f15"
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_micro/model_config.json",
                "07ef444420c3ab474f9270a1027f6da5"
            ],
            "vocab_file": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
                "1c1c1f4fd93c5bed3b4eebec4de976a8"
            ],
            "special_tokens_map": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec"
            ],
            "tokenizer_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
                "59acb0ce78e79180a2491dfd8382b28c"
            ]
        },
        "uie-nano": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_nano_v1.0/model_state.pdparams",
                "ba934463c5cd801f46571f2588543700"
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_nano/model_config.json",
                "e3a9842edf8329ccdd0cf6039cf0a8f8"
            ],
            "vocab_file": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
                "1c1c1f4fd93c5bed3b4eebec4de976a8"
            ],
            "special_tokens_map": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec"
            ],
            "tokenizer_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
                "59acb0ce78e79180a2491dfd8382b28c"
            ]
        },
        # Rename to `uie-medium` and the name of `uie-tiny` will be deprecated in future.
        "uie-tiny": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny_v0.1/model_state.pdparams",
                "15874e4e76d05bc6de64cc69717f172e"
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny/model_config.json",
                "6f1ee399398d4f218450fbbf5f212b15"
            ],
            "vocab_file": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
                "1c1c1f4fd93c5bed3b4eebec4de976a8"
            ],
            "special_tokens_map": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec"
            ],
            "tokenizer_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
                "59acb0ce78e79180a2491dfd8382b28c"
            ]
        },
        "uie-medical-base": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medical_base_v0.1/model_state.pdparams",
                "569b4bc1abf80eedcdad5a6e774d46bf"
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/model_config.json",
                "a36c185bfc17a83b6cfef6f98b29c909"
            ],
            "vocab_file": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
                "1c1c1f4fd93c5bed3b4eebec4de976a8"
            ],
            "special_tokens_map": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec"
            ],
            "tokenizer_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
                "59acb0ce78e79180a2491dfd8382b28c"
            ]
        },
        "uie-base-en": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en_v1.1/model_state.pdparams",
                "2baf0647774d6309e4b2be726ad4283a"
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en/model_config.json",
                "2ca9fe0eea8ff9418725d1a24fcf5c36"
            ],
            "vocab_file": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en/vocab.txt",
                "64800d5d8528ce344256daf115d4965e"
            ],
            "special_tokens_map": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec"
            ],
            "tokenizer_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en/tokenizer_config.json",
                "59acb0ce78e79180a2491dfd8382b28c"
            ]
        },
        "uie-m-base": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_base_v1.0/model_state.pdparams",
                "ed96cb17b4b3283a65ad0846ada7799e"
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_base/model_config.json",
                "05c4b9d050e1402a891b207e36d2e501"
            ],
            "vocab_file": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_base/vocab.txt",
                "e6e1091c984592e72c4460e8eb25045e"
            ],
            "special_tokens_map": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_base/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec"
            ],
            "tokenizer_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_base/tokenizer_config.json",
                "f144bd065ea90cc26eaa91197124bdcc"
            ],
            "sentencepiece_model_file": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_base/sentencepiece.bpe.model",
                "bf25eb5120ad92ef5c7d8596b5dc4046"
            ],
        },
        "uie-m-large": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_large_v1.0/model_state.pdparams",
                "75f3989c515f05f6842e314d3f75ee27"
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_large/model_config.json",
                "22ad69618dc3f4c3fe756e3044c3056e"
            ],
            "vocab_file": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_large/vocab.txt",
                "e6e1091c984592e72c4460e8eb25045e"
            ],
            "special_tokens_map": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_large/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec"
            ],
            "tokenizer_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_large/tokenizer_config.json",
                "f144bd065ea90cc26eaa91197124bdcc"
            ],
            "sentencepiece_model_file": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_large/sentencepiece.bpe.model",
                "bf25eb5120ad92ef5c7d8596b5dc4046"
            ],
        },
        "uie-x-base": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_x_base_v0.1/model_state.pdparams",
                "0e06dd6f9598578b16917c246c77ebc1"
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_x_base/model_config.json",
                "c0588ac92ec6a8468f6dd2c2a896bbc5"
            ],
            "vocab_file": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_x_base/vocab.txt",
                "e6e1091c984592e72c4460e8eb25045e"
            ],
            "special_tokens_map": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_x_base/special_tokens_map.json",
                "ba000b17745bb5b5b40236789318847f"
            ],
            "tokenizer_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_x_base/tokenizer_config.json",
                "09456ba644dac6f9d0b367353a36abe7"
            ],
            "sentencepiece_model_file": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_x_base/sentencepiece.bpe.model",
                "bf25eb5120ad92ef5c7d8596b5dc4046"
            ],
        },
    }

    def __init__(self, task, model, schema, schema_lang="ch", **kwargs):
        super().__init__(task=task, model=model, **kwargs)

        if model in ['uie-m-base', 'uie-m-large', 'uie-x-base']:
            self._multilingual = True
            self.resource_files_names[
                'sentencepiece_model_file'] = "sentencepiece.bpe.model"
        else:
            self._multilingual = False
            if 'sentencepiece_model_file' in self.resource_files_names.keys():
                del self.resource_files_names['sentencepiece_model_file']
        self._is_en = True if model in ['uie-base-en'
                                        ] or schema_lang == 'en' else False

        self._summary_token_num = 3
        self._multimodal = True if model in ['uie-x-base'] else False
        self._ocr_lang = kwargs.get("ocr_lang", "ch")
        if self._multimodal:
            self._construct_ocr_engine(lang=self._ocr_lang)
            self._summary_token_num = 4  # [CLS] prompt [SEP] [SEP] text [SEP] for UIE-X

        self._schema_tree = None
        self.set_schema(schema)
        self._check_task_files()
        self._check_predictor_type()
        self._get_inference_model()
        self._usage = usage

        self._max_seq_len = kwargs.get("max_seq_len", 512)
        self._batch_size = kwargs.get("batch_size", 64)
        self._split_sentence = kwargs.get("split_sentence", False)
        self._position_prob = kwargs.get("position_prob", 0.5)
        self._lazy_load = kwargs.get("lazy_load", False)
        self._num_workers = kwargs.get("num_workers", 0)
        self.use_faster = kwargs.get("use_faster", False)
        self._construct_tokenizer()

    def set_schema(self, schema):
        if isinstance(schema, dict) or isinstance(schema, str):
            schema = [schema]
        self._schema_tree = self._build_tree(schema)

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        if self._multimodal:
            self._input_spec = [
                paddle.static.InputSpec(shape=[None, None],
                                        dtype="int64",
                                        name='input_ids'),
                paddle.static.InputSpec(shape=[None, None],
                                        dtype="int64",
                                        name='token_type_ids'),
                paddle.static.InputSpec(shape=[None, None],
                                        dtype="int64",
                                        name='position_ids'),
                paddle.static.InputSpec(shape=[None, None],
                                        dtype="int64",
                                        name='attention_mask'),
                paddle.static.InputSpec(shape=[None, None, 4],
                                        dtype="int64",
                                        name='bbox'),
                paddle.static.InputSpec(shape=[None, 3, 224, 224],
                                        dtype="int64",
                                        name='image'),
            ]
        elif self._multilingual:
            self._input_spec = [
                paddle.static.InputSpec(shape=[None, None],
                                        dtype="int64",
                                        name='input_ids'),
                paddle.static.InputSpec(shape=[None, None],
                                        dtype="int64",
                                        name='pos_ids'),
            ]
        else:
            self._input_spec = [
                paddle.static.InputSpec(shape=[None, None],
                                        dtype="int64",
                                        name='input_ids'),
                paddle.static.InputSpec(shape=[None, None],
                                        dtype="int64",
                                        name='token_type_ids'),
                paddle.static.InputSpec(shape=[None, None],
                                        dtype="int64",
                                        name='pos_ids'),
                paddle.static.InputSpec(shape=[None, None],
                                        dtype="int64",
                                        name='att_mask'),
            ]

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        if self._multimodal:
            model_instance = UIEX.from_pretrained(self._task_path)
        elif self._multilingual:
            model_instance = UIEM.from_pretrained(self._task_path)
        else:
            model_instance = UIE.from_pretrained(self._task_path)
        self._model = model_instance
        self._model.eval()

    def _construct_tokenizer(self):
        """
        Construct the tokenizer for the predictor.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._task_path, use_faster=self.use_faster)

    def _preprocess(self, inputs):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        if self._multimodal:
            inputs = self._check_input_doc(inputs)
        else:
            inputs = self._check_input_text(inputs)
        outputs = {}
        outputs['text'] = inputs
        return outputs

    def _single_stage_predict(self, inputs):
        input_texts = [d["text"] for d in inputs]
        prompts = [d["prompt"] for d in inputs]

        # max predict length should exclude the length of prompt and summary tokens
        max_predict_len = self._max_seq_len - len(
            max(prompts)) - self._summary_token_num

        if self._multimodal:
            bboxes_list = [d["bboxes"] for d in inputs]
            short_input_texts, short_bboxes_list, input_mapping = self._auto_splitter(
                input_texts,
                max_predict_len,
                bboxes_list=bboxes_list,
                split_sentence=self._split_sentence)
        else:
            short_input_texts, input_mapping = self._auto_splitter(
                input_texts,
                max_predict_len,
                split_sentence=self._split_sentence)

        short_texts_prompts = []
        for k, v in input_mapping.items():
            short_texts_prompts.extend([prompts[k] for _ in range(len(v))])
        if self._multimodal:
            image_list = []
            for k, v in input_mapping.items():
                image_list.extend([inputs[k]["image"] for _ in range(len(v))])
            short_inputs = [{
                "text": short_input_texts[i],
                "prompt": short_texts_prompts[i],
                "bboxes": short_bboxes_list[i],
                "image": image_list[i]
            } for i in range(len(short_input_texts))]
        else:
            short_inputs = [{
                "text": short_input_texts[i],
                "prompt": short_texts_prompts[i]
            } for i in range(len(short_input_texts))]

        def text_reader(inputs):
            for example in inputs:
                encoded_inputs = self._tokenizer(text=[example["prompt"]],
                                                 text_pair=[example["text"]],
                                                 truncation=True,
                                                 max_seq_len=self._max_seq_len,
                                                 pad_to_max_seq_len=True,
                                                 return_attention_mask=True,
                                                 return_position_ids=True,
                                                 return_offsets_mapping=True)
                if self._multilingual:
                    tokenized_output = [
                        encoded_inputs["input_ids"][0],
                        encoded_inputs["position_ids"][0],
                        encoded_inputs["offset_mapping"][0]
                    ]
                else:
                    tokenized_output = [
                        encoded_inputs["input_ids"][0],
                        encoded_inputs["token_type_ids"][0],
                        encoded_inputs["position_ids"][0],
                        encoded_inputs["attention_mask"][0],
                        encoded_inputs["offset_mapping"][0]
                    ]
                tokenized_output = [
                    np.array(x, dtype="int64") for x in tokenized_output
                ]
                yield tuple(tokenized_output)

        def doc_reader(inputs, pad_id=1, c_sep_id=2):

            def _process_bbox(tokens, bbox_lines, offset_mapping, offset_bias):
                bbox_list = [[0, 0, 0, 0] for x in range(len(tokens))]

                for index, bbox in enumerate(bbox_lines):
                    index_token = map_offset(index + offset_bias,
                                             offset_mapping)
                    if 0 <= index_token < len(bbox_list):
                        bbox_list[index_token] = bbox
                    prev_bbox = bbox
                return bbox_list

            def _pad_image_data(image_data):
                if not image_data:
                    image = np.zeros([3, 224, 224])
                    return image
                # decode image
                data = np.frombuffer(bytearray(image_data), dtype="uint8")
                image = np.array(Image.open(BytesIO(data)).convert('RGB'))
                sample = {"image": image}
                # resize image
                sample = resize_func(sample)
                # norm image
                sample = norm_func(sample)
                # permute
                sample = permute_func(sample)
                return sample['image']

            def _encode_doc(tokenizer, offset_mapping, last_offset, prompt,
                            this_text_line, inputs_ids, q_sep_index,
                            max_seq_len):
                if len(offset_mapping) == 0:
                    content_encoded_inputs = tokenizer(
                        text=[prompt],
                        text_pair=[this_text_line],
                        max_seq_len=max_seq_len,
                        return_dict=False,
                        return_offsets_mapping=True)
                    content_encoded_inputs = content_encoded_inputs[0]
                    inputs_ids = content_encoded_inputs["input_ids"][:-1]
                    sub_offset_mapping = [
                        list(x)
                        for x in content_encoded_inputs["offset_mapping"]
                    ]
                    q_sep_index = content_encoded_inputs["input_ids"].index(
                        2, 1)

                    bias = 0
                    for index in range(len(sub_offset_mapping)):
                        if index == 0:
                            continue
                        mapping = sub_offset_mapping[index]
                        if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
                            bias = sub_offset_mapping[index - 1][-1] + 1
                        if mapping[0] == 0 and mapping[1] == 0:
                            continue
                        sub_offset_mapping[index][0] += bias
                        sub_offset_mapping[index][1] += bias

                    offset_mapping = sub_offset_mapping[:-1]
                    last_offset = offset_mapping[-1][-1]
                else:
                    content_encoded_inputs = tokenizer(
                        text=this_text_line,
                        max_seq_len=max_seq_len,
                        return_dict=False,
                        return_offsets_mapping=True)
                    inputs_ids += content_encoded_inputs["input_ids"][1:-1]
                    sub_offset_mapping = [
                        list(x)
                        for x in content_encoded_inputs["offset_mapping"]
                    ]

                    for sub_list in sub_offset_mapping[1:-1]:
                        offset_mapping += [[
                            last_offset, sub_list[1] - sub_list[0] + last_offset
                        ]]
                        last_offset = offset_mapping[-1][-1]
                return offset_mapping, last_offset, q_sep_index, inputs_ids

            for example in inputs:
                content = example["text"]
                prompt = example["prompt"]
                bbox_lines = example.get("bboxes", None)
                image_buff_string = example.get("image", None)

                # Text
                if bbox_lines is None:
                    encoded_inputs = self._tokenizer(
                        text=[example["prompt"]],
                        text_pair=[example["text"]],
                        truncation=True,
                        max_seq_len=self._max_seq_len,
                        pad_to_max_seq_len=True,
                        return_attention_mask=True,
                        return_position_ids=True,
                        return_offsets_mapping=True,
                        return_dict=False)

                    encoded_inputs = encoded_inputs[0]

                    inputs_ids = encoded_inputs["input_ids"]
                    position_ids = encoded_inputs["position_ids"]
                    attention_mask = encoded_inputs["attention_mask"]

                    q_sep_index = inputs_ids.index(2, 1)
                    c_sep_index = attention_mask.index(0)

                    offset_mapping = [
                        list(x) for x in encoded_inputs["offset_mapping"]
                    ]

                    bbox_list = [[0, 0, 0, 0] for x in range(len(inputs_ids))]
                    token_type_ids = [
                        1 if token_index <= q_sep_index
                        or token_index > c_sep_index else 0
                        for token_index in range(self._max_seq_len)
                    ]
                    padded_image = np.zeros([3, 224, 224])
                # Doc
                else:
                    inputs_ids = []
                    prev_bbox = [-1, -1, -1, -1]
                    this_text_line = ""
                    q_sep_index = -1
                    offset_mapping = []
                    last_offset = 0
                    for char_index, (char, bbox) in enumerate(
                            zip(content, bbox_lines)):
                        if char_index == 0:
                            prev_bbox = bbox
                            this_text_line = char
                            continue

                        if all([bbox[x] == prev_bbox[x] for x in range(4)]):
                            this_text_line += char
                        else:
                            offset_mapping, last_offset, q_sep_index, inputs_ids = _encode_doc(
                                self._tokenizer, offset_mapping, last_offset,
                                prompt, this_text_line, inputs_ids, q_sep_index,
                                self._max_seq_len)
                            this_text_line = char
                        prev_bbox = bbox

                    if len(this_text_line) > 0:
                        offset_mapping, last_offset, q_sep_index, inputs_ids = _encode_doc(
                            self._tokenizer, offset_mapping, last_offset,
                            prompt, this_text_line, inputs_ids, q_sep_index,
                            self._max_seq_len)

                    if len(inputs_ids) > self._max_seq_len:
                        inputs_ids = inputs_ids[:(self._max_seq_len -
                                                  1)] + [c_sep_id]
                        offset_mapping = offset_mapping[:(self._max_seq_len -
                                                          1)] + [[0, 0]]
                    else:
                        inputs_ids += [c_sep_id]
                        offset_mapping += [[0, 0]]

                    offset_bias = offset_mapping[q_sep_index - 1][-1] + 1

                    seq_len = len(inputs_ids)
                    inputs_ids += [pad_id] * (self._max_seq_len - seq_len)
                    token_type_ids = [1] * (q_sep_index + 1) + [0] * (
                        seq_len - q_sep_index - 1)
                    token_type_ids += [pad_id] * (self._max_seq_len - seq_len)

                    bbox_list = _process_bbox(inputs_ids, bbox_lines,
                                              offset_mapping, offset_bias)

                    offset_mapping += [[0, 0]] * (self._max_seq_len - seq_len)

                    # Reindex the text
                    text_start_idx = offset_mapping[1:].index(
                        [0, 0]) + self._summary_token_num - 1
                    for idx in range(text_start_idx, self._max_seq_len):
                        offset_mapping[idx][0] -= offset_bias
                        offset_mapping[idx][1] -= offset_bias

                    position_ids = list(range(seq_len))

                    position_ids = position_ids + [0] * (self._max_seq_len -
                                                         seq_len)
                    attention_mask = [1] * seq_len + [0] * (self._max_seq_len -
                                                            seq_len)

                    image_data = base64.b64decode(
                        image_buff_string.encode("utf8"))
                    padded_image = _pad_image_data(image_data)

                input_list = [
                    inputs_ids, token_type_ids, position_ids, attention_mask,
                    bbox_list, padded_image, offset_mapping
                ]

                assert len(inputs_ids) == self._max_seq_len
                assert len(token_type_ids) == self._max_seq_len
                assert len(position_ids) == self._max_seq_len
                assert len(attention_mask) == self._max_seq_len
                assert len(bbox_list) == self._max_seq_len

                yield tuple([np.array(x, dtype="int64") for x in input_list])

        reader = doc_reader if self._multimodal else text_reader
        infer_ds = load_dataset(reader,
                                inputs=short_inputs,
                                lazy=self._lazy_load)
        batch_sampler = paddle.io.BatchSampler(dataset=infer_ds,
                                               batch_size=self._batch_size,
                                               shuffle=False)

        infer_data_loader = paddle.io.DataLoader(dataset=infer_ds,
                                                 batch_sampler=batch_sampler,
                                                 num_workers=self._num_workers,
                                                 return_list=True)

        sentence_ids = []
        probs = []
        for batch in infer_data_loader:
            if self._multimodal:
                input_ids, token_type_ids, pos_ids, att_mask, bbox, image, offset_maps = batch
            elif self._multilingual:
                input_ids, pos_ids, offset_maps = batch
            else:
                input_ids, token_type_ids, pos_ids, att_mask, offset_maps = batch
            if self._predictor_type == "paddle-inference":
                if self._multimodal:
                    self.input_handles[0].copy_from_cpu(input_ids.numpy())
                    self.input_handles[1].copy_from_cpu(token_type_ids.numpy())
                    self.input_handles[2].copy_from_cpu(pos_ids.numpy())
                    self.input_handles[3].copy_from_cpu(att_mask.numpy())
                    self.input_handles[4].copy_from_cpu(bbox.numpy())
                    self.input_handles[5].copy_from_cpu(image.numpy())
                elif self._multilingual:
                    self.input_handles[0].copy_from_cpu(input_ids.numpy())
                    self.input_handles[1].copy_from_cpu(pos_ids.numpy())
                else:
                    self.input_handles[0].copy_from_cpu(input_ids.numpy())
                    self.input_handles[1].copy_from_cpu(token_type_ids.numpy())
                    self.input_handles[2].copy_from_cpu(pos_ids.numpy())
                    self.input_handles[3].copy_from_cpu(att_mask.numpy())
                self.predictor.run()
                start_prob = self.output_handle[0].copy_to_cpu().tolist()
                end_prob = self.output_handle[1].copy_to_cpu().tolist()
            else:
                if self._multimodal:
                    input_dict = {
                        "input_ids": input_ids.numpy(),
                        "token_type_ids": token_type_ids.numpy(),
                        "position_ids": pos_ids.numpy(),
                        "attention_mask": att_mask.numpy(),
                        "bbox": bbox.numpy(),
                        "image": image.numpy()
                    }
                elif self._multilingual:
                    input_dict = {
                        "input_ids": input_ids.numpy(),
                        "pos_ids": pos_ids.numpy(),
                    }
                else:
                    input_dict = {
                        "input_ids": input_ids.numpy(),
                        "token_type_ids": token_type_ids.numpy(),
                        "pos_ids": pos_ids.numpy(),
                        "att_mask": att_mask.numpy()
                    }
                start_prob, end_prob = self.predictor.run(None, input_dict)
                start_prob = start_prob.tolist()
                end_prob = end_prob.tolist()

            start_ids_list = get_bool_ids_greater_than(
                start_prob, limit=self._position_prob, return_prob=True)
            end_ids_list = get_bool_ids_greater_than(end_prob,
                                                     limit=self._position_prob,
                                                     return_prob=True)

            for start_ids, end_ids, offset_map in zip(start_ids_list,
                                                      end_ids_list,
                                                      offset_maps.tolist()):
                span_set = get_span(start_ids, end_ids, with_prob=True)
                sentence_id, prob = get_id_and_prob(span_set, offset_map)
                sentence_ids.append(sentence_id)
                probs.append(prob)
        results = self._convert_ids_to_results(short_inputs, sentence_ids,
                                               probs)
        results = self._auto_joiner(results, short_input_texts, input_mapping)
        return results

    def _auto_joiner(self, short_results, short_inputs, input_mapping):
        concat_results = []
        is_cls_task = False
        for short_result in short_results:
            if short_result == []:
                continue
            elif 'start' not in short_result[0].keys(
            ) and 'end' not in short_result[0].keys():
                is_cls_task = True
                break
            else:
                break
        for k, vs in input_mapping.items():
            if is_cls_task:
                cls_options = {}
                single_results = []
                for v in vs:
                    if len(short_results[v]) == 0:
                        continue
                    if short_results[v][0]['text'] not in cls_options.keys():
                        cls_options[short_results[v][0]['text']] = [
                            1, short_results[v][0]['probability']
                        ]
                    else:
                        cls_options[short_results[v][0]['text']][0] += 1
                        cls_options[short_results[v][0]['text']][
                            1] += short_results[v][0]['probability']
                if len(cls_options) != 0:
                    cls_res, cls_info = max(cls_options.items(),
                                            key=lambda x: x[1])
                    concat_results.append([{
                        'text':
                        cls_res,
                        'probability':
                        cls_info[1] / cls_info[0]
                    }])
                else:
                    concat_results.append([])
            else:
                offset = 0
                single_results = []
                for v in vs:
                    if v == 0:
                        single_results = short_results[v]
                        offset += len(short_inputs[v])
                    else:
                        for i in range(len(short_results[v])):
                            if 'start' not in short_results[v][
                                    i] or 'end' not in short_results[v][i]:
                                continue
                            short_results[v][i]['start'] += offset
                            short_results[v][i]['end'] += offset
                        offset += len(short_inputs[v])
                        single_results.extend(short_results[v])
                concat_results.append(single_results)
        return concat_results

    def _run_model(self, inputs):
        raw_inputs = inputs['text']
        _inputs = self._parse_inputs(raw_inputs)
        results = self._multi_stage_predict(_inputs)
        inputs['result'] = results
        return inputs

    def _parse_inputs(self, inputs):

        def _normalize_bbox(bbox, size):
            return [
                int(1000 * bbox[0] / size[0]),
                int(1000 * bbox[1] / size[1]),
                int(1000 * bbox[2] / size[0]),
                int(1000 * bbox[3] / size[1]),
            ]

        _inputs = []
        for d in inputs:
            if isinstance(d, dict):
                if self._multimodal and 'doc' in d.keys():

                    image_buff = base64.b64decode(d['doc'])
                    image = np.array(Image.open(BytesIO(image_buff)))
                    image_size = (image.shape[1], image.shape[0])

                    ocr_result = self._ocr.ocr(image, cls=True)
                    ocr_result = ocr_result[0] if len(
                        ocr_result) == 1 else ocr_result

                    image, offset_x, offset_y = doc_parser.expand_image_to_a4_size(
                        image, center=True)

                    content = u""
                    bboxes = []
                    for sub_res in ocr_result:
                        word = sub_res[1][0]
                        source_bbox = sub_res[0]
                        bbox = source_bbox[0] + source_bbox[2]
                        bbox = _normalize_bbox([
                            bbox[0] + offset_x, bbox[1] + offset_y,
                            bbox[2] + offset_x, bbox[3] + offset_y
                        ], image_size)
                        bboxes.extend([bbox for word_index in range(len(word))])
                        content += word
                    _inputs.append({
                        "text": content,
                        "bboxes": bboxes,
                        "image": d['doc']
                    })
                else:
                    _inputs.append({
                        "text": d['text'],
                        "bboxes": None,
                        "image": None
                    })
            else:
                _inputs.append({"text": d, "bboxes": None, "image": None})
        return _inputs

    def _multi_stage_predict(self, data):
        """
        Traversal the schema tree and do multi-stage prediction.

        Args:
            data (list): a list of strings

        Returns:
            list: a list of predictions, where the list's length
                equals to the length of `data`
        """
        results = [{} for _ in range(len(data))]
        # Input check to early return
        if len(data) < 1 or self._schema_tree is None:
            return results

        # Copy to stay `self._schema_tree` unchanged
        schema_list = self._schema_tree.children[:]
        while len(schema_list) > 0:
            node = schema_list.pop(0)
            examples = []
            input_map = {}
            cnt = 0
            idx = 0
            if not node.prefix:
                for one_data in data:
                    examples.append({
                        "text": one_data["text"],
                        "bboxes": one_data["bboxes"],
                        "image": one_data["image"],
                        "prompt": dbc2sbc(node.name)
                    })
                    input_map[cnt] = [idx]
                    idx += 1
                    cnt += 1
            else:
                for pre, one_data in zip(node.prefix, data):
                    if len(pre) == 0:
                        input_map[cnt] = []
                    else:
                        for p in pre:
                            if self._is_en:
                                if re.search(r'\[.*?\]$', node.name):
                                    prompt_prefix = node.name[:node.name.find(
                                        "[", 1)].strip()
                                    cls_options = re.search(
                                        r'\[.*?\]$', node.name).group()
                                    # Sentiment classification of xxx [positive, negative]
                                    prompt = prompt_prefix + p + " " + cls_options
                                else:
                                    prompt = node.name + p
                            else:
                                prompt = p + node.name
                            examples.append({
                                "text": one_data["text"],
                                "bboxes": one_data["bboxes"],
                                "image": one_data["image"],
                                "prompt": dbc2sbc(prompt)
                            })
                        input_map[cnt] = [i + idx for i in range(len(pre))]
                        idx += len(pre)
                    cnt += 1
            if len(examples) == 0:
                result_list = []
            else:
                result_list = self._single_stage_predict(examples)

            if not node.parent_relations:
                relations = [[] for i in range(len(data))]
                for k, v in input_map.items():
                    for idx in v:
                        if len(result_list[idx]) == 0:
                            continue
                        if node.name not in results[k].keys():
                            results[k][node.name] = result_list[idx]
                        else:
                            results[k][node.name].extend(result_list[idx])
                    if node.name in results[k].keys():
                        relations[k].extend(results[k][node.name])
            else:
                relations = node.parent_relations
                for k, v in input_map.items():
                    for i in range(len(v)):
                        if len(result_list[v[i]]) == 0:
                            continue
                        if "relations" not in relations[k][i].keys():
                            relations[k][i]["relations"] = {
                                node.name: result_list[v[i]]
                            }
                        elif node.name not in relations[k][i]["relations"].keys(
                        ):
                            relations[k][i]["relations"][
                                node.name] = result_list[v[i]]
                        else:
                            relations[k][i]["relations"][node.name].extend(
                                result_list[v[i]])
                new_relations = [[] for i in range(len(data))]
                for i in range(len(relations)):
                    for j in range(len(relations[i])):
                        if "relations" in relations[i][j].keys(
                        ) and node.name in relations[i][j]["relations"].keys():
                            for k in range(
                                    len(relations[i][j]["relations"][
                                        node.name])):
                                new_relations[i].append(
                                    relations[i][j]["relations"][node.name][k])
                relations = new_relations

            prefix = [[] for _ in range(len(data))]
            for k, v in input_map.items():
                for idx in v:
                    for i in range(len(result_list[idx])):
                        if self._is_en:
                            prefix[k].append(" of " +
                                             result_list[idx][i]["text"])
                        else:
                            prefix[k].append(result_list[idx][i]["text"] + "的")

            for child in node.children:
                child.prefix = prefix
                child.parent_relations = relations
                schema_list.append(child)
        return results

    def _convert_ids_to_results(self, examples, sentence_ids, probs):
        """
        Convert ids to raw text in a single stage.
        """
        results = []
        for example, sentence_id, prob in zip(examples, sentence_ids, probs):
            if len(sentence_id) == 0:
                results.append([])
                continue
            result_list = []
            text = example["text"]
            prompt = example["prompt"]
            for i in range(len(sentence_id)):
                start, end = sentence_id[i]
                if start < 0 and end >= 0:
                    continue
                if end < 0:
                    start += (len(prompt) + 1)
                    end += (len(prompt) + 1)
                    result = {"text": prompt[start:end], "probability": prob[i]}
                    result_list.append(result)
                else:
                    result = {
                        "text": text[start:end],
                        "start": start,
                        "end": end,
                        "probability": prob[i]
                    }
                    result_list.append(result)
            results.append(result_list)
        return results

    @classmethod
    def _build_tree(cls, schema, name='root'):
        """
        Build the schema tree.
        """
        schema_tree = SchemaTree(name)
        for s in schema:
            if isinstance(s, str):
                schema_tree.add_child(SchemaTree(s))
            elif isinstance(s, dict):
                for k, v in s.items():
                    if isinstance(v, str):
                        child = [v]
                    elif isinstance(v, list):
                        child = v
                    else:
                        raise TypeError(
                            "Invalid schema, value for each key:value pairs should be list or string"
                            "but {} received".format(type(v)))
                    schema_tree.add_child(cls._build_tree(child, name=k))
            else:
                raise TypeError(
                    "Invalid schema, element should be string or dict, "
                    "but {} received".format(type(s)))
        return schema_tree

    def _postprocess(self, inputs):
        """
        This function will convert the model output to raw text.
        """
        return inputs['result']


class GPTask(Task):
    """
    Global Pointer for closed-domain information extraction Task. 
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 
    """
    resource_files_names = {
        "model_state": "model_state.pdparams",
        "model_config": "model_config.json",
        "vocab_file": "vocab.txt",
        "special_tokens_map": "special_tokens_map.json",
        "tokenizer_config": "tokenizer_config.json"
    }

    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._schema_tree = None
        self._load_config()
        self._construct_tokenizer()
        self._get_inference_model()

        self._max_seq_len = kwargs.get("max_seq_len", 256)
        self._batch_size = kwargs.get("batch_size", 64)
        self._lazy_load = kwargs.get("lazy_load", False)
        self._num_workers = kwargs.get("num_workers", 0)

    def _load_config(self):
        model_config_file = os.path.join(
            self._task_path, self.resource_files_names["model_config"])
        with open(model_config_file, encoding="utf-8") as f:
            model_config = json.load(f)
        self._label_maps = model_config["label_maps"]
        self._task_type = model_config["task_type"]
        self._encoder = model_config["encoder"]
        schema = model_config["label_maps"]["schema"]
        self._set_schema(schema)

    def _set_schema(self, schema):
        if isinstance(schema, dict) or isinstance(schema, str):
            schema = [schema]
        self._schema_tree = self._build_tree(schema)

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(shape=[None, None],
                                    dtype="int64",
                                    name='input_ids'),
            paddle.static.InputSpec(shape=[None, None],
                                    dtype="int64",
                                    name='att_mask'),
        ]

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        encoder = AutoModel.from_pretrained(self._encoder)
        if self._task_type == "entity_extraction":
            model_instance = GlobalPointerForEntityExtraction(
                encoder, self._label_maps)
        else:
            model_instance = GPLinkerForRelationExtraction(
                encoder, self._label_maps)
        model_path = os.path.join(self._task_path, "model_state.pdparams")
        state_dict = paddle.load(model_path)
        model_instance.set_dict(state_dict)
        self._model = model_instance
        self._model.eval()

    def _construct_tokenizer(self):
        """
        Construct the tokenizer for the predictor.
        """
        # TODO(zhoushunjie): Will set use_faster=True in future.
        self._tokenizer = AutoTokenizer.from_pretrained(self._task_path)

    def _preprocess(self, inputs):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        inputs = self._check_input_text(inputs)

        def read(inputs):
            for x in inputs:
                tokenized_inputs = self._tokenizer(
                    x,
                    max_length=self._max_seq_len,
                    padding=False,
                    truncation=True,
                    return_attention_mask=True,
                    return_offsets_mapping=True,
                    return_token_type_ids=False,
                )
                tokenized_inputs['text'] = x
                yield tokenized_inputs

        infer_ds = load_dataset(read, inputs=inputs, lazy=self._lazy_load)

        data_collator = DataCollatorGP(self._tokenizer,
                                       label_maps=self._label_maps,
                                       task_type=self._task_type)

        batch_sampler = paddle.io.BatchSampler(dataset=infer_ds,
                                               batch_size=self._batch_size,
                                               shuffle=False)

        infer_data_loader = paddle.io.DataLoader(dataset=infer_ds,
                                                 batch_sampler=batch_sampler,
                                                 collate_fn=data_collator,
                                                 num_workers=self._num_workers,
                                                 return_list=True)
        outputs = {}
        outputs['data_loader'] = infer_data_loader
        outputs['input_texts'] = inputs
        return outputs

    def _run_model(self, inputs):
        all_preds = ([], []) if self._task_type in [
            "opinion_extraction", "relation_extraction"
        ] else []
        for batch in inputs['data_loader']:
            input_ids, attention_masks, offset_mappings, texts = batch
            self.input_handles[0].copy_from_cpu(
                input_ids.numpy().astype('int64'))
            self.input_handles[1].copy_from_cpu(
                attention_masks.numpy().astype('int64'))
            self.predictor.run()
            logits = [
                paddle.to_tensor(self.output_handle[i].copy_to_cpu())
                for i in range(len(self.output_handle))
            ]
            batch_outputs = gp_decode(logits, offset_mappings, texts,
                                      self._label_maps, self._task_type)
            if isinstance(batch_outputs, tuple):
                all_preds[0].extend(batch_outputs[0])  # Entity output
                all_preds[1].extend(batch_outputs[1])  # Relation output
            else:
                all_preds.extend(batch_outputs)
        inputs['result'] = all_preds
        return inputs

    @classmethod
    def _build_tree(cls, schema, name='root'):
        """
        Build the schema tree.
        """
        schema_tree = SchemaTree(name)
        for s in schema:
            if isinstance(s, str):
                schema_tree.add_child(SchemaTree(s))
            elif isinstance(s, dict):
                for k, v in s.items():
                    if isinstance(v, str):
                        child = [v]
                    elif isinstance(v, list):
                        child = v
                    else:
                        raise TypeError(
                            "Invalid schema, value for each key:value pairs should be list or string"
                            "but {} received".format(type(v)))
                    schema_tree.add_child(cls._build_tree(child, name=k))
            else:
                raise TypeError(
                    "Invalid schema, element should be string or dict, "
                    "but {} received".format(type(s)))
        return schema_tree

    def _postprocess(self, inputs):
        if self._task_type == "entity_extraction":
            results = self._postprocess_entity_extraction(inputs)
        elif self._task_type == "opinion_extraction":
            results = self._postprocess_opinion_extraction(inputs)
        else:
            results = self._postprocess_relation_extraction(inputs)
        return results

    def _postprocess_opinion_extraction(self, inputs):
        all_ent_preds, all_rel_preds = inputs['result']
        results = []
        for i in range(len(inputs['input_texts'])):
            result = {}
            aspect_maps = {}
            for ent in all_ent_preds[i]:
                ent_res = {
                    'text': ent['text'],
                    'start': ent['start_index'],
                    'end': ent['start_index'] + len(ent['text']),
                    'probability': ent['probability']
                }
                result.setdefault(ent['type'], []).append(ent_res)
                if ent['type'] == "评价维度":
                    for r in result["评价维度"]:
                        if ent['text'] == r['text'] and ent['start_index'] == r[
                                'start']:
                            aspect_maps[(ent['text'], ent['start_index'])] = r
                            break

            for rel in all_rel_preds[i]:
                r = aspect_maps[(rel['aspect'], rel['aspect_start_index'])]
                r['relations'] = {}
                sentiment = {
                    'probability': rel['probability'],
                    'text': rel['sentiment']
                }
                opinion = {
                    'text': rel['opinion'],
                    'start': rel['opinion_start_index'],
                    'end': rel['opinion_start_index'] + len(rel['opinion']),
                    'probability': rel['probability']
                }
                r['relations'].setdefault('情感倾向[正向，负向]', []).append(sentiment)
                r['relations'].setdefault('观点词', []).append(opinion)
            results.append(result)
        return results

    def _postprocess_relation_extraction(self, inputs):
        all_ent_preds, all_rel_preds = inputs['result']
        results = []
        for input_text_idx in range(len(inputs['input_texts'])):
            result = {}
            schema_list = self._schema_tree.children[:]
            while len(schema_list) > 0:
                node = schema_list.pop(0)
                if node.parent_relations is None:
                    prefix = []
                    relations = [[]]
                    cnt = -1
                    for ent in all_ent_preds[input_text_idx]:
                        if node.name == ent['type']:
                            ent_res = {
                                'text': ent['text'],
                                'start': ent['start_index'],
                                'end': ent['start_index'] + len(ent['text']),
                                'probability': ent['probability']
                            }
                            result.setdefault(node.name, []).append(ent_res)
                            cnt += 1
                            result[node.name][cnt]['relations'] = {}
                            relations[0].append(result[node.name][cnt])
                else:
                    relations = [[] for _ in range(len(node.parent_relations))]
                    for i, rs in enumerate(node.parent_relations):
                        for r in rs:
                            cnt = -1
                            for rel in all_rel_preds[input_text_idx]:
                                if r['text'] == rel['subject'] and r['start'] == rel[
                                        'subject_start_index'] and node.name == rel[
                                            'predicate']:
                                    rel_res = {
                                        'text':
                                        rel['object'],
                                        'start':
                                        rel['object_start_index'],
                                        'end':
                                        rel['object_start_index'] +
                                        len(rel['object']),
                                        'probability':
                                        rel['probability']
                                    }
                                    r['relations'].setdefault(
                                        node.name, []).append(rel_res)
                                    cnt += 1
                                    r['relations'][
                                        node.name][cnt]['relations'] = {}
                                    relations[i].append(
                                        r['relations'][node.name][cnt])
                for child in node.children:
                    child.prefix = prefix
                    child.parent_relations = relations
                    schema_list.append(child)
            results.append(result)
        return results

    def _postprocess_entity_extraction(self, inputs):
        all_preds = inputs['result']
        results = []
        for input_text_idx in range(len(inputs['input_texts'])):
            result = {}
            schema_list = self._schema_tree.children[:]
            while len(schema_list) > 0:
                node = schema_list.pop(0)
                for ent in all_preds[input_text_idx]:
                    if node.name == ent['type']:
                        ent_res = {
                            'text': ent['text'],
                            'start': ent['start_index'],
                            'end': ent['start_index'] + len(ent['text']),
                            'probability': ent['probability']
                        }
                        result.setdefault(node.name, []).append(ent_res)
            results.append(result)
        return results
