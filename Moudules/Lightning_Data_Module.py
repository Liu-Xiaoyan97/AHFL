#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/7/20 14:52
# @Author  : lxy15058247683@aliyun.com
# @FileName: Lightning_Data_Module.py
# @Copyright: MIT
from abc import ABC

import numpy as np
from lightning import LightningDataModule
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class NLPDataModule(LightningDataModule):
    def __init__(self, filename: str, label_map: list, subset: str = None,
                 feature1: str = "Text", feature2: str = None, label: str = "label",
                 filepath: str = None, batch_size: int = 128, num_workers: int = 4,
                 max_length: int = 512,):
        super(NLPDataModule).__init__()
        self.filename = filename
        self.subset = subset
        self.feature1 = feature1
        self.feature2 = feature2
        self.label = label
        self.filepath = filepath
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.label_map = label_map
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.train_set = None
        self.eval_set = None
        self.test_set = None

    def required_dataset(self, split: str = None):
        if self.subset_name:
            return load_dataset(self.dataset_name, self.subset_name, split=split)
        else:
            return load_dataset(self.dataset_name, split=split)

    def setup(self, stage: str) -> None:

        if stage == "fit" or stage == None:
            self.train_set = HuggingFaceDatasetImpl(filename=self.filename, subset=self.subset, mode="train",
                                          feature1=self.feature1, feature2=self.feature2, filepath=self.filepath,
                                          label=self.label, max_length=self.max_length, label_map=self.label_map)
            self.val_set = HuggingFaceDatasetImpl(filename=self.filename, subset=self.subset, mode="validation",
                                          feature1=self.feature1, feature2=self.feature2, filepath=self.filepath,
                                          label=self.label, max_length=self.max_length, label_map=self.label_map)

        if stage == "test" or stage == None:
            self.test_set = HuggingFaceDatasetImpl(filename=self.filename, subset=self.subset, mode="validation",
                                          feature1=self.feature1, feature2=self.feature2, filepath=self.filepath,
                                          label=self.label, max_length=self.max_length, label_map=self.label_map)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.eval_set, self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True)


class NLPModules(Dataset):
    def __init__(self, filepath: str = None, max_length: int = 512, *args, **kwargs):
        super().__init__()
        self.filepath = filepath
        self.max_length = max_length
        self.data = None
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __len__(self):
        return len(self.data)

    @staticmethod
    def normalization(text: str):
        return text.replace("!", "") \
            .replace("@", "") \
            .replace("#", "") \
            .replace("$", "") \
            .replace("%", "") \
            .replace("/", "") \
            .replace("\\", "")

    def row_projection(self, field):
        raise NotImplementedError

    def compute_label(self, field):
        raise NotImplementedError

    def __getitem__(self, item):
        row = self.data[item]
        tokens = self.row_projection(row)
        labels = self.compute_label(row)
        return {
            "tokens": {
                "input_ids": np.array(tokens["input_ids"]),
                "token_type_ids": np.array(tokens["token_type_ids"]),
                "attention_mask": np.array(tokens["attention_mask"])
            },
            "labels": labels
        }


class HuggingFaceDatasetImpl(NLPModules, ABC):

    def __init__(self, filename: str, mode: str, label_map: list, subset: str = None,
                 feature1: str = "Text", feature2: str = None, label: str = "label",
                 max_length: int = 512,  *args, **kwargs):
        super(HuggingFaceDatasetImpl, self).__init__(*args, **kwargs)
        if subset is None:
            self.data = load_dataset(filename, split=mode)
        else:
            self.data = load_dataset(filename, subset, split=mode)
        self.max_length = max_length
        self.features1 = feature1
        self.features2 = feature2
        self.label = label
        self.label_map = {key: value for value, key in enumerate(label_map)}

    def merge_pair_text(self, field):
        if self.features2 is None:
            '''
            tokenization_utils_base.py 定义如下类型，使用对应类型即可
            TextInput = str
            PreTokenizedInput = List[str]
            EncodedInput = List[int]
            TextInputPair = Tuple[str, str]
            PreTokenizedInputPair = Tuple[List[str], List[str]]
            EncodedInputPair = Tuple[List[int], List[int]]
            '''
            return field[self.features1], None
        else:
            return field[self.features1], field[self.features2]

    def row_projection(self, field):
        textInput, textInputPair = self.merge_pair_text(field)
        '''
        HuggingFace推荐使用tokenizer.__call__方法
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] 
        待分词句子1
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] 
        句子对待分词句子2 (可选)
        padding: Union[bool, str, PaddingStrategy] str可选为"max_length"或"longest"， 如果是'max_length' 需追加参数max_length
        如果是False，则不使用填充，默认为"longest"，即填充到最长句子相同长度
        truncation: Union[bool, str, TruncationStrategy] str可选为"only_first"或"longest_first'，如果是'only_first'
        就是只在第一个句子完成截断，如果是"longest_first"则在两个句子中分别截断直到达到要求，优先截断最长句子。默认为'only_first', 可选False
        表示截断
        '''
        return self.tokenizer.__call__(text=textInput, text_pair=textInputPair, padding='max_length',
                                       max_length=self.max_length, truncation='longest_first')

    def compute_label(self, field):
        return np.array(self.label_map[field[self.label]], dtype=int)


# if __name__ == "__main__":
#     cola_config = {
#         "filename": "glue",
#         "subset": "cola",
#         "filepath": None,
#         "label_map": [0, 1],
#         "batch_size": 10,
#         "num_workers": 4,
#         "feature1": "sentence",
#         "feature2": None,
#         "label": "label"
#     }
#     test_loader = HuggingFaceDatasetImpl(**cola_config, mode='validation')
#     print(test_loader.__getitem__(0))
