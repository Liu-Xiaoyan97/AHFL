# 作者 ：D_wh
# 时间 ：2023/7/17 15:49
# 格式化 ：Ctrl+Alt+L
# 清除不用代码 ：Ctrl+Alt+O
# 智能导入 ：Alt+Enter
import argparse
from typing import List, OrderedDict

import lightning
import numpy as np
import torch
from flwr.client import NumPyClient
from torch.utils.data import DataLoader, TensorDataset

from Moudules.Lightning_Data_Module import NLPDataModule, HuggingFaceDatasetImpl
from Moudules.Lightning_MHBAMixer_Module import MHBAMixerModule
from Moudules.Mixer import MixerLayer
from torch import nn
import flwr as fl


class MHBAMixerClient(NumPyClient): # 创建 MHBAMixer 模型的客户端
    def __init__(self, mixers, model_name):
        super(MHBAMixerClient, self).__init__()  # 调用了父类 NumPyClient 的构造函数，确保正确地初始化父类
        self.mixers = mixers
        self.model_name = model_name

    def get_parameters(self, config):
        return _get_parameters(self.mixers)

    def set_parameters(self, parameters: List[np.ndarray]):
        if self.model_name == "MHBAMixer":
            _set_parameters(self.mixers, parameters[:39])
        if self.model_name == "DWTMixer":
            _set_parameters(self.mixers, parameters[39:])

    def fit(self, parameters, config):
        cola_config = {
            "filename": "glue",
            "subset": "cola",
            "filepath": None,
            "label_map": [0, 1],
            "batch_size": 10,
            "num_workers": 4,
            "feature1": "sentence",
            "feature2": None,
            "label": "label",
            "max_length": 128
        }
        train_loader = DataLoader(HuggingFaceDatasetImpl(**cola_config, mode="train"), batch_size=64, shuffle=True)
        val_loader = DataLoader(HuggingFaceDatasetImpl(**cola_config, mode='validation'),batch_size=64)
        self.set_parameters(parameters)
        # 每10轮融合一次指标和模型参数
        trainer = lightning.Trainer(max_epochs=10, accelerator="auto", devices="auto")
        trainer.fit(self.mixers, train_loader, val_loader)
        return self.get_parameters(config={}), len(train_loader), {}

    def evaluate(self, parameters, config):
        cola_config = {
            "filename": "glue",
            "subset": "cola",
            "filepath": None,
            "label_map": [0, 1],
            "batch_size": 10,
            "num_workers": 4,
            "feature1": "sentence",
            "feature2": None,
            "label": "label",
            "max_length": 128
        }
        test_loader = DataLoader(HuggingFaceDatasetImpl(**cola_config, mode='validation'), batch_size=64)
        self.set_parameters(parameters)

        trainer = lightning.Trainer(accelerator="auto", devices="auto", log_every_n_steps=1)
        results = trainer.test(self.mixers, test_loader)
        test_epoch_accuracy = results[0]['test_epoch_accuracy']
        test_epoch_f1score = results[0]['test_epoch_f1score']
        test_epoch_recall = results[0]['test_epoch_recall']
        # test_epoch_roc = results[0]['test_epoch_roc']
        test_epoch_loss = results[0]['test_epoch_loss']
        return test_epoch_loss, len(test_loader), {"accuracy": test_epoch_accuracy,
                                                        "f1score": test_epoch_f1score,
                                                        "recall": test_epoch_recall,
                                                        # "roc": test_epoch_roc,
                                                        "loss": test_epoch_loss}


def _get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def _set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {}
    for k, v in params_dict:
        state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict, strict=True)


def main() -> None:
    # Model and data
    parser = argparse.ArgumentParser()
    parser.add_argument('--mixer', type=str, default="MHBAMixer")
    args = parser.parse_args()
    model_config = {
        "vocab_size": 30522,
        "index": 7,
        "hidden_dim": 64,
        "kernel_size": [ 5, 3, 3, 3, 3, 3, 3, 7 ],
        "dilation": [ 1, 1, 1, 1, 1, 1, 1, 1 ],
        "padding": [ 2, 1, 1, 1, 1, 1, 1, 3 ],
        "n_heads": 2,
        "num_mixers": 2,
        "max_seq_len": 128,
        "num_classes": 2,
        "model_name": args.mixer
    }
    """
    n_heads, max_seq_len: int, hidden_dim: int, index: int, kernel_size: int,
                 dilation: int, padding: int, num_mixers: int, num_classes: int
    """
    model = MHBAMixerModule(**model_config)
    # Flower client
    client = MHBAMixerClient(model, model_name=args.mixer)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

if __name__ == "__main__":
    main()
