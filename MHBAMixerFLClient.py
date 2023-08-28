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
import flwr as fl
from omegaconf import OmegaConf

configs = OmegaConf.load("config.yml")
dataset_conf, model_conf = configs.dataset, configs.model
num_model1, num_model2, num_model3 = model_conf.num_models


class MHBAMixerClient(NumPyClient): # 创建 MHBAMixer 模型的客户端
    def __init__(self, mixers, model_name):
        super(MHBAMixerClient, self).__init__()  # 调用了父类 NumPyClient 的构造函数，确保正确地初始化父类
        self.mixers = mixers
        self.model_name = model_name

    def get_parameters(self, config):
        return _get_parameters(self.mixers)

    def set_parameters(self, parameters: List[np.ndarray]):
        if self.model_name == "MHBAMixer":
            _set_parameters(self.mixers, parameters[:model_conf.num_layers[0]])
        if self.model_name == "DWTMixer":
            _set_parameters(self.mixers, parameters[model_conf.num_layers[0]:model_conf.num_layers[0]+model_conf.num_layers[1]])
        if self.model_name == "TSMixer":
            _set_parameters(self.mixers, parameters[model_conf.num_layers[0]+model_conf.num_layers[1]:])

    def fit(self, parameters, config):

        train_loader = DataLoader(HuggingFaceDatasetImpl(**dataset_conf, mode=dataset_conf.train),
                                  batch_size=dataset_conf["batch_size"], shuffle=True)
        val_loader = DataLoader(HuggingFaceDatasetImpl(**dataset_conf, mode=dataset_conf.validation),
                                batch_size=dataset_conf["batch_size"])
        self.set_parameters(parameters)
        # 每10轮融合一次指标和模型参数
        trainer = lightning.Trainer(max_epochs=1, accelerator="auto", devices="auto")
        trainer.fit(self.mixers, train_loader, val_loader)
        return self.get_parameters(config={}), len(train_loader), {}

    def evaluate(self, parameters, config):
        test_loader = DataLoader(HuggingFaceDatasetImpl(**dataset_conf, mode=dataset_conf.test),
                                 batch_size=dataset_conf["batch_size"])
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
    """
    model_flag, vocab_size, n_heads, max_seq_len: int, hidden_dim: int, index: int, kernel_size: int,
                 dilation: int, padding: int, num_mixers: int, num_classes: int, model_name: str = "MHBAMixer", **kwargs
    """
    mixer_map = {"MHBAMixer": 0, "DWTMixer": 1, "TSMixer": 2}
    model = MHBAMixerModule(model_flag=mixer_map[args.mixer],
                            vocab_size=model_conf.vocab_size,
                            n_heads=model_conf.n_heads,
                            max_seq_len=model_conf.max_seq_len,
                            hidden_dim=model_conf.hidden_dim,
                            index=model_conf.index,
                            kernel_size=model_conf.kernel_size,
                            dilation=model_conf.dilation,
                            padding=model_conf.padding,
                            num_mixers=model_conf.num_mixers,
                            num_classes=model_conf.num_classes,
                            model_name=args.mixer)
    # Flower client
    client = MHBAMixerClient(model, model_name=args.mixer)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

if __name__ == "__main__":
    main()
