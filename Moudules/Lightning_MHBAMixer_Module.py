#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/7/20 14:06
# @Author  : lxy15058247683@aliyun.com
# @FileName: Lightning_MHBAMixer_Module.py
# @Copyright: MIT
from typing import Dict

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy, F1Score, Recall, ROC

from Moudules.DWT_Mixers import DWT_MIXER
from Moudules.Mixer import MHBAMixer

def get_mixer_module(vocab_size, n_heads, max_seq_len: int, hidden_dim: int, index: int, kernel_size: int,
                 dilation: int, padding: int, num_mixers: int, num_classes: int, model_name: str = "MHBAMixer"):
    if model_name == "MHBAMixer":
        return MHBAMixer(vocab_size, n_heads, max_seq_len, hidden_dim, index, kernel_size,
                                dilation, padding, num_mixers, num_classes)
    if model_name == "DWTMixer":
        return DWT_MIXER(vocab_size, num_mixers, max_seq_len, hidden_dim, hidden_dim, num_classes)

class MHBAMixerModule(LightningModule):
    def __init__(self, vocab_size, n_heads, max_seq_len: int, hidden_dim: int, index: int, kernel_size: int,
                 dilation: int, padding: int, num_mixers: int, num_classes: int, model_name: str = "MHBAMixer"):
        super().__init__()
        """n_heads, max_seq_len: int, hidden_dim: int, index: int, kernel_size: int,
                         dilation: int, padding"""
        self.mixers = get_mixer_module(vocab_size, n_heads, max_seq_len, hidden_dim, index, kernel_size, dilation, padding,
                                 num_mixers, num_classes, model_name)
        # print(self.mixers)
        self.accuracy = Accuracy("multiclass", num_classes=num_classes)
        self.f1score = F1Score("multiclass", num_classes=num_classes)
        self.recall = Recall("multiclass", num_classes=num_classes)
        self.roc = ROC("multiclass", num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.train_step_outputs = {"loss": [], "accuracy": [], "f1score": [], "recall": [], "roc": []}
        self.val_step_outputs = {"loss": [], "accuracy": [], "f1score": [], "recall": [], "roc": []}
        self.test_step_outputs = {"loss": [], "accuracy": [], "f1score": [], "recall": [], "roc": []}
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        accuracy, f1score, recall, roc, loss = self.share_step(batch, "train")
        self.train_step_outputs['loss'].append(loss)
        self.train_step_outputs["accuracy"].append(accuracy)
        self.train_step_outputs["f1score"].append(f1score)
        self.train_step_outputs["recall"].append(recall)
        # self.train_step_outputs['roc'].append(roc)
        return loss

    def share_step(self,batch, stage):
        inputs, target = batch["tokens"]["input_ids"], batch["labels"]
        output = self.mixers(inputs).sum(1)
        target = target.long()
        # print(output.shape, target.shape)
        accuracy = self.accuracy(output, target)
        f1score = self.f1score(output, target)
        recall = self.recall(output, target)
        roc = self.roc(output, target)
        loss = self.criterion(output, target)
        self.log(f"{stage}_loss", loss)
        self.log(f"{stage}_accuracy", accuracy)
        self.log(f"{stage}_f1", f1score)
        self.log(f"{stage}_recall", recall)
        # self.log(f"{stage}_roc", roc)
        return accuracy, f1score, recall, roc, loss

    def validation_step(self, batch, batch_idx):
        accuracy, f1score, recall, roc, loss = self.share_step(batch, "val")
        self.val_step_outputs['loss'].append(loss)
        self.val_step_outputs["accuracy"].append(accuracy)
        self.val_step_outputs["f1score"].append(f1score)
        self.val_step_outputs["recall"].append(recall)
        # self.val_step_outputs['roc'].append(roc)

    def test_step(self, batch, batch_idx):
        accuracy, f1score, recall, roc, loss = self.share_step(batch, "test")
        self.test_step_outputs['loss'].append(loss)
        self.test_step_outputs["accuracy"].append(accuracy)
        self.test_step_outputs["f1score"].append(f1score)
        self.test_step_outputs["recall"].append(recall)
        # self.test_step_outputs['roc'].append(roc)

    @staticmethod
    def merge_metrics(metrics_dict: Dict):
        batch_size = len(metrics_dict["accuracy"])
        accuracy = sum(metrics_dict["accuracy"])/batch_size
        f1score = sum(metrics_dict["f1score"])/batch_size
        recall = sum(metrics_dict['recall'])/batch_size
        roc = sum(metrics_dict['roc'])/batch_size
        loss = sum(metrics_dict['loss'])/batch_size
        return accuracy, f1score, recall, roc, loss

    def on_train_epoch_end(self) -> None:
        accuracy, f1score, recall, roc, loss = self.merge_metrics(self.train_step_outputs)
        self.log("train_epoch_accuracy", accuracy)
        self.log("train_epoch_f1score", f1score)
        self.log("train_epoch_recall", recall)
        # self.log("train_epoch_roc", roc)
        self.log("train_epoch_loss", loss)
        self.train_step_outputs = {"loss": [], "accuracy": [], "f1score": [], "recall": [], "roc": []}

    def on_validation_epoch_end(self) -> None:
        accuracy, f1score, recall, roc, loss = self.merge_metrics(self.val_step_outputs)
        self.log("val_epoch_accuracy", accuracy)
        self.log("val_epoch_f1score", f1score)
        self.log("val_epoch_recall", recall)
        # self.log("val_epoch_roc", roc)
        self.log("val_epoch_loss", loss)
        self.val_step_outputs = {"loss": [], "accuracy": [], "f1score": [], "recall": [], "roc": []}

    def on_test_epoch_end(self) -> None:
        accuracy, f1score, recall, roc, loss = self.merge_metrics(self.test_step_outputs)
        self.log("test_epoch_accuracy", accuracy)
        self.log("test_epoch_f1score", f1score)
        self.log("test_epoch_recall", recall)
        # self.log("test_epoch_roc", roc)
        self.log("test_epoch_loss", loss)
        self.test_step_outputs = {"loss": [], "accuracy": [], "f1score": [], "recall": [], "roc": []}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=5e-3)
        return [optimizer], []

