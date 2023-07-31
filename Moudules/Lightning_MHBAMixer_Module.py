#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/7/20 14:06
# @Author  : lxy15058247683@aliyun.com
# @FileName: Lightning_MHBAMixer_Module.py
# @Copyright: MIT
from typing import Dict, List

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
        self.automatic_optimization = False
        self.mixers = get_mixer_module(vocab_size, n_heads, max_seq_len, hidden_dim, index, kernel_size, dilation, padding,
                                 num_mixers, num_classes, model_name)
        # print(self.mixers)
        self.accuracy = Accuracy("multiclass", num_classes=num_classes)
        self.f1score = F1Score("multiclass", num_classes=num_classes)
        self.recall = Recall("multiclass", num_classes=num_classes)
        self.roc = ROC("multiclass", num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        opt = self.optimizers(use_pl_optimizer=True)
        # self.toggle_optimizer(opt)
        result_tensor = self.share_step(batch, "train", batch_idx)
        self.train_step_outputs.append(result_tensor)
        self.manual_backward(result_tensor[-1])
        self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        opt.step()
        opt.optimizer.zero_grad()
        return result_tensor[-1]

    def share_step(self,batch, stage, batch_ids):
        inputs, target = batch["tokens"]["input_ids"], batch["labels"]
        output = self.mixers(inputs)
        target = target.long()
        # print(output.shape, target.shape)
        accuracy = self.accuracy(output, target)
        f1score = self.f1score(output, target)
        recall = self.recall(output, target)
        roc = self.roc(output, target)
        loss = self.criterion(output, target)
        # print(f"{stage}_loss", output[0], target[0])
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{stage}_accuracy", accuracy, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{stage}_f1", f1score, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{stage}_recall", recall, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{stage}_idx", batch_ids)
        result_tensor = torch.cat((accuracy.unsqueeze(0), f1score.unsqueeze(0), recall.unsqueeze(0), loss.unsqueeze(0)), dim=0)
        return result_tensor

    def validation_step(self, batch, batch_idx):
        result_tensor = self.share_step(batch, "val", batch_idx)
        self.val_step_outputs.append(result_tensor)

    def test_step(self, batch, batch_idx):
        result_tensor = self.share_step(batch, "test", batch_idx)
        self.test_step_outputs.append(result_tensor)

    @staticmethod
    def merge_metrics(metrices: List):
        mean_result = torch.stack(metrices)
        mean_result = mean_result.mean(dim=0)
        return mean_result[0], mean_result[1], mean_result[2], mean_result[3]

    def on_train_epoch_end(self) -> None:
        accuracy, f1score, recall, loss = self.merge_metrics(self.train_step_outputs)
        self.log("train_epoch_accuracy", accuracy)
        self.log("train_epoch_f1score", f1score)
        self.log("train_epoch_recall", recall)
        self.log("train_epoch_loss", loss)
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        accuracy, f1score, recall, loss = self.merge_metrics(self.val_step_outputs)
        self.log("val_epoch_accuracy", accuracy)
        self.log("val_epoch_f1score", f1score)
        self.log("val_epoch_recall", recall)
        self.log("val_epoch_loss", loss)
        self.val_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        accuracy, f1score, recall, loss = self.merge_metrics(self.test_step_outputs)
        self.log("test_epoch_accuracy", accuracy)
        self.log("test_epoch_f1score", f1score)
        self.log("test_epoch_recall", recall)
        self.log("test_epoch_loss", loss)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=5e-5, weight_decay=1e-5)
        return [optimizer], []

