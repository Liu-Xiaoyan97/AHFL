# 作者 ：D_wh
# 时间 ：2023/7/16 16:43
# 格式化 ：Ctrl+Alt+L
# 清除不用代码 ：Ctrl+Alt+O
# 智能导入 ：Alt+Enter
from math import sqrt
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader

from Moudules.Lightning_Data_Module import HuggingFaceDatasetImpl


class Bernolli_sampling_nlp:
    def __init__(self, max_len=100, prob=1):
        self.max_len = max_len
        self.prob = prob

    def __call__(self, inputs):
        n_samples, max_seq_len, embedding_dim = inputs.size(0), inputs.size(1), inputs.size(-1)
        Benoulli = torch.distributions.bernoulli.Bernoulli(self.prob)
        masks = Benoulli.sample((n_samples, max_seq_len)).unsqueeze(-1).repeat(1, 1, embedding_dim).bool().cuda()
        inputs = F.softmax(inputs.masked_fill(~masks, -np.inf), dim=-2)
        return inputs


class Bernolli_sampling_cv:
    def __init__(self, max_len=100, prob=1):
        self.max_len = max_len
        self.prob = prob

    def __call__(self, inputs):
        n_samples, max_seq_len, embedding_dim = inputs.size(0), inputs.size(1), inputs.size(-1)
        Benoulli = torch.distributions.bernoulli.Bernoulli(self.prob)
        masks = Benoulli.sample((n_samples, max_seq_len, embedding_dim)).cuda().bool()
        inputs = F.softmax(inputs.masked_fill(~masks, -np.inf), dim=-2)
        return inputs


class HBA(nn.Module):
    def __init__(self, mode, max_seq_len, embedding_dim, prob, kernel_size, dilation, padding):
        super(HBA, self).__init__()
        self.embedding_dim = embedding_dim
        self.local_information = nn.Conv1d(embedding_dim, embedding_dim,
                                                          kernel_size, 1, padding, dilation, groups=embedding_dim)
        self.activate = nn.GELU()
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.global_information = nn.Linear(max_seq_len, max_seq_len)
        self.bernolli_sampling = self.Choice_Bernolli(mode)(max_len=max_seq_len, prob=prob)
        self.softmax = nn.Softmax(-1)

    def Choice_Bernolli(self, mode: str):
        if mode == "cv":
            return Bernolli_sampling_cv
        else:
            return Bernolli_sampling_nlp

    def forward(self, x):
        x = x.transpose(1, 2)
        # [N, embedding_dim 4, max_seq_len 384]
        q = self.bn(self.activate(self.local_information(x)+x))
        k = self.activate(self.bernolli_sampling(x))
        v = self.activate(self.global_information(x))
        attention = self.softmax(torch.bmm(q, k.transpose(1, 2))/sqrt(self.embedding_dim))
        output = torch.bmm(attention, v)
        return output.transpose(1, 2), attention


class MHBA(nn.Module):
    """
            :param n_head: 头数
            :param mode: nlp or cv
            :param max_seq_len: 最大序列长度
            :param embedding_dim: 嵌入维度
            :param prob: 采样概率 一般为0.8，消融实验做过了 0.8 最好
            :param kernel_size: 卷积核大小
            :param dilation: 空洞率 0表示普通卷积，以k=3,d=1的卷积为例，近似等于k=5,d=0的卷积
            :param padding: 填充大小，用于处理边界
            .. math:: output_feature_map = lower(\\frac{(l+2p-k)}{s})
            """
    def __init__(self, n_head, mode, max_seq_len, embedding_dim, prob, kernel_size, dilation, padding):
        super(MHBA, self).__init__()
        assert max_seq_len % n_head == 0, 'max_seq_len must be divisible by the n_head.'
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.max_seq_len = max_seq_len
        self.input_dim = int(max_seq_len // n_head)
        self.local_information = nn.Conv1d(embedding_dim, embedding_dim,
                                                          kernel_size, 1, padding, groups=self.input_dim)
        self.activate = nn.GELU()
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.global_information = nn.Linear(self.input_dim, self.input_dim)
        self.mode = mode
        if mode == "cv":
            self.bernolli_sampling = Bernolli_sampling_cv(prob=prob)
        else:
            self.bernolli_sampling = Bernolli_sampling_nlp(prob=prob)
        self.softmax = nn.Softmax(-1)
        self.trans = Rearrange("b (m h) d -> (b h) d m ", h=n_head)
        self.trans2 = Rearrange("(b h) d m  -> b (m h) d ", h=n_head)

    def forward(self, inputs):
        #  b (chw) (p1 p2)
        # print(inputs.shape)
        if self.mode == "cv":
            q = self.trans(inputs)
            k = self.trans(inputs)
            v = self.trans(inputs)
        else:
            q = inputs.view(-1, self.embedding_dim, self.input_dim)
            k = inputs.view(-1, self.embedding_dim, self.input_dim)
            v = inputs.view(-1, self.embedding_dim, self.input_dim)
        # print(q.shape)
        q = self.bn(self.activate(self.local_information(q)+q))
        k = self.activate(self.bernolli_sampling(k))
        v = self.activate(self.global_information(v))
        # print(q.shape, k.shape, v.shape)
        attention = self.softmax(torch.bmm(q, k.transpose(1, 2)) / sqrt(self.embedding_dim))
        output = torch.bmm(attention, v)
        if self.mode == "cv":
            return self.trans2(output), attention
        else:
            return output.reshape(-1, self.max_seq_len, self.embedding_dim), attention


class MHBAMixer(nn.Module):
    """n_heads, max_seq_len: int, hidden_dim: int, index: int, kernel_size: int,
                     dilation: int, padding"""
    def __init__(self, vocab_size, n_heads, max_seq_len, hidden_dim, index, kernel_size, dilation, padding, num_mixers,
                 num_classes, **kwargs):
        super(MHBAMixer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim)
        """n_heads, max_seq_len: int, hidden_dim: int, index: int, kernel_size: int,
                         dilation: int, padding"""
        self.mixers = nn.Sequential(
            *[MixerLayer(n_heads, max_seq_len, hidden_dim, index, kernel_size[i], dilation[i], padding[i], **kwargs)
                                  for i in range(num_mixers)]
        )
        self.classification = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # features = torch.tensor(inputs["tokens"]["input_ids"]).long()
        outputs = self.embedding(inputs)
        outputs = self.mixers(outputs)
        means = outputs.mean(dim=-1, keepdim=True).mean(1)
        output = self.classification(outputs.mean(1)-means)
        return output


class MixerLayer(nn.Module):
    def __init__(self, n_heads, max_seq_len: int, hidden_dim: int, index: int, kernel_size: int,
                 dilation: int, padding: int, **kwargs):
        super(MixerLayer, self).__init__(**kwargs)
        self.kernel_size, self.dilation, self.padding = kernel_size, dilation, padding
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        # attention = attention_choice(index)
        self.sa = MHBA(n_heads, 'nlp', max_seq_len, hidden_dim, 0.8, kernel_size, dilation, padding)
        # self.sa = TCA("nlp", max_seq_len, hidden_dim, 0.8, kernel_size, dilation, padding)
        self.activate = nn.GELU()
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.mlp_2 = MlpLayer(hidden_dim, hidden_dim)

    def forward(self, inputs) -> torch.Tensor:
        residual = inputs
        outputs = self.layer_norm_1(inputs)
        # print(outputs.shape, residual.shape)
        outputs, attention = self.sa(outputs)
        outputs = self.activate(outputs + residual)
        residual = outputs
        outputs = self.layer_norm_2(outputs)
        outputs = self.activate(self.mlp_2(self.dropout(outputs)) + residual)
        return outputs


class MlpLayer(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, **kwargs):
        super(MlpLayer, self).__init__(**kwargs)
        self.layers = nn.Sequential(*[
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim)
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


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
#         "label": "label",
#         "max_length": 128
#     }
#     test_loader = HuggingFaceDatasetImpl(**cola_config, mode='validation')
#     loader = DataLoader(test_loader, batch_size=2)
#     print(len(loader))
#     # print(test_loader.__getitem__(0))
#     model_config = {
#         "vocab_size": 30522,
#         "index": 7,
#         "hidden_dim": 64,
#         "kernel_size": [5, 3, 3, 3, 3, 3, 3, 7],
#         "dilation": [1, 1, 1, 1, 1, 1, 1, 1],
#         "padding": [2, 1, 1, 1, 1, 1, 1, 3],
#         "n_heads": 2,
#         "num_mixers": 2,
#         "max_seq_len": 128,
#         "num_classes": 2
#     }
#     """
#     n_heads, max_seq_len: int, hidden_dim: int, index: int, kernel_size: int,
#                  dilation: int, padding: int, num_mixers: int, num_classes: int
#     """
#     model = MHBAMixer(**model_config)
#     test_batch = loader.__iter__().__next__()
#     output = model(test_batch)
#     print(output)