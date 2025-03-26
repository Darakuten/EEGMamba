import sys
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
from time import time
import torch.nn.functional as F
from termcolor import cprint
from einops import rearrange
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


class Classifier(nn.Module):
    # NOTE: experimental

    def __init__(self, args):
        super(Classifier, self).__init__()

        # NOTE: Do we need to adjust the accuracies for the dataset size?
        self.factor = 1  # self.batch_size / 241
        self.normalize_image_features = args.normalize_image_features

    def normalize_per_unit(self, tensor):
        print('normalize image_feature along unit dim')
        # array: n_samples x n_units(512)
        tensor = tensor - torch.mean(tensor, 0, keepdim=True)
        tensor = tensor / torch.std(tensor, 0,  keepdim=True)
        return tensor

    @torch.no_grad()
    def forward(self, Z: torch.Tensor, Y: torch.Tensor, test=False, top_k=None) -> torch.Tensor:

        batch_size = Z.size(0)
        diags = torch.arange(batch_size).to(device)
        x = Z.view(batch_size, -1)
        y = Y.view(batch_size, -1)


        if self.normalize_image_features:
            # y = self.normalize_per_unit(y)
            pass
        # x_ = rearrange(x, 'b f -> 1 b f')
        # y_ = rearrange(y, 'b f -> b 1 f')
        # similarity = torch.nn.functional.cosine_similarity(x_, y_, dim=-1)  # ( B, B )

        # NOTE: avoid CUDA out of memory like this
        similarity = torch.empty(batch_size, batch_size).to(device)

        if test:
            pbar = tqdm(total=batch_size, desc="[Similarities]")

        for i in range(batch_size):
            for j in range(batch_size):
                similarity[i, j] = (x[i] @ y[j]) / max((x[i].norm() * y[j].norm()), 1e-8)

            if test:
                pbar.update(1)

        similarity = similarity.T

        # NOTE: max similarity of speech and M/EEG representations is expected for corresponding windows
        top1accuracy = (similarity.argmax(axis=1) == diags).to(torch.float).mean().item()
        try:
            top10accuracy = np.mean(
                [
                    label in row
                    for row, label in zip(torch.topk(similarity, 10, dim=1, largest=True)[1], diags)
                ]
            )
        except:
            print(similarity.size())
            raise
        if top_k is None:

            return top1accuracy, top10accuracy
        else:
            try:
                topkaccuracy = np.mean(
                    [
                        label in row
                        for row, label in zip(torch.topk(similarity, top_k, dim=1, largest=True)[1], diags)
                    ]
                    )
            except:
                print(similarity.size())
                raise
            return top1accuracy, top10accuracy, topkaccuracy