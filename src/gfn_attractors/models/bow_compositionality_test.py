import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from functools import cached_property
from ..data.image_datamodule import ImageDataModule
from ..data.simple_data_module import SimpleDataModule


class BoWModel(nn.Module):

    def __init__(self, num_tokens, num_outputs):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_outputs = num_outputs
        self.model = nn.Embedding(num_tokens, sum(num_outputs))
        self.bias = nn.Parameter(torch.randn(sum(num_outputs)))

    def forward(self, x):
        """
        x: tensor of shape (..., sequence_length)
        """
        logits = self.model(x).sum(-2) + self.bias
        return logits.split(self.num_outputs, dim=-1)


class BoWCompositionalityTest(nn.Module):

    def __init__(self, data_module: ImageDataModule):
        """
        w: tensor of shape (n, sequence_length)
        is_training_set: tensor of shape (num_train)
        obj_indices: tensor of shape (n)
        """
        super().__init__()
        self.data_module = data_module
        self.feature_vectors = data_module.get_feature_vectors()

    @cached_property
    def num_feature_labels(self):
        feature_names, num_feature_labels = self.data_module.get_feature_labels()
        return num_feature_labels

    @cached_property
    def feature_names(self):
        feature_names, num_feature_labels = self.data_module.get_feature_labels()
        return feature_names
    
    @property
    def device(self):
        return self.feature_vectors.device
    
    def to(self, device):
        self.feature_vectors.to(device)
        return self

    def get_w_offset(self, w):
        """
        w: tensor of shape (n, sequence_length)

        Recodes each token so that each (token, position) becomes a unique token.
        """
        offsets = (1 + w.max(dim=0)[0]).cumsum(0)[:-1]
        offsets = F.pad(offsets, (1, 0))
        w_offset = offsets + w
        return w_offset
    
    @torch.enable_grad()
    def train(self, num_updates, w_offset, train_indices, batch_size=512):
        num_tokens = 1 + w_offset.max()

        model = BoWModel(num_tokens, self.num_feature_labels).to(self.device)
        if batch_size is None:
            batch_size = len(self.feature_vectors)
        data_module = SimpleDataModule(batch_size, train_indices.cpu(), w=w_offset.to(self.device), target=self.feature_vectors).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_updates)

        update = 0
        done = False
        while not done:
            for batch in data_module.train_dataloader():
                update += 1
                logits = model(batch['w'])
                loss = 0
                for i, logits_i in enumerate(logits):
                    targets = batch['target'][:,i]
                    loss += F.cross_entropy(logits_i, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                if update == num_updates:
                    done = True
                    break

        df_accuracy = self.get_df_accuracy(model, data_module)
        return model, loss, df_accuracy
    
    @torch.no_grad()
    def get_df_accuracy(self, model: BoWModel, simple_data_module: SimpleDataModule):
        rows = []
        for split in ['train', 'val']:
            batch = simple_data_module.sample(1000, train=split=='train', val=split=='val')
            outputs = model(batch['w'])
            targets = batch['target']

            for i in range(len(outputs)):
                feature = self.feature_names[i]
                predicted = outputs[i].argmax(-1)
                
                actual = targets[:,i]
                accuracy = (predicted == actual).float().mean().item()
                rows.append({'split': split, 'feature': feature, 'accuracy': accuracy})
        return pd.DataFrame(rows)
