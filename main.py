import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

# from src.dataset import ImageNetDataModule
# from src.utils import rename_dir
from src.model import create_model


SEED = 36
L.seed_everything(SEED)


class WaterQualityModel(L.LightningModule):
    def __init__(self, model_name: str, lr: float = 1e-3):
        super().__init__()
        self.model = create_model(model_name)
        self.criterion = nn.MSELoss()
        self.lr = lr

        self.losses = []
        self.labels = []
        self.predictions = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        self.log('val_loss', val_loss)

        self.losses.append(val_loss)
        # self.val_labels.append(y)
        # self.val_predictions.append(y_hat)
        self.log('valid_loss', val_loss)
        return val_loss

    def on_validation_epoch_end(self):
        # labels = np.concatenate(np.array(self.val_labels, dtype=object))
        # predictions = np.concatenate(np.array(self.val_predictions, dtype=object))
        # labels = labels.tolist()
        # predictions = predictions.tolist()

        val_epoch_loss = sum(self.losses)/len(self.losses)

        self.log('val_epoch_loss', val_epoch_loss)
        
        # self.labels.clear()
        # self.predictions.clear()
        self.losses.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.criterion(y_hat, y)
        self.log('test_loss', test_loss)

    def on_test_epoch_end(self):
        # labels = np.concatenate(np.array(self.val_labels, dtype=object))
        # predictions = np.concatenate(np.array(self.val_predictions, dtype=object))
        # labels = labels.tolist()
        # predictions = predictions.tolist()

        test_epoch_loss = sum(self.losses)/len(self.losses)

        self.log('test_epoch_loss', test_epoch_loss)
        
        # self.labels.clear()
        # self.predictions.clear()
        self.losses.clear()

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    

def main(model_name: str, input_size: int, hidden_size: int, num_layers: int, output_size: int, mode: str):
    model = WaterQualityModel(model_name)

    x = torch.randn(4, 10, input_size)
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, default='lstm', help='Model name')
    parser.add_argument('-i', '--input_size', type=int, default=8, help='Feature size')
    parser.add_argument('-h', '--hidden_size', type=int, default=16, help='Model hidden size')
    parser.add_argument('-l', '--num_layers', type=int, default=1, help='Model layers')
    parser.add_argument('-o', '--output_size', type=int, default=1, help='Output size')
    parser.add_argument('-mo', '--mode', type=str, default='train', help='Mode: train or test')
    args = parser.parse_args()

    main(args.model_name, args.input_size, args.hidden_size, args.num_layers, args.output_size, args.mode)