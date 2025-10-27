import os
import glob

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import lightning as L


SEED = 36
L.seed_everything(SEED)


class WaterQualityDataset(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, mode: str = "train"):
        super().__init__()
        if self.mode == "train":
            self.dataset = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
        else:
            batch_size = 1
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.mode = mode

    def setup(self, stage=None):
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size
        
        if self.mode == "train":
            pass
        else:
            pass

        # if stage == 'fit':
        #     self.train_dataset = train_data
        #     self.val_dataset = val_data
        
        # if stage == 'test':
        #     self.test_dataset = test_data

        # if stage == 'predict':
        #     self.pred_dataset = pred_data

    def train_dataloader(self):
        # 학습 데이터로더 반환
        pass

    def val_dataloader(self):
        # 검증 데이터로더 반환
        pass

    def test_dataloader(self):
        # 테스트 데이터로더 반환
        pass