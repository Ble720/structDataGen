import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from torch.utils.data import Dataset, DataLoader, random_split
from utils.bin import get_num_bins

class TabularDataset(Dataset):
    def __init__(self, x_cat, x_bin, x_off, cat_masks, bin_masks, is_discrete_mask):
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.x_bin = torch.tensor(x_bin, dtype=torch.long)
        self.x_off = torch.tensor(x_off, dtype=torch.float32)
        self.cat_masks = torch.tensor(cat_masks, dtype=torch.float32)
        self.bin_masks = torch.tensor(bin_masks, dtype=torch.float32)
        self.is_discrete_mask = torch.tensor(is_discrete_mask, dtype=torch.bool)

    def __len__(self):
        return len(self.x_cat)

    def __getitem__(self, idx):
        return (
            self.x_cat[idx], 
            self.x_bin[idx], 
            self.x_off[idx],
            self.cat_masks[idx], 
            self.bin_masks[idx]
        )

class TabularDataHandler:
    def __init__(self, csv_path, config, batch_size=64):
        self.csv_path = csv_path
        self.config = config
        self.batch_size = batch_size
        self.cat_encoders = {}
        self.bin_discretizer = None
        self.bin_medians = None
        
        self.train_ds = None
        self.val_ds = None

    def prepare_data(self, train_split=0.8, seed=42):
        df = pd.read_csv(self.csv_path)
        
        # 1. Process Categorical
        x_cat_list = []
        for col in self.config['cat']:
            le = LabelEncoder()
            encoded = le.fit_transform(df[col].fillna("N/A").astype(str)) + 1
            self.cat_encoders[col] = le
            x_cat_list.append(encoded)
        x_cat = np.stack(x_cat_list, axis=1) if x_cat_list else np.empty((len(df), 0))

        # 2. Process Binary/Continuous
        bin_cols = self.config['bin']
        if bin_cols:
            self.bin_medians = df[bin_cols].median()
            data_df = df[bin_cols].fillna(self.bin_medians)
            bin_counts = [get_num_bins(data_df[col].values, 5) for col in bin_cols]
            
            self.bin_discretizer = KBinsDiscretizer(
                n_bins=bin_counts, encode='ordinal', strategy='quantile'
            )
            x_bin = self.bin_discretizer.fit_transform(data_df) + 1
            
            edges = self.bin_discretizer.bin_edges_
            x_off = np.zeros_like(data_df.values, dtype=np.float32)
            for i, col in enumerate(bin_cols):
                e = edges[i]
                b = np.clip((x_bin[:, i] - 1).astype(int), 0, len(e) - 2)
                low, high = e[b], e[b+1]
                diff = high - low
                x_off[:, i] = np.where(diff > 0, (data_df.values[:, i] - low) / diff, 0.5)
        else:
            x_bin = x_off = np.empty((len(df), 0))

        is_discrete_mask = np.array([col in self.config['dis'] for col in bin_cols])
        cat_masks = (~df[self.config['cat']].isna()).astype(int).values
        bin_masks = (~df[bin_cols].isna()).astype(int).values

        full_ds = TabularDataset(x_cat, x_bin, x_off, cat_masks, bin_masks, is_discrete_mask)
        train_size = int(train_split * len(full_ds))
        self.train_ds, self.val_ds = random_split(
            full_ds, 
            [train_size, len(full_ds) - train_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        # Store indices so we can save them in metadata
        self.train_indices = self.train_ds.indices
        self.val_indices = self.val_ds.indices
        
        return self.train_ds, self.val_ds

    def get_dataloaders(self):
        train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def train_dataloader(self, drop_last=False):
        if self.train_ds is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, drop_last=drop_last)

    def get_cardinalities(self):
        cat_cards = [int(len(le.classes_) + 1) for le in self.cat_encoders.values()]
        bin_cards = [int(n) + 1 for n in self.bin_discretizer.n_bins_] if self.bin_discretizer else []
        return {"cat": cat_cards, "bin": bin_cards, "num_cont": len(self.config['bin'])}

    def save_metadata(self, path="metadata.joblib"):
        metadata = {
            "cat_encoders": self.cat_encoders,
            "bin_discretizer": self.bin_discretizer,
            "bin_medians": self.bin_medians,
            "config": self.config,
            "cardinalities": self.get_cardinalities(),
            "train_indices": getattr(self, "train_indices", None),
            "val_indices": getattr(self, "val_indices", None)
        }
        joblib.dump(metadata, path)

    @classmethod
    def load_from_metadata(cls, path="metadata.joblib", batch_size=64):
        """Creates a handler instance from saved metadata (for inference)."""
        metadata = joblib.load(path)
        instance = cls(csv_path=None, config=metadata["config"], batch_size=batch_size)
        instance.cat_encoders = metadata["cat_encoders"]
        instance.bin_discretizer = metadata["bin_discretizer"]
        instance.bin_medians = metadata["bin_medians"]
        return instance
    
    def inverse_transform(self, pred_cat, pred_bin, pred_off):
        synth_df = pd.DataFrame()

        for i, col in enumerate(self.config['cat']):
            le = self.cat_encoders[col]
            indices = pred_cat[:, i]
            
            clean_indices = np.clip(indices - 1, 0, len(le.classes_) - 1)
            synth_df[col] = le.inverse_transform(clean_indices)

        bin_cols = self.config['bin']
        edges = self.bin_discretizer.bin_edges_
        
        for i, col in enumerate(bin_cols):
            b_idx = np.clip(pred_bin[:, i] - 1, 0, len(edges[i]) - 2)
            off = pred_off[:, i]
            
            low = edges[i][b_idx]
            high = edges[i][b_idx + 1]
            
            actual_val = low + (off * (high - low))
            synth_df[col] = actual_val

        return synth_df