import torch
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict

class RLDataset(Dataset):
    def __init__(self, ds, tokenizer, target_col, max_length, safety_col=None):
        self.ds = ds
        self.tokenizer = tokenizer
        self.target_col = target_col
        self.max_length = max_length
        self.safety_col = safety_col

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        current_text = self.ds[idx][self.target_col]
        current_text = f"User:{current_text}\n\nAssistant: "  
        
        encoding = self.tokenizer(
            current_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        is_safety = False
        if self.safety_col and self.safety_col in self.ds.column_names:
            is_safety = bool(self.ds[idx][self.safety_col])

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "is_safety": torch.tensor(is_safety, dtype=torch.bool),
        }
        
class PreTrainDataset(Dataset):
    def __init__(self, ds, tokenizer, target_col, max_length):
        self.ds = ds
        self.tokenizer = tokenizer
        self.target_col = target_col
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        current_text = self.ds[idx][self.target_col]
        encoding = self.tokenizer(
            current_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0).clone(),
        }
        
        
def rl_create_train_val_dataloaders(ds, data_tpye, tokenizer, batch_size, val_split, target_col, max_length, safety_col=None, shuffle_train=True):
    if isinstance(ds, DatasetDict):
        ds = ds["train"].train_test_split(test_size=val_split, seed=42)
    else:
        ds = ds.train_test_split(test_size=val_split, seed=42)

    if data_tpye=="RLDataset":
        train_dataset = RLDataset(ds["train"], tokenizer, target_col, max_length, safety_col=safety_col)
        val_dataset = RLDataset(ds["test"], tokenizer, target_col, max_length, safety_col=safety_col)
    elif data_tpye=="PretrainDataset":
        train_dataset = PreTrainDataset(ds["train"], tokenizer, target_col, max_length)
        val_dataset = PreTrainDataset(ds["test"], tokenizer, target_col, max_length)
    else:
        raise ValueError("No such thing bro!")
        
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader
