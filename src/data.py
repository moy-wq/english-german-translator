import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Union
from tokenizers import Tokenizer
from dataclasses import dataclass

class TranslationDataset(Dataset):
    '''Transforma o conjunto dados em um dataset padrão para o torch'''
    def __init__(self, 
                 pairs: List[Union[str, str]], 
                 tokenizer_src: Tokenizer, 
                 tokenizer_tgt: Tokenizer, 
                 max_len: int
        ):
        
        self.pairs = pairs
        self.tokenizer_src = tokenizer_src 
        self.tokenizer_tgt = tokenizer_tgt
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        eng_text, de_text = self.pairs[idx]
        src_ids = torch.tensor(self.tokenizer_src.encode(eng_text).ids[:self.max_len], dtype=torch.long)
        tgt_ids = torch.tensor(self.tokenizer_tgt.encode(de_text).ids[:self.max_len], dtype=torch.long)

        return (src_ids,tgt_ids)
            


@dataclass
class BatchPair:
    """Contâiner tipado encaspulador de lote para o Transformer"""
    src_token_ids: torch.Tensor
    tgt_token_ids: torch.Tensor
    src_mask: torch.Tensor
    tgt_mask: torch.Tensor

    def to(self, device:torch.device)-> "BatchPair":
        return BatchPair(
            src_token_ids  = self.src_token_ids.to(device),
            tgt_token_ids = self.tgt_token_ids.to(device),
            src_mask = self.src_mask.to(device),
            tgt_mask = self.tgt_mask.to(device),
        )

class TranslationCollate:
    """Objeto chamável para ingestão do pad_idx na collate fn do Data Loader"""

    def __init__(self, pad_idx:int = 0):
        self.pad_idx = pad_idx

    def __call__(self, batch: List[Tuple[torch.tensor, torch.tensor]]) -> BatchPair:

        src_list, tgt_list = zip(*batch)
        src_padded = pad_sequence(src_list, batch_first=True, padding_value=self.pad_idx)
        tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=self.pad_idx)

        src_mask = (src_padded != self.pad_idx).long()
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)

        tgt_mask = (tgt_padded != self.pad_idx).long()
        tgt_mask = tgt_mask.unsqueeze(1).unsqueeze(2)
        
        return BatchPair(
            src_token_ids=src_padded,
            tgt_token_ids=tgt_padded,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
        )



def get_dataloaders(
    train_pairs: List[Tuple[str, str]],
    test_pairs: List[Tuple[str, str]],
    val_pairs: List[Tuple[str, str]],
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    batch_size: int,
    pin_memory:bool = True,
    num_workers:int = 2,
    max_len: int = 128
    ):

    """
    Recebe os dados de treinamento e devolve os dataloaders de treino, teste e validação
    """

    train_dataset = TranslationDataset(
                                    pairs = train_pairs, 
                                    tokenizer_src = tokenizer_src,
                                    tokenizer_tgt = tokenizer_tgt,
                                    max_len = max_len
                                    )
    
    val_dataset = TranslationDataset(
                                    pairs = val_pairs, 
                                    tokenizer_src = tokenizer_src,
                                    tokenizer_tgt = tokenizer_tgt,
                                    max_len = max_len
                                    )
    test_dataset = TranslationDataset(
                                    pairs = test_pairs, 
                                    tokenizer_src = tokenizer_src,
                                    tokenizer_tgt = tokenizer_tgt,
                                    max_len = max_len
                                    )
    pad_idx = tokenizer_tgt.token_to_id("[PAD]")
    collate = TranslationCollate(pad_idx)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        )

    return train_data_loader, val_data_loader, test_data_loader





        