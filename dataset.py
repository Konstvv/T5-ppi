from torch.utils.data import Dataset, IterableDataset, DataLoader
import numpy as np
import pandas as pd
from Bio import SeqIO
import torch
from tokenizer import PPITokenizer
import logging

class SequencesDataset:
    def __init__(self, 
                 sequences_path : str, 
                 max_len=None): 
        self.max_len = max_len
        self.sequences_path = sequences_path

        logging.info(f"Reading sequences from {self.sequences_path}")
        self.sequences = SeqIO.to_dict(SeqIO.parse(self.sequences_path, "fasta"))

        if self.max_len is None:
            self.max_len = max([len(str(self.sequences[x].seq)) for x in self.sequences])
            logging.info(f"Max sequence length automatically set to the length of the largest sequence: {self.max_len}")

    def __getitem__(self, idx):
        return self.sequences[idx].seq
    
    def __len__(self):
        return len(self.sequences)


class PairSequenceDataBase(Dataset):
    def __init__(self,
                 pairs_path: str,
                 sequences_dataset: SequencesDataset):

        super().__init__()
        self.pairs_path = pairs_path
        
        self.dtypes = {'seq1': str, 'seq2': str, 'label': np.int8}

        self.tokenizer = PPITokenizer()

        self.sequences_dataset = sequences_dataset
        self.max_len = sequences_dataset.max_len

    def collate_fn(self, batch):
        id1, id2, labels = zip(*batch)

        combined_ids = list(id1) + list(id2)

        tok_combined = self.tokenizer.batch_encode_plus(
            combined_ids,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_len + 2,
            truncation=True,
        )

        input_ids = tok_combined["input_ids"]
        attention_mask = tok_combined["attention_mask"]

        input_ids1 = input_ids[: len(id1)]
        input_ids2 = input_ids[len(id1):]
        attention_mask1 = attention_mask[: len(id1)]
        attention_mask2 = attention_mask[len(id1):]

        labels = torch.tensor(labels)

        return (input_ids1, attention_mask1), (input_ids2, attention_mask2), labels
    

class PairSequenceDataIterable(IterableDataset, PairSequenceDataBase):
    def __init__(self, chunk_size=1000000, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.chunk_size = chunk_size

    def __iter__(self):
        chunk_iterator = pd.read_csv(self.pairs_path, delimiter='\t', names=["seq1", "seq2", "label"],
                                     dtype=self.dtypes, chunksize=self.chunk_size)

        for chunk in chunk_iterator:
            for _, row in chunk.iterrows():
                id1 = str(self.sequences_dataset[row["seq1"]])
                id2 = str(self.sequences_dataset[row["seq2"]])
                label = int(row["label"])
                yield id1, id2, label


class PairSequenceData(PairSequenceDataBase):
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.data = pd.read_csv(self.pairs_path, delimiter='\t', names=["seq1", "seq2", "label"], dtype=self.dtypes)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        id1 = str(self.sequences_dataset[row["seq1"]])
        id2 = str(self.sequences_dataset[row["seq2"]])
        label = int(row["label"])
        return id1, id2, label


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    sequences = SequencesDataset(sequences_path="/home/volzhenin/SENSE-PPI/data/guo_yeast/sequences.fasta", max_len=800)

    data = PairSequenceDataIterable(pairs_path="/home/volzhenin/SENSE-PPI/data/guo_yeast/protein.pairs.tsv",
                            sequences_dataset=sequences, chunk_size=1000)

    loader = DataLoader(dataset=data, batch_size=1, num_workers=1, collate_fn=data.collate_fn)

    for batch in loader:
        (input_ids1, attention_mask1), (input_ids2, attention_mask2), labels = batch
        print(input_ids1, attention_mask1, input_ids2, attention_mask2, labels)
        break