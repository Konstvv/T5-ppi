from torch.utils.data import Dataset, IterableDataset, DataLoader
import numpy as np
import pandas as pd
from Bio import SeqIO
import torch
from tokenizer import PPITokenizer
import logging
import os
import torch.nn.functional as F

class SequencesDataset:
    def __init__(self, sequences_path : str): 

        self.sequences_path = sequences_path

        logging.info(f"Reading sequences from {self.sequences_path}")
        self.sequences = SeqIO.to_dict(SeqIO.parse(self.sequences_path, "fasta"))
        self.sequences = {k: str(v.seq) for k, v in self.sequences.items()}

        self.max_len = max([len(self.sequences[x]) for x in self.sequences])
        logging.info(f"Max sequence length of the fasta file: {self.max_len}")

    def __getitem__(self, idx):
        return self.sequences[idx]
    
    def __len__(self):
        return len(self.sequences)


class PairSequenceDataBase(Dataset):
    def __init__(self,
                 pairs_path: str,
                 sequences_dataset: SequencesDataset,
                 max_len: int = None,
                 tokenizer=None,
                 remove_long_sequences: bool = False,
                 for_esm=False):

        super().__init__()
        self.pairs_path = pairs_path
        self.dtypes = {'seq1': str, 'seq2': str, 'label': np.int8}
        self.tokenizer = tokenizer
        self.sequences_dataset = sequences_dataset
        self.remove_long_sequences = remove_long_sequences
        self.for_esm = for_esm
        
        if max_len is not None:
            self.max_len = max_len
        else:
            self.max_len = sequences_dataset.max_len

    def _tokenize(self, ids):
        tok = self.tokenizer.batch_encode_plus(
            ids,
            return_tensors="pt",
            padding="longest",#"max_length",
            max_length=self.max_len,
            truncation=True,
            add_special_tokens=False,
        )
        return tok["input_ids"], tok["attention_mask"]

    def collate_fn(self, batch):
        id1, id2, labels = zip(*batch)

        labels = torch.tensor(labels)

        if self.for_esm:
            return {'ids1': id1, 'ids2': id2, 'labels': labels}

        else:
            input_ids1, attention_mask1 = self._tokenize(id1)
            input_ids2, attention_mask2 = self._tokenize(id2)

            return {'ids1': (input_ids1, attention_mask1),
                    'ids2': (input_ids2, attention_mask2), 
                    'labels': labels,
                    'seqs1': id1,
                    'seqs2': id2}

class PairSequenceDataIterable(IterableDataset, PairSequenceDataBase):
    def __init__(self, chunk_size=1000000, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.chunk_size = chunk_size

        logging.info(f"Reading pairs from {self.pairs_path}")
        logging.info(f"Chunk size: {self.chunk_size}")


    def __iter__(self):
        chunk_iterator = pd.read_csv(self.pairs_path, delimiter='\t', names=["seq1", "seq2", "label"],
                                     dtype=self.dtypes, chunksize=self.chunk_size)

        for chunk in chunk_iterator:
            chunk['label'] = chunk['label'].astype(np.int8)
            for _, row in chunk.iterrows():
                id1 = self.sequences_dataset[row["seq1"]]
                id2 = self.sequences_dataset[row["seq2"]]
                label = row["label"]
                yield id1, id2, label


class PairSequenceData(PairSequenceDataBase):
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.data = pd.read_csv(self.pairs_path, delimiter='\t', names=["seq1", "seq2", "label"], dtype=self.dtypes)
        logging.info(f"Number of pairs: {len(self.data)}")

        if self.remove_long_sequences:
            unwanted_sequences = set()
            for record_id, sequence in self.sequences_dataset.sequences.items():
                if len(sequence.seq) > self.max_len:
                    unwanted_sequences.add(record_id)
            self.data = self.data[~self.data["seq1"].isin(unwanted_sequences)]
            self.data = self.data[~self.data["seq2"].isin(unwanted_sequences)]
            logging.info(f"Number of pairs after removing the long sequences: {len(self.data)}")

        self.data['label'] = self.data['label'].astype(np.int8)
        self.data = self.data.to_dict(orient="records")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        id1 = self.sequences_dataset[row["seq1"]]
        id2 = self.sequences_dataset[row["seq2"]]
        label = row["label"]
        return id1, id2, label
    
class PairSequenceDataPrecomputed(Dataset):
    def __init__(self,
                 pairs_path: str,
                 emb_dir: str = "esm2_embs_3B"):

        super(PairSequenceDataPrecomputed, self).__init__()
        self.pairs_path = pairs_path
        self.dtypes = {'seq1': str, 'seq2': str, 'label': np.int8}
        self.emb_dir = emb_dir

        self.data = pd.read_csv(self.pairs_path, delimiter='\t', names=["seq1", "seq2", "label"], dtype=self.dtypes)
        logging.info(f"Number of pairs: {len(self.data)}")

        self.data['label'] = self.data['label'].astype(np.int8)
        self.data = self.data.to_dict(orient="records")

    def get_emb(self, emb_id):
        f = os.path.join(self.emb_dir, '{}.pt'.format(emb_id))

        try:
            emb = torch.load(f, weights_only=False)
        except FileNotFoundError as _:
            raise Exception('Embedding file {} not found. Check your fasta file and make sure it contains '
                            'all the sequences used in training/testing.'.format(f))

        tensor_emb = emb['representations'][12] 
        tensor_len = tensor_emb.size(0)

        return tensor_emb, tensor_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        emb1, len1 = self.get_emb(row["seq1"])
        emb2, len2 = self.get_emb(row["seq2"])
        label = row["label"]
        return emb1, len1, emb2, len2, label
    
    def collate_fn(self, batch):
        emb1, len1, emb2, len2, labels = zip(*batch)

        max_len1 = max(len1)
        max_len2 = max(len2)

        labels = torch.tensor(labels)
        len1 = torch.tensor(len1)
        len2 = torch.tensor(len2)

        emb1 = torch.stack([F.pad(e, (0, 0, 0, max_len1 - e.size(0))) for e in emb1])
        emb2 = torch.stack([F.pad(e, (0, 0, 0, max_len2 - e.size(0))) for e in emb2])


        mask1 = torch.arange(emb1.shape[1])[None, :] < len1[:, None]
        mask2 = torch.arange(emb2.shape[1])[None, :] < len2[:, None]

        return {'ids1': (emb1, mask1),
                'ids2': (emb2, mask2), 
                'labels': labels}



if __name__ == '__main__':
    import time
    logging.basicConfig(level=logging.INFO)

    sequences = SequencesDataset(sequences_path="/home/volzhenin/SENSE-PPI/data/guo_yeast/sequences.fasta")

    # data = PairSequenceDataIterable(pairs_path="/home/volzhenin/SENSE-PPI/data/guo_yeast/protein.pairs.tsv",
    #                         sequences_dataset=sequences, max_len=1000, chunk_size=1000)

    data = PairSequenceDataPrecomputed(pairs_path="aphid_small.tsv", emb_dir="esm2_embs_3B")

    loader = DataLoader(dataset=data, batch_size=2, num_workers=1, collate_fn=data.collate_fn)

    epoch_times = []

    for batch in loader:
        start_time = time.time()
        (input_ids1, attention_mask1), (input_ids2, attention_mask2), labels = batch
        end_time = time.time()
        epoch_times.append(end_time - start_time)

    print(f"Average time per batch: {np.mean(epoch_times)}")
