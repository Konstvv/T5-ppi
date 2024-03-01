from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from Bio import SeqIO
import ankh
import torch

class PairSequenceData(Dataset):
    def __init__(self,
                 actions_file,
                 sequences_file,
                 max_len=None,
                 labels=True):

        super(PairSequenceData, self).__init__()
        self.max_len = max_len
        self.action_path = actions_file
        self.sequences_path = sequences_file
        self.labels = labels
        _, self.tokenizer = ankh.load_large_model()

        dtypes = {'seq1': str, 'seq2': str}
        if self.labels:
            dtypes.update({'label': np.float16})
            self.actions = pd.read_csv(self.action_path, delimiter='\t', names=["seq1", "seq2", "label"], dtype=dtypes)
        else:
            self.actions = pd.read_csv(self.action_path, delimiter='\t', usecols=[0, 1], names=["seq1", "seq2"], dtype=dtypes)

        self.sequences = SeqIO.to_dict(SeqIO.parse(self.sequences_path, "fasta"))

        if max_len is not None:
            self.actions = self.actions[
                (self.actions["seq1"].apply(lambda x: len(str(self.sequences[x].seq))) <= max_len) &
                (self.actions["seq2"].apply(lambda x: len(str(self.sequences[x].seq))) <= max_len)
            ].reset_index(drop=True)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        id1 = str(self.sequences[self.actions["seq1"][idx]].seq)
        id2 = str(self.sequences[self.actions["seq2"][idx]].seq)

        ids = self.tokenizer.batch_encode_plus([id1, id2], add_special_tokens=True, padding="longest")

        if self.labels:
            label = int(self.actions["label"][idx])
        else:
            label = 0

        return {"input_ids": torch.tensor(ids['input_ids']),
                'attention_mask': torch.tensor(ids['attention_mask']),
                "label": label,
                "lens": [len(id1), len(id2)]}


if __name__ == '__main__':
    data = PairSequenceData(actions_file="../SENSE-PPI/data/guo_yeast_data/protein.pairs.tsv",
                            sequences_file="../SENSE-PPI/data/guo_yeast_data/sequences.fasta",
                            max_len=800)
    
    print(len(data))
    print(data[0])
