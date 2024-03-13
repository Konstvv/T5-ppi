from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from Bio import SeqIO
import torch
import torch.nn.functional as F
from tokenizer import PPITokenizer


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

        self.tokenizer = PPITokenizer()

        dtypes = {'seq1': str, 'seq2': str}
        if self.labels:
            dtypes.update({'label': np.float16})
            self.actions = pd.read_csv(self.action_path, delimiter='\t', names=["seq1", "seq2", "label"], dtype=dtypes)
        else:
            self.actions = pd.read_csv(self.action_path, delimiter='\t', usecols=[0, 1], names=["seq1", "seq2"],
                                       dtype=dtypes)

        self.sequences = SeqIO.to_dict(SeqIO.parse(self.sequences_path, "fasta"))

        if self.max_len is not None:
            self.actions = self.actions[
                (self.actions["seq1"].apply(lambda x: len(str(self.sequences[x].seq))) <= self.max_len) &
                (self.actions["seq2"].apply(lambda x: len(str(self.sequences[x].seq))) <= self.max_len)
                ].reset_index(drop=True)
            for id in list(self.sequences.keys()):
                if len(str(self.sequences[id].seq)) > self.max_len:
                    del self.sequences[id]

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        id1 = str(self.sequences[self.actions["seq1"][idx]].seq)
        id2 = str(self.sequences[self.actions["seq2"][idx]].seq)

        if self.labels:
            label = int(self.actions["label"][idx])
        else:
            label = 0

        tok1 = self.tokenizer.encode_plus(id1, return_tensors="pt", padding="max_length", max_length=self.max_len+2)
        tok2 = self.tokenizer.encode_plus(id2, return_tensors="pt", padding="max_length", max_length=self.max_len+2)

        return {"tok1": tok1,
                "tok2": tok2,
                "label": label}


if __name__ == '__main__':
    data = PairSequenceData(actions_file="../SENSE-PPI/data/guo_yeast_data/protein.pairs.tsv",
                            sequences_file="../SENSE-PPI/data/guo_yeast_data/sequences.fasta",
                            max_len=800)

    print(len(data))
    print(data[0]["tok1"]['input_ids'].shape)

# ADD masks for padding!!!!

# from torchtext.vocab import build_vocab_from_iterator
#
# vocab_list = [text.strip() for text in open("vocab.txt").readlines()]
# vocab = build_vocab_from_iterator(vocab_list, specials=['<unk>'])
# vocab.set_default_index(vocab['<unk>'])
#
# print(len(vocab))
