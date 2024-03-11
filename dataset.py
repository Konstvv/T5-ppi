from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from Bio import SeqIO
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders, trainers, processors, models

def create_tokenizer():
        vocab = [text.strip() for text in open("vocab.txt").readlines()]

        tokenizer = Tokenizer(models.BPE())
        tokenizer.add_tokens(vocab)
        tokenizer.unk_token = "[UNK]"
        tokenizer.normalizer = normalizers.Sequence([normalizers.Strip(),
                                               normalizers.Replace("\n", "")])
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.ByteLevel()])
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.save("tokenizer.json")
        return tokenizer

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

        self.tokenizer = create_tokenizer()

        dtypes = {'seq1': str, 'seq2': str}
        if self.labels:
            dtypes.update({'label': np.float16})
            self.actions = pd.read_csv(self.action_path, delimiter='\t', names=["seq1", "seq2", "label"], dtype=dtypes)
        else:
            self.actions = pd.read_csv(self.action_path, delimiter='\t', usecols=[0, 1], names=["seq1", "seq2"], dtype=dtypes)

        self.sequences = SeqIO.to_dict(SeqIO.parse(self.sequences_path, "fasta"))

        if self.max_len is not None:
            self.actions = self.actions[
                (self.actions["seq1"].apply(lambda x: len(str(self.sequences[x].seq))) < self.max_len) &
                (self.actions["seq2"].apply(lambda x: len(str(self.sequences[x].seq))) < self.max_len)
                ].reset_index(drop=True)
            for id in list(self.sequences.keys()):
                if len(str(self.sequences[id].seq)) >= self.max_len:
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

        tok1 = torch.tensor(self.tokenizer.encode(id1).ids)
        tok2 = torch.tensor(self.tokenizer.encode(id2).ids)

        # Pad to self.max_len
        pad_token = self.tokenizer.token_to_id("[PAD]")
        
        tok1 = F.pad(tok1, (0, self.max_len - len(tok1)), value=pad_token)
        tok2 = F.pad(tok2, (0, self.max_len - len(tok2)), value=pad_token)
        
        return {"tok1": tok1,
                "id2": tok2,
                "label": label}


if __name__ == '__main__':
    data = PairSequenceData(actions_file="../SENSE-PPI/data/guo_yeast_data/protein.pairs.tsv",
                            sequences_file="../SENSE-PPI/data/guo_yeast_data/sequences.fasta",
                            max_len=800)

    print(len(data))
    print(data[0])