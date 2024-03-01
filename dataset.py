from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from Bio import SeqIO
import ankh
import torch
from tqdm import tqdm


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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ankh_model, self.tokenizer = ankh.load_large_model()
        self.ankh_model.to(self.device)

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

        ids = self.tokenizer.batch_encode_plus([str(self.sequences[x].seq) for x in list(self.sequences.keys())],
                                               add_special_tokens=True,
                                               padding="max_length",
                                               max_length=self.max_len)
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

        batch_size = 32
        self.actions = self.actions[:len(self.actions) - (len(self.actions) % batch_size)]

        embeddings = torch.tensor([]).to(self.device)
        for i in tqdm(range(0, len(input_ids), batch_size)):
            with torch.no_grad():
                embedding_repr = self.ankh_model(input_ids=input_ids[i:i+batch_size], attention_mask=attention_mask[i:i+batch_size])
            embeddings = torch.cat((embeddings, embedding_repr.last_hidden_state), dim=0)

        self.embeddings = {}
        for i, id in enumerate(list(self.sequences.keys())):
            self.embeddings[id] = embedding_repr.last_hidden_state[i]

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        # id1 = str(self.sequences[self.actions["seq1"][idx]].seq)
        # id2 = str(self.sequences[self.actions["seq2"][idx]].seq)

        emb_0 = self.embeddings[self.actions["seq1"][idx]]
        emb_1 = self.embeddings[self.actions["seq2"][idx]]

        if self.labels:
            label = int(self.actions["label"][idx])
        else:
            label = 0

        # return {"input_ids": torch.tensor(ids['input_ids']),
        #         'attention_mask': torch.tensor(ids['attention_mask']),
        #         "label": label}
        #         # "lens": torch.tensor([len(id1), len(id2)])}

        return {"emb_0": emb_0,
                "emb_1": emb_1,
                "label": label}


if __name__ == '__main__':
    data = PairSequenceData(actions_file="../SENSE-PPI/data/guo_yeast_data/protein.pairs.tsv",
                            sequences_file="../SENSE-PPI/data/guo_yeast_data/sequences.fasta",
                            max_len=800)

    print(len(data))
    print(data[0]['input_ids'].shape)