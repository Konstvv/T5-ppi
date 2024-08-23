# coding=utf-8
# Copyright 2022 Meta and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import List, Optional
from Bio import SeqIO

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import BPEDecoder

VOCAB_LIST = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I',
              'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H',
              'W', 'C', 'X', 'B', 'U', 'Z', 'O']


class PPITokenizer(PreTrainedTokenizer):

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
            self,
            unk_token="<unk>",
            cls_token="<cls>",
            pad_token="<pad>",
            mask_token="<mask>",
            eos_token="<eos>",
            **kwargs,
    ):
        self.all_tokens = VOCAB_LIST
        self._id_to_token = dict(enumerate(self.all_tokens))
        self._token_to_id = {tok: ind for ind, tok in enumerate(self.all_tokens)}
        super().__init__(
            unk_token=unk_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            eos_token=eos_token,
            **kwargs,
        )

        self.unique_no_split_tokens = self.all_tokens
        self._update_trie(self.unique_no_split_tokens)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def _tokenize(self, text, **kwargs):
        return text.split()

    def get_vocab(self):
        base_vocab = self._token_to_id.copy()
        base_vocab.update(self.added_tokens_encoder)
        return base_vocab

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        cls = [self.cls_token_id]
        sep = [self.eos_token_id]
        if token_ids_1 is None:
            if self.eos_token_id is None:
                return cls + token_ids_0
            else:
                return cls + token_ids_0 + sep
        elif self.eos_token_id is None:
            raise ValueError("Cannot tokenize multiple sequences when EOS token is not set!")
        return cls + token_ids_0 + sep + token_ids_1 + sep  # Multiple inputs always have an EOS token

    def get_special_tokens_mask(
            self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )

            return [1 if token in self.all_special_ids else 0 for token in token_ids_0]
        mask = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            mask += [0] * len(token_ids_1) + [1]
        return mask

    def save_vocabulary(self, save_directory, **kwargs):
        vocab_file = os.path.join(save_directory, "vocab.txt")
        with open(vocab_file, "w") as f:
            f.write("\n".join(self.all_tokens))

    @property
    def vocab_size(self) -> int:
        return len(self.all_tokens)


def train_tokenizer(vocab_size: int, fasta_file: str, save_dir: str):
    with open(fasta_file) as f:
        records = SeqIO.parse(f, "fasta")
        seqs = [str(record.seq) for record in records]

    special_tokens = ['[PAD]', '[UNK]']

    tokenizer = Tokenizer(BPE())
    tokenizer.decoder = BPEDecoder()
    tokenizer.add_special_tokens(special_tokens)

    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.train_from_iterator(seqs, trainer=trainer)

    pre_trained_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token=special_tokens[1], pad_token=special_tokens[0])
    pre_trained_tokenizer.save_pretrained(save_dir)


if __name__ == '__main__':
    vocab_size=1000
    fasta_file = '/home/volzhenin/T5-ppi/string12.0_combined_score_900.fasta'
    save_dir = '/home/volzhenin/T5-ppi/tokenizer'

    train_tokenizer(vocab_size, fasta_file, save_dir)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(save_dir)

    prot = "MKVWAVAKLKLW"
    print(prot)
    print(tokenizer.batch_encode_plus([prot, prot+prot], return_tensors="pt", padding='longest', max_length=8, truncation=True))
    print(len(tokenizer))
    print(tokenizer.special_tokens_map)
    print(tokenizer.decode(tokenizer.encode(prot)))
