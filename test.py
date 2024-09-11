import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import PPITransformerModel

from dataset import PairSequenceData, SequencesDataset
from transformers import PreTrainedTokenizerFast
import os

def pred_model(seq1, seq2):
    toks = tokenizer.batch_encode_plus([seq1, seq2], 
                                        return_tensors="pt",
                                        padding="longest",#"max_length",
                                        max_length=1000,
                                        truncation=True,
                                        add_special_tokens=False,
                                    )
    
    input_ids1 = toks['input_ids'][0]
    input_ids2 = toks['input_ids'][1]
    attention_mask1 = toks['attention_mask'][0]
    attention_mask2 = toks['attention_mask'][1]

    parser = argparse.ArgumentParser()
    parser = PPITransformerModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    params = parser.parse_args()
    params.max_len = 1000
    
    model = PPITransformerModel(params, 
                    ntoken=len(tokenizer), 
                    embed_dim=512,#64, #512
                    hidden_dim=2048, #512 #2048
                    num_siamese_layers=12, #6
                    num_cross_layers=6, #3
                    num_heads=8, #8
                    dropout=0.1)

    checkpoint_folder = os.path.join('ppi-transformer/6qfgdx0p/', 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_folder, os.listdir(checkpoint_folder)[0])
    print('Loading model from checkpoint:', checkpoint_path)
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt['state_dict'])

    with torch.no_grad():
        pred = model([(input_ids1.unsqueeze(0), attention_mask1.unsqueeze(0)), (input_ids2.unsqueeze(0), attention_mask2.unsqueeze(0))])
        return pred

def test_model(checkpoint_folder, dataset):
    parser = argparse.ArgumentParser()
    parser = PPITransformerModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    params = parser.parse_args()

    params.max_len = 1000

    model = PPITransformerModel(params, 
                        ntoken=len(dataset.tokenizer), 
                        embed_dim=512,#64, #512
                        hidden_dim=2048, #512 #2048
                        num_siamese_layers=12, #6
                        num_cross_layers=6, #3
                        num_heads=8, #8
                        dropout=0.1)

    checkpoint_folder = os.path.join(checkpoint_folder, 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_folder, os.listdir(checkpoint_folder)[0])
    print('Loading model from checkpoint:', checkpoint_path)
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt['state_dict'])

    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'auto',
                          devices=params.devices, 
                          num_nodes=params.num_nodes)

    pred_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=8, shuffle=False, collate_fn=dataset.collate_fn)
    trainer.test(model, pred_loader)

if __name__ == '__main__':
    tokenizer = PreTrainedTokenizerFast.from_pretrained('tokenizer')

    # sequences = SequencesDataset(sequences_path="/home/volzhenin/SENSE-PPI/data/string12.0_species/sequences_4932_yeast.fasta")

    # dataset = PairSequenceData(pairs_path="/home/volzhenin/SENSE-PPI/data/string12.0_species/protein.pairs_4932_yeast.tsv",
    #                             sequences_dataset=sequences, tokenizer=tokenizer, max_len=800)
    
    # # test_model(checkpoint_folder='logs/AttentionModelBase/version_173', dataset=dataset)
    # test_model(checkpoint_folder='ppi-transformer/6qfgdx0p/', dataset=dataset)

    seq1 = 'QSALTQPPSASGSLGQSVTISCTGTSSDVGGYNYVSWYQQHAGKAPKVIIYEVNKRPSGVPDRFSGSKSGNTASLTVSGLQAEDEADYYCSSYEGSDNFVFGTGTKVTVLGQPKANPTVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADGSPVKAGVETTKPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTECS'
    seq2 = 'QSALTQPPSASGSLGQSVTISCTGTSSDVGGYNYVSWYQQHAGKAPKVIIYEVNKRPSGVPDRFSGSKSGNTASLTVSGLQAEDEADYYCSSYEGSDNFVFGTGTKVTVLGQPKANPTVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADGSPVKAGVETTKPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTECS'

    print(pred_model(seq1, seq2))