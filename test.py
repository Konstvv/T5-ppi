import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dataset import PairSequenceData, SequencesDataset
from transformers import PreTrainedTokenizerFast
import os

def pred_model(model, seq1, seq2):
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

    model.eval()

    with torch.no_grad():
        pred, (attn1, attn2) = model([(input_ids1.unsqueeze(0), attention_mask1.unsqueeze(0)), (input_ids2.unsqueeze(0), attention_mask2.unsqueeze(0))], return_attention=True)

    attn1 = attn1.mean(-1).squeeze()
    attn2 = attn2.mean(-1).squeeze()

    tokens_seq1 = tokenizer.convert_ids_to_tokens(input_ids1)
    tokens_seq2 = tokenizer.convert_ids_to_tokens(input_ids2)

    return pred, (attn1, attn2), (tokens_seq1, tokens_seq2)

def test_model(model, dataset):

    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'auto',
                          devices=params.devices, 
                          num_nodes=params.num_nodes)

    pred_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=8, shuffle=False, collate_fn=dataset.collate_fn)
    trainer.test(model, pred_loader)

if __name__ == '__main__':
    from model_esm import PPITransformerModel
    # from model_esm import PPITransformerModel

    tokenizer = PreTrainedTokenizerFast.from_pretrained('tokenizer')

    parser = argparse.ArgumentParser()
    parser = PPITransformerModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    params = parser.parse_args()
    params.max_len = 1000
    
    # model = PPITransformerModel(params, 
    #                 ntoken=len(tokenizer), 
    #                 embed_dim=512,#64, #512
    #                 hidden_dim=2048, #512 #2048
    #                 num_siamese_layers=12, #6
    #                 num_cross_layers=6, #3
    #                 num_heads=8, #8
    #                 dropout=0.1)
    # checkpoint_folder = os.path.join('ppi-transformer/59lrxzqy/', 'checkpoints')

    model = PPITransformerModel(params, 
                           ntoken=len(tokenizer), 
                           embed_dim=480, #64/ #512
                            hidden_dim=2048, #512/ #2048
                            num_cross_layers=6, #3
                            num_heads=8, #8
                            dropout=0.1)
    checkpoint_folder = os.path.join('ppi-transformer/tmg60lyz', 'checkpoints')

    checkpoint_path = os.path.join(checkpoint_folder, os.listdir(checkpoint_folder)[0])
    print('Loading model from checkpoint:', checkpoint_path)
    ckpt = torch.load(checkpoint_path)
    for key in list(ckpt['state_dict'].keys()):
        if key.startswith('embedding'):
            del ckpt['state_dict'][key]
    model.load_state_dict(ckpt['state_dict'])

    sequences = SequencesDataset(sequences_path="/home/volzhenin/T5-ppi/aphid_small.fasta")

    dataset = PairSequenceData(pairs_path="/home/volzhenin/T5-ppi/aphid_small.tsv",
                                sequences_dataset=sequences,
                                # tokenizer=tokenizer, 
                                for_esm=True)
    
    # test_model(checkpoint_folder='logs/AttentionModelBase/version_173', dataset=dataset)
    test_model(model, dataset)

    # seq1 = 'QSVLTQPPSVSAAPGQKVTISCSNVGKNFVSWYQQFPGTAPKVVIYDTDKRPSDIPDRFSGSKSGTSATLDITGLQTGDEADYYCGTWDSGLNGGVFGGGTKVTVLGQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPVKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHKSYSCQVTHEGSTVEKTVAPTECS'
    # seq2 = 'QSVLTQPPSVSAAPGQKVTISCSNVGKNFVSWYQQFPGTAPKVVIYDTDKRPSDIPDRFSGSKSGTSATLDITGLQTGDEADYYCGTWDSGLNGGVFGGGTKVTVLGQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPVKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHKSYSCQVTHEGSTVEKTVAPTECS'

    # print(seq1)
    # print(seq2)

    # pred, (attn1, attn2), (tokens_seq1, tokens_seq2) = pred_model(model, seq1, seq2)

    # print(pred)

    # from matplotlib import pyplot as plt
    # import numpy as np
    # from mpl_toolkits.mplot3d import Axes3D

    # fig, ax = plt.subplots(1, 2, figsize=(40, 20))

    # attn1 = attn1.numpy()
    # attn1[attn1 == 0] = np.nan
    # ax[0].imshow(attn1)
    # # the first dimantion corresponds to tokens_seq1, the second to tokens_seq2
    # ax[0].set_xticks(range(len(tokens_seq2)))
    # ax[0].set_xticklabels(tokens_seq2, rotation=90)
    # ax[0].set_yticks(range(len(tokens_seq1)))
    # ax[0].set_yticklabels(tokens_seq1)

    # attn2 = attn2.numpy()
    # attn2[attn2 == 0] = np.nan
    # ax[1].imshow(attn2)
    # ax[1].set_xticks(range(len(tokens_seq1)))
    # ax[1].set_xticklabels(tokens_seq1, rotation=90)
    # ax[1].set_yticks(range(len(tokens_seq2)))
    # ax[1].set_yticklabels(tokens_seq2)

    # plt.savefig('attn.pdf')



#esm2
#     test_BinaryAUROC          0.9221996665000916
#     test_BinaryAccuracy        0.9392727017402649
# test_BinaryAveragePrecision    0.7358049750328064
#     test_BinaryF1Score         0.6802603602409363
# test_BinaryMatthewsCorrCoef    0.6474987864494324
#    test_BinaryPrecision        0.6524054408073425
#      test_BinaryRecall         0.7106000185012817
#          test_loss             0.19551974534988403

#      test_BinaryAUROC           0.953594446182251
#     test_BinaryAccuracy        0.9434000253677368
# test_BinaryAveragePrecision    0.8152315616607666
#     test_BinaryF1Score         0.7209823131561279
# test_BinaryMatthewsCorrCoef    0.6944939494132996
#    test_BinaryPrecision        0.6532402038574219
#      test_BinaryRecall         0.8044000267982483
#          test_loss             0.16178685426712036