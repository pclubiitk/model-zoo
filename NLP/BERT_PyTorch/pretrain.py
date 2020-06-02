import argparse
import torch.nn as nn
import model_pretrain
import preprocess_pretraining
import data_loader_for_pretrain
from pytorch_pretrained_bert.tokenization import BertTokenizer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
# model config
parser.add_argument('--dim', type=int, default=768)
parser.add_argument('--max_len', type=int, default=512)
parser.add_argument('--heads', type=int, default=12)
parser.add_argument('--n_segs', type=int, default=2)

parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--corpus', type=str,required=True)

parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--decay', type=float, default=0.01)
parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--total_steps', type=int, default=1000000)

args = parser.parse_args()

data_loader=data_loader_for_pretrain.DataLoader(args.corpus,args.batch_size,args.max_len)
tokenizer1=BertTokenizer.from_pretrained('bert-base-uncased')

model=model_pretrain.BertPreTrain(args.dim,args.heads,args.max_len,args.n_seg).to(device)



