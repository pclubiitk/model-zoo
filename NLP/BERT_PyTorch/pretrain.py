import argparse
import torch
import torch.nn as nn
import model_pretrain
import preprocess_pretraining
import data_loader_for_pretrain
from pytorch_pretrained_bert.tokenization import BertTokenizer
import transformers
from utils import save_model


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


model=model_pretrain.BertPreTrain(args.dim,args.heads,args.max_len,args.n_segs).to(device)
data_loader=data_loader_for_pretrain.DataLoader(args.corpus,args.batch_size,args.max_len)
tokenizer1=BertTokenizer.from_pretrained('bert-base-uncased')


criterion1=nn.CrossEntropyLoss().to(device)
criterion2=nn.CrossEntropyLoss().to(device)

optimizer=transformers.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1,args.beta2), weight_decay=args.decay)
lr_scheduler=transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup, num_training_steps=args.total_steps, last_epoch=- 1)

def loss_func(model,batch):
  input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next = batch
  clsf,mlm=model(batch)
  lossclf=criterion1(clsf,is_next)
  losslm=criterion2(mlm.transpose(1,2),masked_ids)
  return lossclf,losslm

step=0
for epoch in range(args.epochs):
  for i,batch in enumerate(data_loader):
    batch = [t.to(device) for t in batch]
    optimizer.zero_grad()
    lossclf,losslm=loss_func(model,batch)
    loss=lossclf+losslm
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    step=step+1
    
    print("LOSS:%f LOSSCLF:%f LOSSLM:%f "%(loss,lossclf,losslm),"epoch[%d/%d] step[%d/%d]"%(epoch+1,args.epochs,step,args.total_steps))
 
 
  save_model(epoch,model,args.save_dir)



