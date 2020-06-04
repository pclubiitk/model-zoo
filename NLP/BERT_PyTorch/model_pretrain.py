import torch
import torch.nn as nn
import math
from utils import gelu,LayerNorm
from pytorch_pretrained_bert.tokenization import BertTokenizer

class embedding(nn.Module):
  def __init__(self,dim,vocab_size,max_len,n_segs):
    super().__init__()
    self.embed=nn.Embedding(vocab_size,dim)
    self.embedpos=nn.Embedding(max_len,dim)
    self.segembed=nn.Embedding(n_segs,dim)
    self.norm = LayerNorm(dim)
    self.drop = nn.Dropout(0.1)
  def forward(self,x,seg):
    seq_len = x.size(1)
    pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
    pos = pos.unsqueeze(0).expand_as(x)
    return self.norm(self.drop(self.embed(x)+self.embedpos(pos)+self.segembed(seg)))

class Attention(nn.Module):
  def __init__(self,dim,heads,max_len):
    super().__init__()
    self.q_mat=nn.Linear(dim,dim)
    self.k_mat=nn.Linear(dim,dim)
    self.v_mat=nn.Linear(dim,dim)
    self.dim=dim
    self.heads=heads
    self.max_len=max_len
    self.dk=dim//heads
    self.drop=nn.Dropout(0.1)
    self.softmax=nn.Softmax(-1)
    self.out = nn.Linear(dim,dim)
  def forward(self,x,mask=None):
    bs=x.size(0)
    q=self.q_mat(x).view(bs,-1,self.heads,self.dk)
    k=self.k_mat(x).view(bs,-1,self.heads,self.dk)
    v=self.v_mat(x).view(bs,-1,self.heads,self.dk)

    q=q.transpose(1,2)
    k=k.transpose(1,2)
    v=v.transpose(1,2)

    scores=torch.matmul(q,k.transpose(2,3))/math.sqrt(self.dk)

    if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)

    scores = self.drop(self.softmax(scores))
    output = torch.matmul(scores, v)

    concat = output.transpose(1,2).contiguous()\
    .view(bs, -1, self.dim)

    output=self.out(concat)   
  
    
    return output
    
class feedforward(nn.Module):
  def __init__(self,dim,heads,max_len):
    super().__init__()
    self.fc1=nn.Linear(dim,dim*4)
    self.fc2=nn.Linear(dim*4,dim)
  def forward(self,x):
    out=self.fc2(gelu(self.fc1(x)))
    return out
    
class Encoder(nn.Module):
  def __init__(self,dim,heads,max_len):
    super().__init__()
    self.attention=Attention(dim,heads,max_len)
    self.norm1=LayerNorm(dim)
    self.ff=feedforward(dim,heads,max_len)
    self.norm2=LayerNorm(dim)
    self.drop = nn.Dropout(0.1)
  def forward(self,x,mask):
    out=self.attention(x,mask)
    out=x+out
    out=self.norm1(x)
    f=out
    out=self.ff(out)
    out=self.norm2(out+f)
    return out

class AllEncode(nn.Module):
  def __init__(self,dim,heads,max_len,n_segs):
    super().__init__()
    self.tokenizer1=BertTokenizer.from_pretrained('bert-base-uncased')
    self.embed=embedding(dim,len(self.tokenizer1.vocab),max_len,n_segs)
    self.encoder1=Encoder(dim,heads,max_len)
    self.encoder2=Encoder(dim,heads,max_len)
    self.encoder3=Encoder(dim,heads,max_len)
    self.encoder4=Encoder(dim,heads,max_len)
    self.encoder5=Encoder(dim,heads,max_len)
    self.encoder6=Encoder(dim,heads,max_len)
    self.encoder7=Encoder(dim,heads,max_len)
    self.encoder8=Encoder(dim,heads,max_len)
    self.encoder9=Encoder(dim,heads,max_len)
    self.encoder10=Encoder(dim,heads,max_len)
    self.encoder11=Encoder(dim,heads,max_len)
    self.encoder12=Encoder(dim,heads,max_len)

  def forward(self,x,mask,seg):
    out=self.embed(x,seg)
    out=self.encoder1(out,mask)
    out=self.encoder2(out,mask)
    out=self.encoder3(out,mask)
    out=self.encoder4(out,mask)
    out=self.encoder5(out,mask)
    out=self.encoder6(out,mask)
    out=self.encoder7(out,mask)
    out=self.encoder8(out,mask)
    out=self.encoder9(out,mask)
    out=self.encoder10(out,mask)
    out=self.encoder11(out,mask)
    out=self.encoder12(out,mask)

    return out
    
class BertPreTrain(nn.Module):
  def __init__(self,dim,heads,max_len,n_seg):
    super().__init__()
    self.allenc=AllEncode(dim,heads,max_len,n_seg)
    self.fc1=nn.Linear(dim,dim)
    self.tanh=nn.Tanh()
    self.fc2=nn.Linear(dim,2)
    self.norm=LayerNorm(dim)
    embed_weight = self.allenc.embed.embed.weight
    n_vocab, n_dim = embed_weight.size()
    self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
    self.decoder.weight = embed_weight
    self.linear = nn.Linear(dim,dim)

  def forward(self,batch):
    input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next=batch

    out=self.allenc(input_ids,input_mask,segment_ids)

    out1=self.fc1(out[:,0])
    out1=self.tanh(out1)
    out1=self.fc2(out1)

    masked_pos1 = masked_pos[:, :, None].expand(-1, -1, out.size(-1))
    h_masked = torch.gather(out, 1, masked_pos1)
    h_masked = self.norm(gelu(self.linear(h_masked)))
    out2 = self.decoder(h_masked)

    return out1,out2    
    
