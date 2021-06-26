from train import StackGAN
import helper_functions.config as cfg
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--eval_', type=bool, default=True,help='whether to evaluate by generating images or not')
parser.add_argument('--channels', type=int, default=3,help='number of input colour channels')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batchSize', type=int, default=40,help='Batch Size')
parser.add_argument('--saveInterval', type=int, default=1,help='after how many epochs to save the model')
parser.add_argument('--embeddingsDim', type=int, default=128, help='embedding dimensions')
parser.add_argument('--StageNum', type=int, default=3, help='Number of Stages')
parser.add_argument('--generatorLR', type=float, default=2e-4)
parser.add_argument('--discriminatorLR', type=float, default=2e-4)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--zDim', type=int, default=100)

args = parser.parse_args()

if __name__ == "__main__":
    import sys
    print(sys.version)
    import os
    try:
        os.makedirs("data", exist_ok=True)
        print("data folder made")
    except:
        print("data folder already present")
    try:
        os.makedirs("images", exist_ok=True)
        print("images folder made")
    except:
        print("images folder already present")
    try:
        os.makedirs("models", exist_ok=True)
        print("models folder made")
    except:
        print("models folder already present")
    
    
    GAN = StackGAN(eval_ = args.eval_, 
                   inn_channels = args.channels, 
                   generatorLR = args.generatorLR, 
                   discriminatorLR = args.discriminatorLR, 
                   StageNum = args.StageNum,
                   beta1=args.beta1,
                   beta2=args.beta2,
                   zDim=args.zDim)
    GAN.train(args.epochs, args.batchSize, args.saveInterval)
