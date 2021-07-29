import argparse
import numpy as np
from prettytable import PrettyTable
from models import Inception_Inflated3d

parser = argparse.ArgumentParser()
parser.add_argument('--type',help='Please Specify the type, 1 stream i.e RGB(3 CHANNELS) or FLOW(2 CHANNELS) or BOTH COMBINED(RGB and OPTICAL FLOW)',choices=['rgb','optical_flow','both']);
parser.add_argument('--pretrain', help='Please specify whether you want to use model pretrained on kinetics dataset or imagenet or both datasets',choices=['kinetics','bothsets'])
args = parser.parse_args()
classes = [ text.strip() for text in open('assets/label_map.txt','r') ]

#Checking whether the user wants to use the RGB or the Optical flow model trained on the kinetics-400 and the Imagenet Datasets.

if args.type == 'rgb':
    im = np.load('assets/v_CricketShot_g04_c01_rgb.npy')
    if args.pretrain == 'kinetics':
        rgb_model = Inception_Inflated3d(
                    include_top= True,
                    weights= 'rgb_kinetics_only',
                    input_shape= (79, 224, 224, 3),
                    classes= 400)
    elif args.pretrain == 'bothsets':
        rgb_model = Inception_Inflated3d(
                    include_top= True,
                    weights= 'rgb_imagenet_and_kinetics',
                    input_shape= (79, 224, 224, 3),
                    classes= 400)

    logits = rgb_model.predict(im)

elif args.type == 'optical_flow':
    im = np.load('assets/v_CricketShot_g04_c01_flow.npy')
    if args.pretrain == 'kinetics':
        flow_model = Inception_Inflated3d(
                    include_top= True,
                    weights= 'flow_kinetics_only',
                    input_shape= (79, 224, 224, 2),
                    classes= 400)
    elif args.pretrain == 'bothsets':
        flow_model = Inception_Inflated3d(
                    include_top= True,
                    weights= 'flow_imagenet_and_kinetics',
                    input_shape= (79, 224, 224, 2),
                    classes= 400)

    logits = flow_model.predict(im)                

elif args.type == 'both':
    im1 = np.load('assets/v_CricketShot_g04_c01_rgb.npy')
    im2 = np.load('assets/v_CricketShot_g04_c01_flow.npy')
    if args.pretrain == 'kinetics':
        rgb_model = Inception_Inflated3d(
                    include_top= True,
                    weights= 'rgb_kinetics_only',
                    input_shape= (79, 224, 224, 3),
                    classes= 400)
        flow_model = Inception_Inflated3d(
                    include_top= True,
                    weights= 'flow_kinetics_only',
                    input_shape= (79, 224, 224, 2),
                    classes= 400)
    elif args.pretrain == 'bothsets':
        rgb_model = Inception_Inflated3d(
                    include_top= True,
                    weights= 'rgb_imagenet_and_kinetics',
                    input_shape= (79, 224, 224, 3),
                    classes= 400)
        flow_model = Inception_Inflated3d(
                    include_top= True,
                    weights= 'flow_imagenet_and_kinetics',
                    input_shape= (79, 224, 224, 2),
                    classes= 400)            
    
    logits = rgb_model.predict(im1) + flow_model.predict(im2)

#Evaluation of class-probabilities from the Logits predicted from the Model.

results = [] 
logits = logits[0] 
predictions = np.exp(logits) / np.sum(np.exp(logits))
indices = np.argsort(predictions)[::-1]
print('\nMagnitude of logits: %f' % np.linalg.norm(logits))
print('------------------------------------')
table = PrettyTable(['Probabilities', 'Logits', 'Class'])
for index in indices[:20]:
    results.append([predictions[index], logits[index], classes[index]])
for output in results:    
    table.add_row(output)
print(table)    

#prettytable is used in order to tabulate the class-probabilities, classes and the corresponding logits for better visualisation.
    

