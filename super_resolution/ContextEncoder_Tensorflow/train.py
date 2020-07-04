from model import define_generator,define_discriminator,define_gan
from utils import sample_images,mask_randomly
from dataloader import *

import argparse
import tensorflow as tf
import numpy as np
import datetime


###################################################################################
# Parsing all arguments

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate_g', type = float, 
	default = 5e-4, help = 'learning rate for generator')
parser.add_argument('--learning_rate_d', type = float, 
	default = 1e-4, help = 'learning rate for discriminator')
parser.add_argument('--n_epoch', type = int, 
	default = 50, help = 'max number of epoch')
parser.add_argument('--n_update', type = int, 
	default = 50, help = 'max number of iterations to validate model')
parser.add_argument('--batch_size', type = int, 
	default = 64, help = '# of batch size')
parser.add_argument('--num_img', type = int, 
	default = 6, help = '# Number of images to be generated')
parser.add_argument("--lambda_adv",type=float,
	default=0.001,help="Weightage for Adversarial loss")
parser.add_argument('--mask_height', type = int, 
	default = 16, help = 'Masked portion height')
parser.add_argument('--mask_width', type = int, 
	default = 16, help = 'Masked portion width')
parser.add_argument('--samples_dir', type = str, 
	default = './samples/', help = 'directory for sample output')
parser.add_argument('--save_dir', type = str,
	default = './models/', help = 'directory for checkpoint models')

#####################################################################################
# Creating directory for Tensorboard

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

####################################################################################
# Trainig loop

def train(args):
	cnt=0
	for epoch in range(args.n_epoch):
		
		for batch in args.ds:
			cnt+=1
			valid = np.ones((len(batch), 1))
			fake = np.zeros((len(batch), 1))
			
			masked_imgs, masked_parts, _ = mask_randomly(args,batch)
			
			gen_parts = args.gen(masked_imgs)
			d_loss_real = args.dis.train_on_batch(masked_parts,valid)
			d_loss_fake = args.dis.train_on_batch(gen_parts,fake)
			d_loss = 0.5*(d_loss_real + d_loss_fake)
			
			g_loss1 = args.gan.train_on_batch(masked_imgs,valid)
			g_loss2 = args.gen.train_on_batch(masked_imgs,masked_parts)
			g_loss = g_loss1 + g_loss2
			
			with train_summary_writer.as_default():
				tf.summary.scalar("Generator loss",g_loss,step=cnt)
				tf.summary.scalar("Discriminator loss",d_loss,step=cnt)
				tf.summary.scalar("Real Discriminator loss",d_loss_real,step=cnt)
				tf.summary.scalar("Fake Discrminator loss",d_loss_fake,step=cnt)
				tf.summary.scalar("Pixel wise loss",g_loss2,step=cnt)
				tf.summary.scalar("Adverserial loss",g_loss1,step=cnt)

			if cnt%args.n_update==0:
				print('>%d, %d , g1=%0.3f, g2=%0.3f, d1=%.3f, d2=%.3f' %
				(epoch+1, cnt, g_loss1,g_loss2, d_loss_real, d_loss_fake))
				
				sample_images(args, cnt, args.valid_ds)
			
if __name__ == '__main__':

	args = parser.parse_args()
	args.ds,args.valid_ds = get_dataset(args)

	# Initialize Networks
	args.gen = define_generator(in_shape=(args.data_shape))
	args.dis = define_discriminator(in_shape=(args.mask_height,args.mask_width,3))
	args.gan = define_gan(args.gen,args.dis) 
 
	# Initialize Optimizer
	args.gen_opt = tf.keras.optimizers.Adam(args.learning_rate_g)
	args.dis_opt = tf.keras.optimizers.Adam(args.learning_rate_d)
	
	# Customized adversarial function
	def adverserial_loss(y_t,y_p):

		adv_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
		loss = adv_loss(y_t,y_p)
		return args.lambda_adv*loss

	# Initialize Metrics
	args.dis.compile(optimizer=args.dis_opt,loss=tf.keras.losses.binary_crossentropy)
	args.gan.compile(optimizer=args.gen_opt,loss= adverserial_loss)
	args.gen.compile(optimizer=args.gen_opt,loss=tf.keras.losses.mean_squared_error)
   
	train(args)
