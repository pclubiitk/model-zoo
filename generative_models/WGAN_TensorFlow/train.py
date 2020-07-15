from model import *
from utils import *
from dataloader import *

import argparse
import tensorflow as tf
import numpy as np
import datetime


###################################################################################
# Parsing all arguments

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, choices = ['mnist', 'cifar-10'], 
	default = 'mnist', help = 'choice of dataset')
parser.add_argument('--learning_rate', type = float, 
	default = 5e-4, help = 'initial learning rate')
parser.add_argument('--n_clip', type = float, 
	default = 1e-2, help = 'Cliping weight')
parser.add_argument('--n_epoch', type = int, 
	default = 50, help = 'max # of epoch')
parser.add_argument('--n_update_dis', type = int, 
	default = 5, help = '# of updates of discriminator per update of generator')
parser.add_argument('--noise_dim', type = int, 
	default = 32, help = 'dimension of random noise')
parser.add_argument('--batch_size', type = int, 
	default = 64, help = '# of batch size')
parser.add_argument('--samples_dir', type = str, 
	default = './samples/', help = 'directory for sample output')
parser.add_argument('--save_dir', type = str,
	default = './models/', help = 'directory for checkpoint models')

#####################################################################################
# Creating directory for Tensorboard

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

#######################################################################################
# Training Steps

def train_step_gen(args):
	with tf.GradientTape() as tape:
		z = tf.random.uniform([args.batch_size, args.noise_dim], -1.0, 1.0)
		fake_sample = args.gen(z)
		fake_score = args.dis(fake_sample)
		loss = - tf.reduce_mean(fake_score)
	gradients = tape.gradient(loss, args.gen.trainable_variables)
	args.gen_opt.apply_gradients(zip(gradients, args.gen.trainable_variables))
	args.gen_loss(loss)

def train_step_dis(args, real_sample):
	batch_size = real_sample.get_shape().as_list()[0]
	with tf.GradientTape() as tape:
		z = tf.random.uniform([batch_size, args.noise_dim], -1.0, 1.0)
		fake_sample = args.gen(z)
		real_score = args.dis(real_sample)
		fake_score = args.dis(fake_sample)
		alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
		inter_sample = fake_sample * alpha + real_sample * (1 - alpha)
		with tf.GradientTape() as tape_gp:
			tape_gp.watch(inter_sample)
			inter_score = args.dis(inter_sample)
		gp_gradients = tape_gp.gradient(inter_score, inter_sample)
		gp_gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gp_gradients), axis = [1, 2, 3]))
		gp = tf.reduce_mean((gp_gradients_norm - 1.0) ** 2)

		loss = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score) + gp * 20

	gradients = tape.gradient(loss, args.dis.trainable_variables)
	args.dis_opt.apply_gradients(zip(gradients, args.dis.trainable_variables))
		
	args.dis_loss(loss)

def test_step(args, epoch):
	z = tf.random.uniform([args.batch_size, args.noise_dim], -1.0, 1.0)
	fake_sample = args.gen(z)
	generate_and_save_images(epoch,fake_sample.numpy())

####################################################################################
# Trainig loop

def train(args):
    for epoch in range(args.n_epoch):
        cnt = 0
        for batch in args.ds:
            cnt += 1
            if cnt % (args.n_update_dis + 1) > 0:
                train_step_dis(args, batch)
            else:
                train_step_gen(args)

        template = 'Epoch {}, Gen Loss: {}, Dis Loss: {}'
        print (template.format(epoch + 1, args.gen_loss.result(), 
                    args.dis_loss.result()))
        
        with train_summary_writer.as_default():
            tf.summary.scalar("Discriminator loss",args.dis_loss.result(),step=epoch)
            tf.summary.scalar("Generator loss",args.gen_loss.result(),step=epoch)
        test_step(args, epoch)
        args.dis_loss.reset_states()
        args.gen_loss.reset_states()

if __name__ == '__main__':

	args = parser.parse_args()
	args.ds = get_dataset(args)

	# Initialize Networks
	args.gen = Generator()
	args.dis = Discriminator()

	# Initialize Optimizer
	args.gen_opt = tf.keras.optimizers.RMSprop(args.learning_rate)
	args.dis_opt = tf.keras.optimizers.RMSprop(args.learning_rate)

	# Initialize Metrics
	args.gen_loss = tf.keras.metrics.Mean(name = 'Generator_Loss')
	args.dis_loss = tf.keras.metrics.Mean(name = 'Discriminator_Loss')
     
	train(args)
