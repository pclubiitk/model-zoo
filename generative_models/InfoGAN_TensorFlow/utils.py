import tensorflow as tf
import numpy as np
import glob
import imageio
import matplotlib.pyplot as plt
import os
from copy import deepcopy

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

###################################

def generate_latent_points(num_examples_to_generate, noise_dim, categorical_dim, continuous_dim):
	# generate points in the latent space
	z_latent = tf.random.normal([num_examples_to_generate, noise_dim])
	cat_codes = np.random.randint(0, categorical_dim, num_examples_to_generate)
	cat_codes = tf.keras.utils.to_categorical(cat_codes, num_classes=categorical_dim)
	cont_codes = tf.random.uniform(shape = [num_examples_to_generate, continuous_dim], minval = -1, maxval = 1)
	z_input = tf.concat([z_latent, cat_codes, cont_codes], axis = 1)
	return [z_input, cat_codes, cont_codes]
	
###########################################
	
def generate_and_save_images(model, j, test_input, outdir,  dataset = "MNIST", samples = False):
  predictions = model(test_input, training=False)
  fig = plt.figure(figsize=(10,10))
  for i in range(predictions.shape[0]):
      plt.subplot(10, 10, i+1)
      if dataset == 'MNIST':
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      else:
        plt.imshow((predictions[i, :, :, :] + 1)/2)
      plt.axis('off')
  if samples == False:
    plt.savefig('{}/assets/{}/{}_at_epoch_{:04d}.png'.format(outdir, dataset, dataset, j))
  else:
    plt.savefig(f"{outdir}/assets/{dataset}/{dataset}_cont_dim_{j}.png")
  plt.show()
  
#########################################

def save_gif(outdir, dataset = "MNIST"):
    anim_file = f'{outdir}/assets/{dataset}/{dataset}.gif'
    with imageio.get_writer(anim_file, mode='I') as writer:
      filenames = glob.glob(f'{outdir}/assets/{dataset}/{dataset}*.png')
      filenames = sorted(filenames)
      last = -1
      for i,filename in enumerate(filenames):
        frame = 2*i
        if round(frame) > round(last):
          last = frame
        else:
          continue
        image = imageio.imread(filename)
        writer.append_data(image)
      image = imageio.imread(filename)
      writer.append_data(image)

#########################################
# For producing outputs with constant noise and varying continuous and categorical latent codes

def generate_varying_outputs(generator, num_examples_to_generate, noise_dim, dataset, outdir, categorical_dim = 10, continuous_dim = 2):
    noise = tf.random.normal([1, noise_dim])
    noise = tf.repeat(noise, num_examples_to_generate, 0) #constant noise
    
    categorical_samples= np.arange(categorical_dim)
    cat_codes = tf.keras.utils.to_categorical(categorical_samples, num_classes=categorical_dim)
    cat_codes = tf.repeat(cat_codes, num_examples_to_generate//categorical_dim, 0) #equal samples from each categorical code
    cont_temp = tf.zeros([num_examples_to_generate, continuous_dim])
    without_cont = tf.concat([noise, cat_codes, cont_temp], axis = 1)
    without_cont = np.asarray(without_cont)
    
    values = np.linspace(-1,1,num_examples_to_generate//categorical_dim)
    values = np.asarray(values)
    res = [values for i in range(categorical_dim)]
    res = np.reshape(res, (num_examples_to_generate,))
    
    for i in range(continuous_dim):
      sample = deepcopy(without_cont)
      sample[:,noise_dim + categorical_dim + i] = res
      sample = tf.convert_to_tensor(sample)
      generate_and_save_images(generator, i, sample, outdir, dataset, samples = True)
                                 

#########################################

def clear(): 
    # for windows 
    if os.name == 'nt': 
        _ = os.system('cls') 
    # for mac and linux 
    else: 
        _ = os.system('clear') 
