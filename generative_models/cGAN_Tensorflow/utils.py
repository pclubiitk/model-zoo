import tensorflow as tf
import matplotlib.pyplot as plt
import os
import imageio
import math

def plot_images(generator, noise_input, noise_class, outdir, show=False, epoch=1):
    os.makedirs(outdir+'Image/', exist_ok=True)
    images = generator.predict([noise_input, noise_class])
    print("Labels for generated images: ", np.argmax(noise_class, axis=1))
    plt.figure(figsize=(10, 10))
    num_images = images.shape[0]
    image_size = images.shape[1]
    rows = int(math.sqrt(noise_input.shape[0]))
    for i in range(num_images):
        plt.subplot(rows, rows, i + 1)
        image = np.reshape(images[i], [image_size, image_size])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.savefig(outdir+'Image/'+str(epoch)+'.png')
    if show:
        plt.show()
    else:
        plt.close('all')

def make_gif(outdir, model_name, epochs):
    image = []
    for i in range(1,epochs+1):
      image.append(imageio.imread(outdir+'Image/'+str(i)+'.png'))
    imageio.mimsave(model_name+'.gif', image, fps=5)
