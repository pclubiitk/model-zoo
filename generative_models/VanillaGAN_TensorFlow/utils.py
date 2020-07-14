import matplotlib.pyplot as plt
import glob
import imageio

def plot_generated_images(epoch, generator, noise,outdir):
    examples = 25
    figsize = (5,5)
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples,28,28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(5, 5, i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('{}/img_at_epoch_{:04d}.png'.format(outdir, epoch))
    plt.show()

def make_gif(outdir):
    anim_file = '{}/gan.gif'.format(outdir)
    with imageio.get_writer(anim_file, mode='I') as writer:
      filenames = glob.glob('{}/img*.png'.format(outdir))
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
