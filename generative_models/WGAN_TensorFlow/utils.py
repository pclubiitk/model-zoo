import matplotlib.pyplot as plt

def generate_and_save_images(epoch, predictions):
  fig = plt.figure(figsize=(8,8))

  for i in range(predictions.shape[0]):
      plt.subplot(8, 8, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:03d}.png'.format(epoch))
  plt.show()