import tensorflow as tf
import os
import glob
from tensorflow.python.data.experimental import AUTOTUNE

#downloading the dataset same as used in official paper
def download_data():
    if not os.path.exists("drums"):
        link = 'http://deepyeti.ucsd.edu/cdonahue/wavegan/data/drums.tar.gz'
        print("downloading Dataset from {}".format(link))
        tf.keras.utils.get_file(
                'ds.tar.gz', link, cache_subdir="/content/", extract=True)
        os.system('rm ds.tar.gz')
    else:
        print("dataset already exists")

def load_audio(filename,im_shape):
    f = tf.io.read_file(filename)
    data,_ = tf.audio.decode_wav(f)
    data = tf.reshape(data,im_shape)
    return data

#function to get the data pipe ready
def get_dataset(split='train',batch_size=64,im_shape=(128,128,1)):
    download_data()
    buffer_size = 60000 #as used in google's implementation of DCGAN
    files = glob.glob('drums/'+split+'/*')
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.map(lambda x: load_audio(x,im_shape), num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(buffer_size)
    ds = ds.batch(batch_size)
    return ds