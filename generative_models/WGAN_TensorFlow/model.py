import tensorflow as tf

class Discriminator(tf.keras.Model):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.conv1 = tf.keras.layers.Conv2D(64, 5, 2, activation = tf.nn.leaky_relu)
		self.dropout1 = tf.keras.layers.Dropout(0.3)
		self.conv2 = tf.keras.layers.Conv2D(128, 5, 2, activation = tf.nn.leaky_relu)
		self.dropout2 = tf.keras.layers.Dropout(0.3)
		self.flatten = tf.keras.layers.Flatten()
		self.fc1 = tf.keras.layers.Dense(1)
		self.batch_norm = tf.keras.layers.BatchNormalization()

	def call(self, x):
		x = self.dropout1(self.conv1(x))
		x = self.dropout2(self.conv2(x))
		x = self.batch_norm(x)
		x = self.flatten(x)
		x = self.fc1(x)
		return x

class Generator(tf.keras.Model):
	def __init__(self):
		super(Generator, self).__init__()

		def _reshape_func(x):
			dims = x.get_shape().as_list()
			return tf.reshape(x, [dims[0], 7, 7, 128])

		self.fc1 = tf.keras.layers.Dense(7 * 7 * 128)
		self.reshape = _reshape_func
		self.conv1 = tf.keras.layers.Conv2DTranspose(64, 5, 2, activation = tf.nn.relu, padding = 'same')
		self.conv2 = tf.keras.layers.Conv2DTranspose(32, 5, 2, activation = tf.nn.relu, padding = 'same')
		self.conv3 = tf.keras.layers.Conv2DTranspose(1, 3, 1, activation = tf.nn.tanh, padding = 'same')
		
	def call(self, x):
		x = self.fc1(x)
		x = self.reshape(x)
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		return x
