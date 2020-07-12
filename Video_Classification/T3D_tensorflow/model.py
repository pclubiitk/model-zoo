import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np

#model

def DenseLayer(input, growth_rate, bn_size, drop_rate):
  x = layers.BatchNormalization()(input)
  x = layers.Activation('relu')(x)
  x = layers.Conv3D(filters=bn_size*growth_rate,
                    kernel_size=1,
                    activation=None,
                    padding='same',
                    use_bias=False)(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)
  x = layers.Conv3D(filters=bn_size*growth_rate,
                    kernel_size=3,
                    activation=None,
                    padding='same',
                    use_bias=False)(x)
  if drop_rate > 0:
    x = layers.Dropout(drop_rate)(x)
  return tf.concat([input, x], 4)

def DenseBlock(input, num_layers, bn_size, growth_rate, drop_rate):
  for i in range(num_layers):
    x = DenseLayer(input, growth_rate, bn_size, drop_rate)
  
  return x

def Transition_DenseNet(input, bn_size, growth_rate):
  x = layers.BatchNormalization()(input)
  x = layers.Activation('relu')(x)
  x = layers.Conv3D(filters=bn_size*growth_rate,
                    kernel_size=1,
                    activation=None,
                    padding='same',
                    use_bias=False)(x)
  
  x = layers.AveragePooling3D(pool_size=(2,2,2),
                              strides=(2,2,2))(x)
  return x

def TemporalTransitionLayer(inputs, TTL_config, i):
  x1 = layers.BatchNormalization()(inputs)
  x1 = layers.Activation('relu')(x1)
  x1 = layers.Conv3D(filters=128,
                    kernel_size=(1,1,1),
                    activation=None,
                    padding='same',
                    use_bias=False)(x1)

  x2 = layers.BatchNormalization()(inputs)
  x2 = layers.Activation('relu')(x2)
  x2 = layers.Conv3D(filters=128,
                    kernel_size=(3,3,3),
                    activation=None,
                    padding='same', 
                    use_bias=False)(x2)

  x3 = layers.BatchNormalization()(inputs)
  x3 = layers.Activation('relu')(x3)
  x3 = layers.Conv3D(filters=128,
                    kernel_size=(TTL_config[i],3,3),
                    activation=None,
                    padding='same',
                    use_bias=False)(x3)

  x = tf.concat([x1, x2, x3], 4)
  x = layers.AveragePooling3D(pool_size=(2,2,2),
                              strides=(2,2,2))(x)
  return(x)

def ClassificationLayer(inputs, num_classes):
  x = layers.AveragePooling3D(pool_size=(2,7,7),
                              strides=(2,1,1))(inputs)
  x = layers.Dense(num_classes, activation='softmax')(x)
  return x

def FirstConvolution(inputs):
  x = layers.BatchNormalization()(inputs)
  x = layers.Activation('relu')(x)
  x = layers.Conv3D(filters=16,
                    kernel_size=(3, 7, 7),
                    activation=None,
                    strides=2,
                    padding='same',
                    use_bias=False)(x)
  print('First 3D Conv:', x.shape)
  x = layers.MaxPool3D(pool_size=(3, 3, 3),
                       strides=(1, 2, 2),
                       padding='same')(x)
  return x

def DenseNet3D(inputs, growth_rate, bn_size, drop_rate, Dense_config, num_classes):
    x = FirstConvolution(inputs)
    print('3D Max Pool:', x.shape)

    for i in range(3):
        num_layers = Dense_config[i]

        x = DenseBlock(x, num_layers, bn_size, growth_rate, drop_rate)
        print('DenseBlock {}:'.format(i+1), x.shape)

        x = Transition_DenseNet(x, bn_size, growth_rate)
        print('Transition_DenseNet {}:'.format(i+1), x.shape)
    
    num_layers = Dense_config[3]
    x = DenseBlock(x, num_layers, bn_size, growth_rate, drop_rate)
    print('DenseBlock 4:', x.shape)

    outputs = ClassificationLayer(x, num_classes)
    print('Classification Layer:', outputs.shape)

    return outputs

def T3D(inputs, growth_rate, bn_size, drop_rate, Dense_config, TTL_config, num_classes):
    x = FirstConvolution(inputs)
    print('3D Max Pool:', x.shape)

    for i in range(3):
        num_layers = Dense_config[i]
        
        x = DenseBlock(x, num_layers, bn_size, growth_rate, drop_rate)
        print('DenseBlock {}:'.format(i+1), x.shape)

        x = TemporalTransitionLayer(x, TTL_config, i)
        print('TTL {}:'.format(i+1), x.shape)
    
    num_layers = Dense_config[3]
    x = DenseBlock(x, num_layers, bn_size, growth_rate, drop_rate)
    print('DenseBlock 4:', x.shape)

    outputs = ClassificationLayer(x, num_classes)
    print('Classification Layer:', outputs.shape)

    return outputs

def DenseNet3D_121(inputs):
  num_layers = np.array([6, 12, 24, 16])
  return DenseNet3D(inputs=inputs, growth_rate=32, bn_size=4, drop_rate=0, Dense_config=num_layers, num_classes=101)

def T3D_121(inputs):
  num_layers = np.array([6, 12, 24, 16])
  TTL_config = np.array([6, 4, 4])
  return T3D(inputs=inputs, growth_rate=32, bn_size=4, drop_rate=0, Dense_config=num_layers, TTL_config=TTL_config, num_classes=101)

def T3D_169(inputs):
  num_layers = np.array([6, 12, 32, 32])
  TTL_config = np.array([6, 4, 4])
  return T3D(inputs=inputs, growth_rate=32, bn_size=4, drop_rate=0, Dense_config=num_layers, TTL_config=TTL_config, num_classes=101)
