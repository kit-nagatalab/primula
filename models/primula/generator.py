from numpy import prod
from keras.layers import Input, Concatenate, Flatten, multiply, BatchNormalization, Activation, Embedding, Reshape, LeakyReLU, UpSampling2D, Conv2D
from keras.models import Model
from prism.models import Base

class Generator(Base):
    @staticmethod
    def conv2d(x, filters, bn=True):
        x = Conv2D(filters, (3, 3), strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        if bn:
            x = BatchNormalization(momentum=0.8)(x)
        return x

    @staticmethod
    def deconv2d(x, skip_input, filters):
        x = UpSampling2D(2)(x)
        x = Conv2D(filters, (3, 3), strides=1, padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Concatenate()([x, skip_input])
        return x

    @staticmethod
    def embedding(x, label, n_labels):
        shape = x.shape.as_list()[1:]
        count = prod(shape)
        embedding = Embedding(n_labels, count)(label)
        embedding = Reshape(shape)(embedding)
        x = Concatenate()([x, embedding])
        return x

    def create_model(self, filters=16):
        shape = self.p.shape
        n_labels = self.p.n_labels

        style = Input(shape=shape)
        label = Input(shape=(1, ), dtype='int32')

        layers = [None] * 7

        x = style

        x = layers[0] = Generator.conv2d(x, filters * 1, bn=False)
        x = layers[1] = Generator.conv2d(x, filters * 2)
        x = layers[2] = Generator.conv2d(x, filters * 4)
        x = layers[3] = Generator.conv2d(x, filters * 8)
        x = layers[4] = Generator.conv2d(x, filters * 8)
        x = layers[5] = Generator.conv2d(x, filters * 8)
        x = layers[6] = Generator.conv2d(x, filters * 8)

        x = Generator.embedding(x, label, n_labels)

        x = Generator.deconv2d(x, layers[5], filters * 8)
        x = Generator.deconv2d(x, layers[4], filters * 8)
        x = Generator.deconv2d(x, layers[3], filters * 8)
        x = Generator.deconv2d(x, layers[2], filters * 4)
        x = Generator.deconv2d(x, layers[1], filters * 2)
        x = Generator.deconv2d(x, layers[0], filters * 1)

        x = UpSampling2D(2)(x)
        x = Conv2D(3, (3, 3), strides=1, padding='same', activation='tanh', name='image')(x)

        return Model(inputs=[style, label], outputs=x, name='generator')