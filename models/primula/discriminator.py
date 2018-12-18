from keras.layers import Input, Dense, Concatenate, Flatten, multiply, BatchNormalization, Activation, Embedding, LeakyReLU, UpSampling2D, Conv2D
from keras.models import Model
from prism.models import Base

class Discriminator(Base):
    @staticmethod
    def conv2d(x, filters, bn=True):
        x = Conv2D(filters, (3, 3), strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        if bn:
            x = BatchNormalization(momentum=0.8)(x)
        return x

    def create_model(self, filters=64):
        shape = self.p.shape
        n_labels = self.p.n_labels
        n = n_labels + 1

        style = Input(shape=shape)
        image = Input(shape=shape)

        x = Concatenate(axis=-1)([style, image])

        x = Discriminator.conv2d(x, filters*1, bn=False)
        x = Discriminator.conv2d(x, filters*2)
        x = Discriminator.conv2d(x, filters*4)
        x = Discriminator.conv2d(x, filters*8)
        x = Flatten()(x)

        valid = Dense(1, activation='sigmoid', name='valid')(x)
        label = Dense(n, activation='softmax', name='label')(x)

        return Model(inputs=[style, image], outputs=[valid, label], name='discriminator')
