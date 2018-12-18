from tensorflow import device
from keras.layers import Input, Dense, Concatenate, Flatten, multiply, BatchNormalization, Activation, Embedding, LeakyReLU, UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from numpy import ones, zeros, add
from prism.utils.io import write

from .base import Base
from .generator import Generator
from .discriminator import Discriminator
from .combined import Combined


class Primula(Base):
    def create_model(self):
        p = self.p
        shape = self.p.shape

        optimizer = Adam(0.0002, 0.5)
        d_loss = { "valid": "binary_crossentropy", "label": "sparse_categorical_crossentropy" }
        # TODO: FIX
        c_loss = { "model_1_1": "binary_crossentropy", "model_1_2": "sparse_categorical_crossentropy", "model_2": "mae" }

        # Build and compile the discriminator
        discriminator = Discriminator(p)
        self.discriminator = discriminator.get()
        self.discriminator.compile(loss=d_loss, optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        generator = Generator(p)
        self.generator = generator.get()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        style = Input(shape=shape)
        label = Input(shape=(1, ))
        image = self.generator([style, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        d_valid, d_label = self.discriminator([style, image])

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        combined = Combined(p, style=style, label=label, d_valid=d_valid, d_label=d_label, g_image=image)
        self.combined = combined.get()
        self.combined.compile(loss=c_loss, optimizer=optimizer)

        self.save(generator, "generator")
        self.save(discriminator, "discriminator")
        self.save(combined, "combined")

    def save(self, model, name):
        d = self.p.dirs['model']
        model_to_save = model.model()
        write("%s%s.yaml" % (d, name), model_to_save.to_yaml())

    def train_on_batch(self, styles, images, labels):
        n_labels = self.p.n_labels
        batch_size = self.p.batch_size

        # Adversarial ground truths
        valid = ones((batch_size, 1))
        fake = zeros((batch_size, 1))

        fake_images = self.generator.predict([styles, labels])

        # Image labels. 0-8 if image is valid or 9 if it is generated (fake)
        fake_labels = n_labels * ones(labels.shape)

        # Train the discriminator
        d_loss_real = self.discriminator.train_on_batch([styles, images], [valid, labels])
        d_loss_fake = self.discriminator.train_on_batch([styles, fake_images], [fake, fake_labels])
        d_loss = 0.5 * add(d_loss_real, d_loss_fake)

        # Train the generator
        g_loss = self.combined.train_on_batch([styles, labels], [valid, labels, images])

        return g_loss, d_loss
