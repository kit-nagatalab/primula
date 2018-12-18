import numpy as np
from .callbacks import Callbacks

class Base():

    def __init__(self, p, x_train, y_train, z_train, x_test, y_test, z_test):
        self.p = p
        self.x_train = x_train
        self.y_train = y_train
        self.z_train = z_train
        self.x_test = x_test
        self.y_test = y_test
        self.z_test = z_test

        self.size = len(x_train)
        self.n_batches = int(self.size / self.p.batch_size)

        self.generator = None
        self.discriminator = None

        self.create_model()

    def create_model(self):
        pass

    def load_batch(self):
        for i in range(self.n_batches-1):
            s = self.p.batch_size
            styles = self.x_train[i*s:(i+1)*s]
            images = self.y_train[i*s:(i+1)*s]
            labels = self.z_train[i*s:(i+1)*s]
            yield styles, images, labels

    def train_on_batch(self, styles, images, labels):
        return None, None

    def train(self, sample_interval=100):
        callbacks = Callbacks(self.p, self.generator, self.discriminator, self.x_train, self.y_train, self.z_train, self.x_test, self.y_test, self.z_test, self.n_batches)

        callbacks.on_train_start()

        for epoch in range(1, self.p.epochs + 1):

            callbacks.on_epoch_start(epoch)

            for batch, (styles, images, labels) in enumerate(self.load_batch()):

                callbacks.on_batch_start(batch)

                g_loss, d_loss = self.train_on_batch(styles, images, labels)

                callbacks.on_batch_end(g_loss, d_loss)

            callbacks.on_epoch_end()

        callbacks.on_train_end()
