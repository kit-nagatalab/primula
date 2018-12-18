import csv
from os import makedirs
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Callbacks:
    def __init__(self, p, generator, discriminator, x_train, y_train, z_train, x_test, y_test, z_test, n_batches):
        self.p = p
        self.root = self.p.root + '/' if self.p.root else ''
        self.namespace = self.p.namespace + '/' if self.p.namespace else ''
        self.current = self.root + self.namespace

        self.generator = generator
        self.discriminator = discriminator

        self.styles = x_train
        self.images = y_train
        self.labels = z_train
        self.test_styles = x_test
        self.test_images = y_test
        self.test_labels = z_test

        self.n_batches = n_batches

        self.best = None
        self.g_loss = []
        self.d_loss = []

    def on_train_start(self):
        self.csv_file = open(self.p.files['log'], 'w')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['epoch', 'g_loss', 'd_loss', 'time'])
        self.csv_file.flush()

        self.train_start = datetime.now()

    def on_train_end(self):
        self.csv_file.close()

        self.train_end = datetime.now()
        print('[Total: %s]' % (self.train_end - self.train_start))

    def on_epoch_start(self, epoch):
        self.epoch = epoch
        print('[Epoch: %5d/%5d]' % (self.epoch, self.p.epochs), flush=True)

    def on_epoch_end(self):
        g_loss = np.mean(self.g_loss)
        d_loss = np.mean(self.d_loss)
        time = datetime.now()

        self.sample_images(self.epoch)
        self.model_checkpoint(self.epoch, g_loss, d_loss)

        self.writer.writerow([self.epoch, g_loss, d_loss, time])
        self.csv_file.flush()
        self.g_loss = []
        self.d_loss = []

    def on_batch_start(self, batch_index):
        self.batch_index = batch_index + 1
        self.batch_start = datetime.now()

    def on_batch_end(self, g_loss, d_loss):
        self.batch_end = datetime.now()
        time = (self.batch_end - self.batch_start).total_seconds()
        indent = ' ' * 4
        text = '[Batch: %2d/%2d] [D loss: %6.4f, Acc: %3d%%] [G loss: %6.4f] [Time: %5.2fs] { D: %s, G: %s }'
        params = (self.batch_index, self.n_batches, d_loss[0], 100*d_loss[2], g_loss[0], time, str(d_loss), str(g_loss))
        print(indent + text % params, flush=True)

        self.g_loss.append(g_loss[0])
        self.d_loss.append(d_loss[0])

    def model_checkpoint(self, epoch, g_loss, d_loss):
        if self.best is None or g_loss < self.best or epoch % 100 is 0:
            self.best = g_loss

            def save(name, model):
                d = self.p.dirs['weight'] + name + '/'
                f = 'weights.%05d-%.2f.hdf5' % (epoch, g_loss)
                makedirs(d, exist_ok=True)
                model.save_weights(d + f, overwrite=True)

            save('generator', self.generator)
            save('discriminator', self.discriminator)

    def sample_images(self, epoch):
        d = self.current + 'images/'
        f = '%05d.png' % epoch
        makedirs(d, exist_ok=True)
        rows, cols = 10, 6

        styles, test_styles = self.styles[:rows], self.test_styles[:rows]
        images, test_images = self.images[:rows], self.test_images[:rows]
        labels, test_labels = self.labels[:rows], self.test_labels[:rows]

        fake_A = self.generator.predict([styles, labels])
        fake_B = self.generator.predict([test_styles, test_labels])

        alls = np.concatenate([styles, fake_A, images, test_styles, fake_B, test_images])
        alls = (alls + 1) / 2

        fig, axs = plt.subplots(rows, cols, figsize=(cols, rows))
        index = 0
        for i in range(cols):
            for j in range(rows):
                ax = axs[j, i]
                ax.imshow(alls[index])
                ax.axis('off')
                index += 1
        fig.savefig(d + f)
        plt.close()
