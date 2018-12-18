from json import loads
from glob import glob
from pathlib import Path
import numpy as np
from prism.preprocessing import image
from sklearn.model_selection import train_test_split
from prism.preferences import Base as Preference
from .loader import Loader


class Preprocessor:
    def __init__(self, preference, **kwargs):
        assert isinstance(preference, Preference)
        self.p = preference
        assert len(self.p.shape) is 3
        assert self.p.shape[2] in [1, 3]
        self.loader = Loader(preference)
        self.color_mode = 'rgb' if self.p.shape[2] is 3 else 'grayscale'
        self.size = self.p.size
        self.style = kwargs.get('style', self.p.dirs['style'])
        self.image = kwargs.get('image', self.p.dirs['image'])
        self.meta = kwargs.get('meta', self.p.dirs['meta'])

    def load_meta(self):
        meta = Path(self.meta)
        files = sorted(meta.glob("*"))

        files = [open(str(file), encoding='utf-8') for file in files]
        lines = [line for file in files for line in file.readlines()]
        meta = [loads(line) for line in lines]
        return meta

    def get_meta(self):
        labels = self.p.labels

        meta = {}
        for data in self.load_meta():
            image_id = data["id"]
            tags = [tags["name"] for tags in data["tags"]]

            label = [labels.index(tag) for tag in tags if tag in labels]
            assert len(label) is 1
            label = label[0]
            meta[image_id] = label

        images = image.list(self.image)
        images = [Path(image).stem for image in images]

        meta = [meta[image_id] for image_id in images]
        meta = np.array(meta)[:1000]

        return meta

    def get_images(self, dir):
        images = image.list(dir)[:1000]
        images = image.load(images, color_mode=self.color_mode, target_size=self.size)
        return images

    @staticmethod
    def debug(message):
        from datetime import datetime

        print("%s:" % datetime.today(), message, flush=True)

    def get_train_test(self):
        X = None
        Y = None
        Z = None

        Preprocessor.debug("Load start")

        if self.loader.exists():
            X, Y, Z = self.loader.load()
        else:
            X = self.get_images(self.style)
            Y = self.get_images(self.image)
            Z = self.get_meta()
            self.loader.store(X, Y, Z)

        # X = self.get_images(self.style)
        # Y = self.get_images(self.image)
        # Z = self.get_meta()

        Preprocessor.debug("Load end")

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        x_train, x_test, z_train, z_test = train_test_split(X, Z, test_size=0.2, random_state=0)

        x_train = (x_train - 0.5) * 2
        x_test = (x_test - 0.5) * 2
        y_train = (y_train - 0.5) * 2
        y_test = (y_test - 0.5) * 2

        return x_train, y_train, z_train, x_test, y_test, z_test
