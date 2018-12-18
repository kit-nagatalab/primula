from pathlib import Path
import numpy as np
from prism.preferences import Base as Preference


class Loader:
    def __init__(self, preference):
        assert isinstance(preference, Preference)
        self.p = preference
        self.cache = Path(self.p.dirs['cache'])
        self.cache_file = self.cache / (self.cache.stem + '.npz')

    def exists(self):
        return self.cache_file.exists()

    def store(self, style, image, meta):
        np.savez(self.cache_file, style=style, image=image, meta=meta)

    def load(self):
        data = np.load(self.cache_file)
        return data['style'], data['image'], data['meta']