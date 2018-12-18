from prism.preferences import Base
from keras.optimizers import Adam

class Preference(Base):
    epochs = 100000
    batch_size = 16

    width = 128
    height = 128
    size = (height, width)
    shape = (height, width, 3)

    labels = ["blonde_hair", "brown_hair", "black_hair", "blue_hair", "purple_hair", "pink_hair", "silver_hair", "green_hair", "red_hair"]
    n_labels = len(labels)

    dirs = {
        'cache': '/root/Datasets/danbooru2017/wbg_cfg_wpd_safe_smhc_slc/',
        'style': '/root/Datasets/danbooru2017/wbg_cfg_wpd_safe_smhc_slc/styles',
        'image': '/root/Datasets/danbooru2017/wbg_cfg_wpd_safe_smhc_slc/512px',
        'meta': '/root/Datasets/danbooru2017/wbg_cfg_wpd_safe_smhc_slc/metadata'
    }

    root = 'experiments'

    namespace = 'primula-100000'
    description = 'GANで線画を着色する'
