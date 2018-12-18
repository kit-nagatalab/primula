from keras.models import Model
from prism.models import Base


class Combined(Base):
    def create_model(self, style, label, d_valid, d_label, g_image):
        return Model(inputs=[style, label], outputs=[d_valid, d_label, g_image])
