from models import Primula
from preferences.primula import Preference
from models.primula import Preprocessor

p = Preference()

preprocessor = Preprocessor(preference=p)
x_train, y_train, z_train, x_test, y_test, z_test = preprocessor.get_train_test()

model = Primula(p, x_train, y_train, z_train, x_test, y_test, z_test)
model.train()
