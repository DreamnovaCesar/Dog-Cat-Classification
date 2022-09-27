
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model

# ? define the standalone discriminator model

def define_discriminator(Input_shape = (224, 224, 3)):
    """
    _summary_

    _extended_summary_

    Args:
        Input_shape (tuple, optional): _description_. Defaults to (224, 224, 3).

    Returns:
        _type_: _description_
    """
    model = Sequential()
    model.add(Conv2D(256, (3, 3), strides = (2, 2), padding = 'same', input_shape = Input_shape))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(128, (3, 3), strides = (2, 2), padding = 'same'))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3, 3), strides = (2, 2), padding = 'same'))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation = 'sigmoid'))

    # compile model

    opt = Adam(lr = 0.0002, beta_1 = 0.5)
    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
    return model

# define model
model = define_discriminator()

# summarize the model
model.summary()