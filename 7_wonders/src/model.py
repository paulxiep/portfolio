import numpy as np
from tensorflow.keras.layers import Dense, Multiply
from tensorflow.keras.models import Model


class FullyConnected(Model):
    def __init__(
            self, hidden_size: int = 256
    ):
        super().__init__()
        self.fc0 = Dense(hidden_size * 2, activation='relu')
        self.fc1 = Dense(hidden_size * 2, activation='relu')
        self.fc2 = Dense(hidden_size, activation='relu')

    def call(self, input):
        return self.fc2(self.fc1(self.fc0(input)))


class DQHead(Model):
    def __init__(
            self, space
    ):
        super().__init__()
        self.head = Dense(space)

    def call(self, input):
        return self.head(input)


class DQNetwork(Model):
    def __init__(
            self, action_space, hidden_size: int = 128
    ):
        super().__init__()

        self.fc = FullyConnected(hidden_size=hidden_size)
        self.head = DQHead(action_space)
        self.multiply = Multiply()

    def call(self, obs, training=None, mask=None):
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
            mask = np.expand_dims(mask, 0)
        x = self.head(self.fc(obs))
        return self.multiply([x, mask])

    def save_weights(self, filepath,
                     overwrite=True,
                     save_format=None,
                     options=None):
        self.fc.save_weights(f'{filepath}_fc.h5')
        self.head.save_weights(f'{filepath}_head.h5')

    def load_weights(self, filepath, skip_mismatch=False, by_name=False, options=None):
        self.fc.load_weights(f'{filepath}_fc.h5')
        self.head.load_weights(f'{filepath}_head.h5')
