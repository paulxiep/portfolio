import gymnasium
from gymnasium.spaces.utils import flatten_space
from tensorflow.keras.layers import Dense, Dropout, Input, Activation, Multiply
from tensorflow.keras.models import Model
import numpy as np
from tensorflow import divide, reduce_sum

class DeepQNetwork(Model):
    def __init__(
        self, action_space: gymnasium.Space, hidden_size: int = 128
    ):
        super().__init__()

        self.fc1 = Dense(hidden_size, activation='relu')
        self.fc2 = Dense(hidden_size, activation='relu')
        self.do = Dropout(0.2)
        self.head = {'choose': Dense(flatten_space(action_space['choose']).shape[0], activation='softmax'),
                     'play': Dense(flatten_space(action_space['play']).shape[0], activation='softmax'),
                     'pay': Dense(flatten_space(action_space['pay']).shape[0])}

    def call(self, obs, head, mask=None):
        def scale(tensor):
            return divide(tensor, reduce_sum(tensor))

        x = self.fc2(self.fc1(np.expand_dims(obs, 0)))
        if head == 'pay':
            return self.head[head](x)
        else:
            return scale(Multiply()([self.head[head](x), np.expand_dims(mask, 0)]))
