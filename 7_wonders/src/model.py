# import gymnasium
# from gymnasium.spaces.utils import flatten_space
import numpy as np
from tensorflow.keras.layers import Dense, Multiply
from tensorflow.keras.models import Model


# from tensorflow import divide, reduce_sum

class FullyConnected(Model):
    def __init__(
            self, hidden_size: int = 256
    ):
        super().__init__()
        self.fc0 = Dense(hidden_size * 4, activation='relu')
        self.fc1 = Dense(hidden_size * 2, activation='relu')
        self.fc2 = Dense(hidden_size, activation='relu')
        # self.fc3 = Dense(hidden_size // 2, activation='relu')

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
        self.head = DQHead(action_space['choose'])

    def call(self, obs, head, training=None, mask=None):
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
            mask = np.expand_dims(mask, 0)
        x = self.fc(obs)
        return Multiply()([self.head(x), mask])

    def save_weights(self, filepath,
                     overwrite=True,
                     save_format=None,
                     options=None):
        self.fc.save_weights(f'{filepath}_fc.h5')
        self.head.save_weights(f'{filepath}_dq_choose.h5')
        # self.head['play'].save_weights(f'{filepath}_dq_play.h5')

    def load_weights(self, filepath, skip_mismatch=False, by_name=False, options=None):
        self.fc.load_weights(f'{filepath}_fc.h5')
        self.head.load_weights(f'{filepath}_dq_choose.h5')
        # self.head['play'].load_weights(f'{filepath}_dq_play.h5')

# class PGNetwork(Model):
#     def __init__(
#         self, action_space: gymnasium.Space, hidden_size: int = 128
#     ):
#         super().__init__()
#
#         self.fc1 = Dense(hidden_size, activation='relu')
#         self.fc2 = Dense(hidden_size, activation='relu')
#         self.do = Dropout(0.2)
#         self.head = {'choose': Dense(flatten_space(action_space['choose']).shape[0], activation='softmax'),
#                      'play': Dense(flatten_space(action_space['play']).shape[0], activation='softmax'),
#                      'pay': Dense(flatten_space(action_space['pay']).shape[0])}
#
#     def call(self, obs, head, mask=None):
#         def scale(tensor):
#             return divide(tensor, reduce_sum(tensor))
#         if len(obs.shape) == 1:
#             obs = np.expand_dims(obs, 0)
#             mask = np.expand_dims(mask, 0)
#         x = self.fc2(self.fc1(obs))
#         if head == 'pay':
#             return self.head[head](x)
#         else:
#             return scale(Multiply()([self.head[head](x), mask]))
