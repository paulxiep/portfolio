import random

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import wandb
from elements import DQPlayer
from game import Game
from src.model import DQNetwork


class DQTrainer:
    '''
    class for training DQPlayer
    '''

    def __init__(self, gamma=1.):
        self.model = DQNetwork({'choose': 80, 'play': 3}, hidden_size=256)
        self.target_model = DQNetwork({'choose': 80, 'play': 3}, hidden_size=256)
        self.partial_model = {'choose': Sequential([self.model.fc, self.model.head])}
        # 'play': Sequential([self.model.fc, self.model.head['play']])}
        self.partial_model['choose'].build([1, 484])
        self.target_model.fc.build([1, 484])
        self.target_model.head.build([1, 256])
        # self.partial_model['play'].build([1, 484])
        self.gamma = gamma
        self.mse = tf.keras.losses.MeanSquaredError()
        self.optimizer = Adam(0.001)
        wandb.init(project='reinforcement-learning')

    def load_weights(self, path):
        self.model.load_weights(path)
        self.target_model.load_weights(path)
        self.partial_model['choose'].layers[0] = self.model.fc
        # self.partial_model['play'].layers[0] = self.model.fc
        self.partial_model['choose'].layers[1] = self.model.head
        # self.partial_model['play'].layers[1] = self.model.head['play']

    def run_game(self):
        '''
        from 3 to 6 players. Helicarnassus is not yet supported.
        '''
        players = [DQPlayer(model=self.model) for _ in range(random.choice(range(2, 6)))] + [DQPlayer()]
        session = Game(players)
        session.setup()
        chooses = session.run_and_return()
        # print(f'game of {len(players)} simulated!')
        del session
        del players
        return {'choose': chooses}

    def calculate_loss(self, batch, choice, i, j):
        obs = tf.convert_to_tensor([b[0] for b in batch], dtype='float32')
        action = tf.convert_to_tensor([b[1] for b in batch])
        # next_obs = tf.convert_to_tensor([b[2] for b in batch][:-1] + [batch[-2][2]], dtype='float32')
        next_obs = tf.convert_to_tensor([b[0] for b in batch][1:] + [batch[-1][0]], dtype='float32')
        reward = tf.convert_to_tensor([b[3] for b in batch], dtype='float32')
        done = tf.convert_to_tensor([b[4] for b in batch], dtype='float32')
        next_mask = tf.convert_to_tensor([b[6] for b in batch][1:] + [batch[-1][6]], dtype='float32')
        mask = tf.convert_to_tensor([b[6] for b in batch])
        q = tf.gather(self.model(obs, choice, mask=mask, training=True), action, axis=1, batch_dims=1)
        if j == 1 and i % 10 == 0:
            wandb.log({'q -1': q[-1], 'q -6': q[-6], 'q -11': q[-11], 'q -16': q[-16]}, step=i)
        y = reward + (1 - done) * tf.convert_to_tensor([self.gamma], dtype='float32') * \
            tf.math.reduce_max(self.target_model(next_obs, choice, mask=next_mask), axis=1)
        return self.mse(q, y)

    def train_game(self, i):
        data = self.run_game()
        n_players = len(data['choose'])
        accu_loss = 0
        j = 0
        while len(data.keys()) > 0:
            choice = random.choice(list(data.keys()))
            # non_choice = [x for x in ['choose', 'play'] if x != choice][0]
            batch = data[choice].pop(random.randrange(len(data[choice])))
            if len(data[choice]) == 0:
                data.pop(choice)

            with tf.GradientTape() as tape:
                loss = self.calculate_loss(batch, choice, i, j)
                accu_loss += loss
                grads = tape.gradient(loss, self.partial_model[choice].trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.partial_model[choice].trainable_weights))
            self.model.fc = self.partial_model[choice].layers[0]
            # self.partial_model[non_choice].layers[0] = self.partial_model[choice].layers[0]
            self.model.head = self.partial_model[choice].layers[1]
            j += 1
        return accu_loss / n_players

    def sync_networks(self):
        self.model.save_weights('weights/temp')
        self.target_model.load_weights('weights/temp')

    def train(self, iterations):
        accu_loss = 0
        for i in range(iterations):
            loss = self.train_game(i)
            accu_loss += loss
            if i % 10 == 0 and i > 0:
                wandb.log({'loss': accu_loss / 10}, step=i)
                accu_loss = 0
            if i % 200 == 0 and i > 0:
                self.sync_networks()
