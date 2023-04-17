import random

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from collections import deque

import wandb
from elements import DQPlayer
from game import Game
from src.model import DQNetwork


class DQTrainer:
    '''
    class for training DQPlayer
    '''

    def __init__(self, gamma=1.):
        # self.dq_1 = DQNetwork(240, hidden_size=512)
        # self.dq_1.fc.build([1, 484])
        # self.dq_1.head.build([1, 512])
        # self.dq_s_1 = DQNetwork(240, hidden_size=512)
        # self.dq_s_1.fc.build([1, 484])
        # self.dq_s_1.head.build([1, 512])
        self.model = DQNetwork(240, hidden_size=512)
        self.target_model = DQNetwork(240, hidden_size=512)
        self.model_s = DQNetwork(240, hidden_size=512)
        self.target_model_s = DQNetwork(240, hidden_size=512)
        self.model.fc.build([1, 484])
        self.model.head.build([1, 512])
        self.target_model.fc.build([1, 484])
        self.target_model.head.build([1, 512])
        self.model_s.fc.build([1, 484])
        self.model_s.head.build([1, 512])
        self.target_model_s.fc.build([1, 484])
        self.target_model_s.head.build([1, 512])
        self.gamma = gamma
        self.mse = tf.keras.losses.MeanSquaredError()
        self.optimizer = Adam(0.0003)
        self.optimizer_s = Adam(0.0003)
        self.replay = deque()
        wandb.init(project='reinforcement-learning')

    def load_weights(self, path):
        self.model.load_weights(path)
        self.target_model.load_weights(path)

    def load_weights_s(self, path):
        self.model_s.load_weights(path)
        self.target_model_s.load_weights(path)

    def run_game(self):
        '''
        from 3 to 6 players. Helicarnassus is not yet supported.
        '''
        def get_player(pid):
            return DQPlayer(model=model_dict[pid], explore=True)
        model_dict = {0: None, 'dq': self.model,
                      'dq_s': self.model_s}#, 1: self.dq_1, 2: self.dq_s_1}
        n_other = random.choice([1, 2]), random.choice([1, 2])
        player_ids = list(random.choices(['dq', 'dq_s', 0], k=n_other[0])), \
                     list(random.choices(['dq', 'dq_s', 0], k=n_other[1]))
        players = [DQPlayer(model=model_dict['dq'], explore=False)] + \
                    [get_player(x) for x in player_ids[0]] + \
                    [DQPlayer(model=model_dict['dq_s'], explore=False)] + \
                    [get_player(x) for x in player_ids[1]]
        player_ids = ['dq'] + player_ids[0] + ['dq_s'] + player_ids[1]
        session = Game(players)
        session.setup()
        record = session.run_and_return()
        del session
        del players
        return record, player_ids, n_other

    def calculate_loss(self, batch):
        obs = tf.convert_to_tensor([b[0] for b in batch], dtype='float32')
        action = tf.convert_to_tensor([b[1] for b in batch])
        next_obs = tf.convert_to_tensor([b[2] for b in batch], dtype='float32')
        reward = tf.convert_to_tensor([b[3] for b in batch], dtype='float32')
        reward_s = tf.convert_to_tensor([b[4] for b in batch], dtype='float32')
        done = tf.convert_to_tensor([b[5] for b in batch], dtype='float32')
        mask = tf.convert_to_tensor([b[6] for b in batch])
        next_mask = tf.convert_to_tensor([b[7] for b in batch], dtype='float32')
        x = self.model(obs, mask=mask, training=True)
        x_s = self.model_s(obs, mask=mask, training=True)
        q = tf.gather(x, action, axis=1, batch_dims=1)
        q_s = tf.gather(x_s, action, axis=1, batch_dims=1)
        y = tf.stop_gradient(reward + (1 - done) * tf.convert_to_tensor([self.gamma], dtype='float32') * \
            tf.math.reduce_max(self.target_model(next_obs, mask=next_mask), axis=1))
        y_s = tf.stop_gradient(reward_s + (1 - done) * tf.convert_to_tensor([self.gamma], dtype='float32') * \
            tf.math.reduce_max(self.target_model_s(next_obs, mask=next_mask), axis=1))
        return self.mse(q, y), self.mse(q_s, y_s)

    def gather_batch(self, batch):
        obs = [b[0] for b in batch]
        action = [b[1] for b in batch]
        next_obs = [b[0] for b in batch][1:] + [batch[-1][0]]
        reward_s = [b[3] for b in batch]
        reward = [0 for b in batch[:-1]] + [batch[-1][3]]
        done = [b[4] for b in batch]
        mask = [b[7] for b in batch]
        next_mask = [b[7] for b in batch][1:] + [batch[-1][7]]
        end_points = batch[-1][5]
        end_science = batch[-1][6]
        return list(zip(obs, action, next_obs, reward, reward_s, done, mask, next_mask)), end_points, end_science

    def train_game(self, i):
        if i % 5 == 0 or i < 10:
            data, player_ids, n_other = self.run_game()
            for j in range(len(player_ids)):
                batch = data.pop(0)
                batch, end_points, end_science = self.gather_batch(batch)
                if j == 0 and i % 10 == 0 and i != 0:
                    wandb.log({'dq_points': end_points, 'dq_science': end_science}, step=i)
                if j == (n_other[0] + 1) and i % 10 == 0 and i != 0:
                    wandb.log({'dq_s_points': end_points, 'dq_s_science': end_science}, step=i)
                self.replay += batch
        accu_loss = 0
        accu_loss_s = 0
        for i in range(4):
            with tf.GradientTape(persistent=True) as tape:
                loss, loss_s = self.calculate_loss(random.sample(self.replay, 32))
            accu_loss += loss
            accu_loss_s += loss_s
            grads = tape.gradient(loss, self.model.trainable_weights)
            grads_s = tape.gradient(loss_s, self.model_s.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            self.optimizer_s.apply_gradients(zip(grads_s, self.model_s.trainable_weights))
            del tape

        while len(self.replay) > 4000:
            self.replay.popleft()

        return accu_loss / 4, accu_loss_s / 4

    def sync_networks(self):
        self.model.save_weights('weights/temp')
        self.target_model.load_weights('weights/temp')
        self.model_s.save_weights('weights/temp_s')
        self.target_model_s.load_weights('weights/temp_s')

    def train(self, iterations):
        accu_loss = 0
        accu_loss_s = 0
        for i in range(iterations):
            loss, loss_s = self.train_game(i)
            accu_loss += loss
            accu_loss_s += loss_s
            if i % 10 == 0 and i > 0:
                wandb.log({'dq_loss': accu_loss / 10, 'dq_s_loss': accu_loss_s / 10,
                           'step': i}, step=i)
                accu_loss = 0
                accu_loss_s = 0
            if i % 500 == 0 and i > 0:
                self.sync_networks()
