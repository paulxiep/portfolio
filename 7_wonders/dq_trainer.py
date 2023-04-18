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
        self.dq_1 = DQNetwork(240, hidden_size=512)
        self.dq_1.fc.build([1, 484])
        self.dq_1.head.build([1, 512])
        self.dq_s_1 = DQNetwork(240, hidden_size=512)
        self.dq_s_1.fc.build([1, 484])
        self.dq_s_1.head.build([1, 512])
        self.dq_c_1 = DQNetwork(240, hidden_size=512)
        self.dq_c_1.fc.build([1, 484])
        self.dq_c_1.head.build([1, 512])
        self.model = DQNetwork(240, hidden_size=512)
        self.target_model = DQNetwork(240, hidden_size=512)
        self.model_s = DQNetwork(240, hidden_size=512)
        self.target_model_s = DQNetwork(240, hidden_size=512)
        self.model_c = DQNetwork(240, hidden_size=512)
        self.target_model_c = DQNetwork(240, hidden_size=512)
        self.model.fc.build([1, 484])
        self.model.head.build([1, 512])
        self.target_model.fc.build([1, 484])
        self.target_model.head.build([1, 512])
        self.model_s.fc.build([1, 484])
        self.model_s.head.build([1, 512])
        self.target_model_s.fc.build([1, 484])
        self.target_model_s.head.build([1, 512])
        self.model_c.fc.build([1, 484])
        self.model_c.head.build([1, 512])
        self.target_model_c.fc.build([1, 484])
        self.target_model_c.head.build([1, 512])
        self.gamma = gamma
        self.mse = tf.keras.losses.MeanSquaredError()
        self.optimizer = Adam(0.00003)
        self.optimizer_s = Adam(0.00003)
        self.optimizer_c = Adam(0.00003)
        self.replay = deque()
        wandb.init(project='reinforcement-learning')

    def load_weights(self, path):
        self.model.load_weights(path)
        self.target_model.load_weights(path)

    def load_weights_s(self, path):
        self.model_s.load_weights(path)
        self.target_model_s.load_weights(path)

    def load_weights_c(self, path):
        self.model_c.load_weights(path)
        self.target_model_c.load_weights(path)

    def run_game(self):
        '''
        from 3 to 6 players. Helicarnassus is not yet supported.
        '''
        def get_player(pid):
            return DQPlayer(model=model_dict[pid])
        model_dict = {0: None, 'dq': self.model, 'dq_s': self.model_s,
                      'dq_c': self.model_c, 1: self.dq_1, 2: self.dq_s_1, 3: self.dq_c_1}
        n_other = random.randrange(2, 6)
        focused_player_id = random.choice(['dq', 'dq_s', 'dq_c'])
        player_ids = list(random.choices(['dq', 'dq_s', 'dq_c', 0, 1, 2, 3], k=n_other))
        players = [DQPlayer(model=model_dict[focused_player_id], explore=False)] + \
                    [get_player(x) for x in player_ids]
        player_ids = [focused_player_id] + player_ids
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
        reward_c = tf.convert_to_tensor([b[5] for b in batch], dtype='float32')
        done = tf.convert_to_tensor([b[6] for b in batch], dtype='float32')
        mask = tf.convert_to_tensor([b[7] for b in batch])
        next_mask = tf.convert_to_tensor([b[8] for b in batch], dtype='float32')
        x = self.model(obs, mask=mask, training=True)
        x_s = self.model_s(obs, mask=mask, training=True)
        x_c = self.model_c(obs, mask=mask, training=True)
        q = tf.gather(x, action, axis=1, batch_dims=1)
        q_s = tf.gather(x_s, action, axis=1, batch_dims=1)
        q_c = tf.gather(x_c, action, axis=1, batch_dims=1)
        y = tf.stop_gradient(reward + (1 - done) * tf.convert_to_tensor([self.gamma], dtype='float32') * \
            tf.math.reduce_max(self.target_model(next_obs, mask=next_mask), axis=1))
        y_s = tf.stop_gradient(reward_s + (1 - done) * tf.convert_to_tensor([self.gamma], dtype='float32') * \
            tf.math.reduce_max(self.target_model_s(next_obs, mask=next_mask), axis=1))
        y_c = tf.stop_gradient(reward_c + (1 - done) * tf.convert_to_tensor([self.gamma], dtype='float32') * \
            tf.math.reduce_max(self.target_model_c(next_obs, mask=next_mask), axis=1))
        return self.mse(q, y), self.mse(q_s, y_s), self.mse(q_c, y_c)

    def gather_batch(self, batch):
        obs = [b[0] for b in batch]
        action = [b[1] for b in batch]
        next_obs = [b[0] for b in batch][1:] + [batch[-1][0]]
        reward_s = [b[3] for b in batch]
        reward = [0 for _ in batch[:-1]] + [batch[-1][3]]
        reward_c = [0 for _ in batch[:-1]] + [batch[-1][3] + batch[-1][7]/2]
        done = [b[4] for b in batch]
        mask = [b[8] for b in batch]
        next_mask = [b[8] for b in batch][1:] + [batch[-1][8]]
        end_points = batch[-1][5]
        end_science = batch[-1][6]
        raw_points = batch[-1][7]
        return list(zip(obs, action, next_obs, reward, reward_s, reward_c, done, mask, next_mask)), end_points, end_science, raw_points

    def train_game(self, i):
        if i % 5 == 0 or i < 10:
            data, player_ids, n_other = self.run_game()
            for j in range(len(player_ids)):
                batch = data.pop(0)
                batch, end_points, end_science, raw_points = self.gather_batch(batch)
                if j == 0 and i % 10 == 0 and i != 0:
                    wandb.log({f'{player_ids[0]}_points': end_points,
                               f'{player_ids[0]}_science': end_science,
                               f'{player_ids[0]}_civilian_wonder': raw_points}, step=i)
                self.replay += batch
        accu_loss = 0
        accu_loss_s = 0
        accu_loss_c = 0
        for i in range(4):
            with tf.GradientTape(persistent=True) as tape:
                loss, loss_s, loss_c = self.calculate_loss(random.sample(self.replay, 32))
            accu_loss += loss
            accu_loss_s += loss_s
            accu_loss_c += loss_c
            grads = tape.gradient(loss, self.model.trainable_weights)
            grads_s = tape.gradient(loss_s, self.model_s.trainable_weights)
            grads_c = tape.gradient(loss_c, self.model_c.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            self.optimizer_s.apply_gradients(zip(grads_s, self.model_s.trainable_weights))
            self.optimizer_c.apply_gradients(zip(grads_c, self.model_c.trainable_weights))
            del tape

        while len(self.replay) > 4000:
            self.replay.popleft()

        return accu_loss / 4, accu_loss_s / 4, accu_loss_c / 4

    def sync_networks(self):
        self.model.save_weights('weights/temp')
        self.target_model.load_weights('weights/temp')
        self.model_s.save_weights('weights/temp_s')
        self.target_model_s.load_weights('weights/temp_s')
        self.model_c.save_weights('weights/temp_c')
        self.target_model_c.load_weights('weights/temp_c')

    def train(self, iterations):
        accu_loss = 0
        accu_loss_s = 0
        accu_loss_c = 0
        for i in range(iterations):
            loss, loss_s, loss_c = self.train_game(i)
            accu_loss += loss
            accu_loss_s += loss_s
            accu_loss_c += loss_c
            if i % 10 == 0 and i > 0:
                wandb.log({'dq_loss': accu_loss / 10, 'dq_s_loss': accu_loss_s / 10,
                            'dq_c_loss': accu_loss_c / 10,
                           'step': i}, step=i)
                accu_loss = 0
                accu_loss_s = 0
                accu_loss_c = 0
            if i % 500 == 0 and i > 0:
                self.sync_networks()
