import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from collections import deque
from functools import reduce

import wandb
from elements import DQPlayer
from game import Game
from src.model import DQNetwork


class DQTrainer:
    '''
    class for training DQPlayer
    main function is Deep-Q training
    contains minor Monte-Carlo loss update feature as commented out section
    supports distributed training, specify the devices in __init__ section
    '''

    def __init__(self, gamma=1., distributed=False):
        if not distributed:
            self.model_init()
            self.mse = tf.keras.losses.MeanSquaredError()
        else:
            self.mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/cpu:0"])
            with self.mirrored_strategy.scope():
                self.model_init()
                self.mse = tf.keras.losses.MeanSquaredError(reduction='none')
        '''
        monte-carlo feature was not integrated with distributed training
        '''
        # self.mc_optimizer = Adam(0.00003)
        # self.mc_optimizer_s = Adam(0.00003)
        # self.mc_optimizer_c = Adam(0.00003)
        # self.mc_range = range(2, 7)
        self.distributed = distributed
        self.gamma = gamma
        self.replay = deque()
        self.model_dict = {0: None, 'dq': self.model, 'dq_s': self.model_s,
                      'dq_c': self.model_c, #1: self.dq_1, 2: self.dq_s_1, 3: self.dq_c_1,
                      # 4: self.dq_2, 5: self.dq_s_2, 6: self.dq_c_2,
                      7: self.dq_3, 8: self.dq_s_3, 9: self.dq_c_3,
                      10: self.dq_4, 11: self.dq_s_4, 12: self.dq_c_4,
                           13: self.dq_5, 14: self.dq_s_5, 15: self.dq_c_5,}

        wandb.init(project='reinforcement-learning')

    def model_init(self):
        # self.dq_1 = DQNetwork(240, hidden_size=512)
        # self.dq_1.fc.build([1, 380])
        # self.dq_1.head.build([1, 512])
        # self.dq_s_1 = DQNetwork(240, hidden_size=512)
        # self.dq_s_1.fc.build([1, 380])
        # self.dq_s_1.head.build([1, 512])
        # self.dq_c_1 = DQNetwork(240, hidden_size=512)
        # self.dq_c_1.fc.build([1, 380])
        # self.dq_c_1.head.build([1, 512])
        # self.dq_2 = DQNetwork(240, hidden_size=512)
        # self.dq_2.fc.build([1, 380])
        # self.dq_2.head.build([1, 512])
        # self.dq_s_2 = DQNetwork(240, hidden_size=512)
        # self.dq_s_2.fc.build([1, 380])
        # self.dq_s_2.head.build([1, 512])
        # self.dq_c_2 = DQNetwork(240, hidden_size=512)
        # self.dq_c_2.fc.build([1, 380])
        # self.dq_c_2.head.build([1, 512])
        self.dq_3 = DQNetwork(240, hidden_size=512)
        self.dq_3.fc.build([1, 380])
        self.dq_3.head.build([1, 512])
        self.dq_s_3 = DQNetwork(240, hidden_size=512)
        self.dq_s_3.fc.build([1, 380])
        self.dq_s_3.head.build([1, 512])
        self.dq_c_3 = DQNetwork(240, hidden_size=512)
        self.dq_c_3.fc.build([1, 380])
        self.dq_c_3.head.build([1, 512])
        self.dq_4 = DQNetwork(240, hidden_size=512)
        self.dq_4.fc.build([1, 380])
        self.dq_4.head.build([1, 512])
        self.dq_s_4 = DQNetwork(240, hidden_size=512)
        self.dq_s_4.fc.build([1, 380])
        self.dq_s_4.head.build([1, 512])
        self.dq_c_4 = DQNetwork(240, hidden_size=512)
        self.dq_c_4.fc.build([1, 380])
        self.dq_c_4.head.build([1, 512])
        self.dq_5 = DQNetwork(240, hidden_size=512)
        self.dq_5.fc.build([1, 380])
        self.dq_5.head.build([1, 512])
        self.dq_s_5 = DQNetwork(240, hidden_size=512)
        self.dq_s_5.fc.build([1, 380])
        self.dq_s_5.head.build([1, 512])
        self.dq_c_5 = DQNetwork(240, hidden_size=512)
        self.dq_c_5.fc.build([1, 380])
        self.dq_c_5.head.build([1, 512])
        self.model = DQNetwork(240, hidden_size=512)
        self.target_model = DQNetwork(240, hidden_size=512)
        self.model_s = DQNetwork(240, hidden_size=512)
        self.target_model_s = DQNetwork(240, hidden_size=512)
        self.model_c = DQNetwork(240, hidden_size=512)
        self.target_model_c = DQNetwork(240, hidden_size=512)
        self.model.fc.build([1, 380])
        self.model.head.build([1, 512])
        self.target_model.fc.build([1, 380])
        self.target_model.head.build([1, 512])
        self.model_s.fc.build([1, 380])
        self.model_s.head.build([1, 512])
        self.target_model_s.fc.build([1, 380])
        self.target_model_s.head.build([1, 512])
        self.model_c.fc.build([1, 380])
        self.model_c.head.build([1, 512])
        self.target_model_c.fc.build([1, 380])
        self.target_model_c.head.build([1, 512])
        self.dq_optimizer = Adam(0.00001)
        self.dq_optimizer_s = Adam(0.00001)
        self.dq_optimizer_c = Adam(0.00001)

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
        def get_player(pid):
            return DQPlayer(model=self.model_dict[pid])
        n_other = random.randrange(2, 7)
        focused_player_id = random.choice(['dq', 'dq_s', 'dq_c'])
        player_ids = list(random.choices(['dq', 'dq_s', 'dq_c', 0, 7, 8, 9, 10, 11, 12, 13, 14, 15], k=n_other))
        players = [DQPlayer(model=self.model_dict[focused_player_id], explore=False)] + \
                    [get_player(x) for x in player_ids]
        player_ids = [focused_player_id] + player_ids
        session = Game(players)
        session.setup()
        record = session.run_and_return()
        del session
        del players
        return record, player_ids, n_other

    def dq_loss(self, batch):
        obs = tf.convert_to_tensor([b[0] for b in batch], dtype='float32')
        action = tf.convert_to_tensor([b[1] for b in batch])
        next_obs = tf.convert_to_tensor([b[2] for b in batch], dtype='float32')
        reward = tf.convert_to_tensor([b[3] for b in batch], dtype='float32')
        reward_s = tf.convert_to_tensor([b[4] for b in batch], dtype='float32')
        reward_c = tf.convert_to_tensor([b[5] for b in batch], dtype='float32')
        done = tf.convert_to_tensor([b[6] for b in batch], dtype='float32')
        mask = tf.convert_to_tensor([b[7] for b in batch], dtype='float32')
        next_mask = tf.convert_to_tensor([b[8] for b in batch], dtype='float32')
        x = tf.math.multiply(mask, self.model(obs))
        x_s = tf.math.multiply(mask, self.model_s(obs))
        x_c = tf.math.multiply(mask, self.model_c(obs))
        q = tf.gather(x, action, axis=1, batch_dims=1)
        q_s = tf.gather(x_s, action, axis=1, batch_dims=1)
        q_c = tf.gather(x_c, action, axis=1, batch_dims=1)
        y = tf.stop_gradient(reward + (1 - done) * tf.convert_to_tensor([self.gamma], dtype='float32') * \
            tf.math.reduce_max(tf.math.multiply(next_mask, self.target_model(next_obs)), axis=1))
        y_s = tf.stop_gradient(reward_s + (1 - done) * tf.convert_to_tensor([self.gamma], dtype='float32') * \
            tf.math.reduce_max(tf.math.multiply(next_mask, self.target_model_s(next_obs)), axis=1))
        y_c = tf.stop_gradient(reward_c + (1 - done) * tf.convert_to_tensor([self.gamma], dtype='float32') * \
            tf.math.reduce_max(tf.math.multiply(next_mask, self.target_model_c(next_obs)), axis=1))
        return self.mse(q, y), self.mse(q_s, y_s), self.mse(q_c, y_c)

    def loss_distributed(self, *args):
        obs, action, next_obs, reward, reward_s, reward_c, done, mask, next_mask = tuple([*args])
        x = tf.math.multiply(mask, self.model(obs))
        x_s = tf.math.multiply(mask, self.model_s(obs))
        x_c = tf.math.multiply(mask, self.model_c(obs))
        q = tf.gather(x, tf.cast(action, tf.int32), axis=1, batch_dims=1)
        q_s = tf.gather(x_s, tf.cast(action, tf.int32), axis=1, batch_dims=1)
        q_c = tf.gather(x_c, tf.cast(action, tf.int32), axis=1, batch_dims=1)
        y = tf.stop_gradient(reward + (1 - done) * tf.convert_to_tensor([self.gamma], dtype='float32') * \
                             tf.math.reduce_max(tf.math.multiply(next_mask, self.target_model(next_obs)), axis=1))
        y_s = tf.stop_gradient(reward_s + (1 - done) * tf.convert_to_tensor([self.gamma], dtype='float32') * \
                               tf.math.reduce_max(tf.math.multiply(next_mask, self.target_model_s(next_obs)), axis=1))
        y_c = tf.stop_gradient(reward_c + (1 - done) * tf.convert_to_tensor([self.gamma], dtype='float32') * \
                               tf.math.reduce_max(tf.math.multiply(next_mask, self.target_model_c(next_obs)), axis=1))
        return self.mse(q, y), self.mse(q_s, y_s), self.mse(q_c, y_c)

    def mc_loss(self, mc_batch, player_id):
        obs = tf.convert_to_tensor([mc_batch[0][0]], dtype='float32')
        action = tf.convert_to_tensor([mc_batch[0][1]])
        mask = tf.convert_to_tensor([mc_batch[0][7]], dtype='float32')
        x = tf.math.multiply(mask, self.model_dict[player_id](obs))
        v = tf.gather(x, action, axis=1, batch_dims=1)
        if player_id == 'dq':
            reward = tf.convert_to_tensor([mc_batch[-1][3]], dtype='float32')
        elif player_id == 'dq_c':
            reward = tf.convert_to_tensor([mc_batch[-1][5]], dtype='float32')
        elif player_id == 'dq_s':
            reward = tf.convert_to_tensor([reduce(lambda x, y: x + y[4], mc_batch, 0)], dtype='float32')
        return self.mse(v, reward)

    def gather_batch(self, batch):
        obs = [b[0] for b in batch]
        action = [b[1] for b in batch]
        next_obs = [b[0] for b in batch][1:] + [batch[-1][0]]
        reward_s = [b[3] for b in batch]
        reward = [0 for _ in batch[:-1]] + [batch[-1][5]]
        reward_c = [0 for _ in batch[:-1]] + [batch[-1][5] + batch[-1][7]/2]
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
                '''
                below commented section was for Monte-Carlo loss
                '''
                # if j == 0 and i % 5 == 0:
                #     for mc_depth in random.sample(self.mc_range, k=2):
                #         mc_batch = batch[-mc_depth:]
                #         with tf.GradientTape() as tape:
                #             mc_loss = self.mc_loss(mc_batch, player_ids[j])
                #             mc_grads = tape.gradient(mc_loss, self.model_dict[player_ids[j]].trainable_weights)
                #             if 's' in player_ids[j]:
                #                 self.mc_optimizer_s.apply_gradients(zip(mc_grads, self.model_s.trainable_weights))
                #             elif 'c' in player_ids[j]:
                #                 self.mc_optimizer_c.apply_gradients(zip(mc_grads, self.model_c.trainable_weights))
                #             else:
                #                 self.mc_optimizer.apply_gradients(zip(mc_grads, self.model.trainable_weights))
                #     if i % 50 == 0 and i != 0:
                #         loss_name = 'mc_' + player_ids[j][-1] if '_' in player_ids[j] else 'mc'
                #         wandb.log({f'{loss_name}_loss': mc_loss})

                if j == 0 and i % 50 == 0 and i != 0:
                    wandb.log({f'{player_ids[0]}_points': end_points,
                               f'{player_ids[0]}_science': end_science,
                               f'{player_ids[0]}_civilian_wonder': raw_points}, step=i)
                self.replay += batch
        accu_loss = 0
        accu_loss_s = 0
        accu_loss_c = 0
        for i in range(2):
            with tf.GradientTape(persistent=True) as tape:
                loss, loss_s, loss_c = self.dq_loss(random.sample(self.replay, 64))
            accu_loss += loss
            accu_loss_s += loss_s
            accu_loss_c += loss_c
            grads = tape.gradient(loss, self.model.trainable_weights)
            grads_s = tape.gradient(loss_s, self.model_s.trainable_weights)
            grads_c = tape.gradient(loss_c, self.model_c.trainable_weights)
            self.dq_optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            self.dq_optimizer_s.apply_gradients(zip(grads_s, self.model_s.trainable_weights))
            self.dq_optimizer_c.apply_gradients(zip(grads_c, self.model_c.trainable_weights))
            del tape

        while len(self.replay) > 4000:
            self.replay.popleft()

        return accu_loss / 2, accu_loss_s / 2, accu_loss_c / 2

    def train_game_distributed(self, dataset):
        obs, action, next_obs, reward, reward_s, reward_c, done, mask, next_mask = dataset

        with tf.GradientTape(persistent=True) as tape:
            loss, loss_s, loss_c = self.loss_distributed(obs, action, next_obs, reward, reward_s, reward_c, done, mask, next_mask)
        grads = tape.gradient(loss, self.model.trainable_weights)
        grads_s = tape.gradient(loss_s, self.model_s.trainable_weights)
        grads_c = tape.gradient(loss_c, self.model_c.trainable_weights)
        self.dq_optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.dq_optimizer_s.apply_gradients(zip(grads_s, self.model_s.trainable_weights))
        self.dq_optimizer_c.apply_gradients(zip(grads_c, self.model_c.trainable_weights))
        del tape

        return loss, loss_s, loss_c


    def sync_networks(self):
        self.model.save_weights('weights/temp')
        self.target_model.load_weights('weights/temp')
        self.model_s.save_weights('weights/temp_s')
        self.target_model_s.load_weights('weights/temp_s')
        self.model_c.save_weights('weights/temp_c')
        self.target_model_c.load_weights('weights/temp_c')

    def train_single(self, iterations):
        accu_loss = 0
        accu_loss_s = 0
        accu_loss_c = 0
        for i in range(iterations):
            loss, loss_s, loss_c = self.train_game(i)
            accu_loss += loss
            accu_loss_s += loss_s
            accu_loss_c += loss_c
            if i % 100 == 0 and i > 0:
                wandb.log({'dq_loss': accu_loss / 100, 'dq_s_loss': accu_loss_s / 100,
                            'dq_c_loss': accu_loss_c / 100,
                           'step': i}, step=i)
                accu_loss = 0
                accu_loss_s = 0
                accu_loss_c = 0
            if i % 500 == 0 and i > 0:
                self.sync_networks()

    @tf.function
    def train_distributed_step(self, x):
        loss, loss_s, loss_c = \
            self.mirrored_strategy.run(self.train_game_distributed, args=(x,))

        return self.mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None),\
                self.mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, loss_s, axis=None),\
                self.mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, loss_c, axis=None)

    def train_distributed(self, iterations):
        for i in range(10):
            data, player_ids, n_other = self.run_game()
            for j in range(len(player_ids)):
                batch = data.pop(0)
                batch, end_points, end_science, raw_points = self.gather_batch(batch)
                self.replay += batch

        for i in range(iterations):
            if i % 5 == 0:
                data, player_ids, n_other = self.run_game()
                for j in range(len(player_ids)):
                    batch = data.pop(0)
                    batch, end_points, end_science, raw_points = self.gather_batch(batch)
                    if j == 0 and i % 50 == 0:
                        wandb.log({f'{player_ids[0]}_points': end_points,
                                                  f'{player_ids[0]}_science': end_science,
                                                  f'{player_ids[0]}_civilian_wonder': raw_points}, step=i)
                    self.replay += batch
                while len(self.replay) > 4000:
                    self.replay.popleft()

            distributed_dataset = self.mirrored_strategy.experimental_distribute_dataset(tf.data.Dataset.from_tensor_slices(tuple(map(lambda z: tf.convert_to_tensor(list(z), dtype=tf.float32), zip(*random.sample(self.replay, k=256))))).batch(32))
            accu_loss = 0
            accu_loss_s = 0
            accu_loss_c = 0
            n_batches = 0
            for x in distributed_dataset:
                loss, loss_s, loss_c = self.train_distributed_step(x)
                accu_loss += loss
                accu_loss_s += loss_s
                accu_loss_c += loss_c
                n_batches += 1
            if i % 50 == 0 and i != 0:
                wandb.log({'dq_loss': accu_loss / n_batches, 'dq_s_loss': accu_loss_s / n_batches,
                           'dq_c_loss': accu_loss_c / n_batches,
                           'step': i}, step=i)
            if i % 500 == 0 and i != 0:
                self.sync_networks()

    def train(self, iterations):
        '''
        entry point for training
        '''
        if self.distributed:
            self.train_distributed(iterations)
        else:
            self.train_single(iterations)
