import random
import numpy as np
from net import PolicyValueNet
import pickle
import time
import configparser
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config = configparser.ConfigParser()
config_file_path = os.path.join(BASE_DIR, 'config.ini')
with open(config_file_path, 'r', encoding='utf-8') as f:
    config.read_file(f)


# 定义整个训练流程
class TrainPipeline:

    def __init__(self, init_model=None):
        # 训练参数
        self.iters = None
        self.data_buffer = None
        self.learn_rate = 1e-3
        self.lr_multiplier = 1  # 基于KL自适应的调整学习率
        self.temp = 1.0
        self.batch_size = int(config.get('Train', 'batch_size'))  # 训练的batch大小
        self.epoch = int(config.get('Train', 'epoch'))  # 每次更新的train_step数量
        self.kl_targ = float(config.get('Train', 'kl_targ'))  # kl散度控制
        self.check_freq = 100  # 保存模型的频率
        self.game_batch_num = int(config.get('Train', 'game_batch_num'))  # 训练更新的次数

        if init_model:
            try:
                self.policy_value_net = PolicyValueNet(model_file=init_model)
                print('The last final model was loaded')
            except FileNotFoundError:
                # 从零开始训练
                print("Model doesn't exist. Start retraining")
                self.policy_value_net = PolicyValueNet()
        else:
            print('Start retraining')
            self.policy_value_net = PolicyValueNet()

    def policy_update(self):
        """更新策略价值网络"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)

        state_batch = [data[0] for data in mini_batch]
        state_batch = np.array(state_batch).astype('float32')

        mcts_probs_batch = [data[1] for data in mini_batch]
        mcts_probs_batch = np.array(mcts_probs_batch).astype('float32')

        winner_batch = [data[2] for data in mini_batch]
        winner_batch = np.array(winner_batch).astype('float32')

        # 旧的策略，旧的价值函数
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        kl = loss = entropy = 0
        for i in range(self.epoch):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier
            )
            # 新的策略，新的价值函数
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)

            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1))
            if kl > self.kl_targ * 4:  # 如果KL散度很差，则提前终止
                break

        # 自适应调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))

        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def run(self):
        """开始训练"""
        try:
            for i in range(self.game_batch_num):
                time.sleep(int(config.get('Train', 'update_time')))
                while True:
                    try:
                        with open(config.get('Train', 'train_data_buffer_path'), 'rb') as data_dict:
                            data_file = pickle.load(data_dict)
                            self.data_buffer = data_file['data_buffer']
                            self.iters = data_file['iters']
                            del data_file
                        print('Loaded data')
                        break
                    # 什么异常还没想好
                    except:
                        time.sleep(30)
                print('step i {}: '.format(self.iters))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                self.policy_value_net.save_model(config.get('Train', 'model_path'))
                if (i + 1) % self.check_freq == 0:
                    print('current selfplay batch: {}'.format(i + 1))
                    self.policy_value_net.save_model('models/policy_batch{}.model'.format(i + 1))
        except KeyboardInterrupt:
            print('\n\rUser exit')


training_pipeline = TrainPipeline(init_model='policy.pkl')
training_pipeline.run()
