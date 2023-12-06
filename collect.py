from collections import deque
import copy
import os
import pickle
import time
from chess_game import Board, Game, action_to_id, id_to_action, flip_map
from montecarlo import MCTSPlayer
from net import PolicyValueNet
import configparser
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config = configparser.ConfigParser()
config_file_path = os.path.join(BASE_DIR, 'config.ini')
with open(config_file_path, 'r', encoding='utf-8') as f:
    config.read_file(f)


# 定义整个对弈收集数据流程
def get_equi_data(play_data):
    """
    左右对称变换，扩充数据集一倍，加速一倍训练速度
    :param play_data: 原始数据 (state, mcts_prob, winner)
    :type play_data: list
    :return: 应用对称变换后的扩展数据
    :rtype: list
    """
    extend_data = []
    # 棋盘状态shape is [9, 10, 9], 走子概率，赢家
    for state, mcts_prob, winner in play_data:
        # 原始数据
        extend_data.append((state, mcts_prob, winner))
        # 水平翻转后的数据
        state_flip = state.transpose([1, 2, 0])
        state = state.transpose([1, 2, 0])
        for i in range(10):
            for j in range(9):
                state_flip[i][j] = state[i][8 - j]
        state_flip = state_flip.transpose([2, 0, 1])
        mcts_prob_flip = copy.deepcopy(mcts_prob)
        for i in range(len(mcts_prob_flip)):
            mcts_prob_flip[i] = mcts_prob[action_to_id[flip_map(id_to_action[i])]]
        extend_data.append((state_flip, mcts_prob_flip, winner))
    return extend_data


class CollectPipeline:

    def __init__(self, init_model=None):
        # 象棋逻辑和棋盘
        self.episode_len = None
        self.mcts_player = None
        self.policy_value_net = None
        self.board = Board()
        self.game = Game(self.board)
        # 对弈参数
        self.temp = 1  # 温度
        self.n_playout = int(config.get('Game', 'play_out'))  # 每次移动的模拟次数
        self.c_puct = int(config.get('Train', 'c_puct'))  # u的权重
        self.buffer_size = int(config.get('Train', 'buffer_size'))  # 经验池大小
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0

    # 从主体加载模型
    def load_model(self):
        try:
            self.policy_value_net = PolicyValueNet(model_file=config.get('Train', 'model_path'))
            print('The latest model has been loaded')
        except FileNotFoundError:
            self.policy_value_net = PolicyValueNet()
            print('Initial model has been loaded')
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def collect_selfplay_data(self, n_games=1):
        """
        收集自我对弈的数据
        :param n_games: 进行自我对弈的游戏次数
        :type n_games: int
        :return: 完成自我对弈后的迭代次数
        :rtype: int
        """
        for i in range(n_games):
            self.load_model()  # 从本体处加载最新模型
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp, is_shown=False)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # 增加数据
            play_data = get_equi_data(play_data)

            if os.path.exists(config.get('Train', 'train_data_buffer_path')):
                while True:
                    try:
                        with open(config.get('Train', 'train_data_buffer_path'), 'rb') as data_dict:
                            data_file = pickle.load(data_dict)
                            self.data_buffer = data_file['data_buffer']
                            self.iters = data_file['iters']
                            del data_file
                            self.iters += 1
                            self.data_buffer.extend(play_data)
                        print('Loaded data')
                        break
                    except:
                        time.sleep(30)
            else:
                self.data_buffer.extend(play_data)
                self.iters += 1
            data_dict = {'data_buffer': self.data_buffer, 'iters': self.iters}
            with open(config.get('Train', 'train_data_buffer_path'), 'wb') as data_file:
                pickle.dump(data_dict, data_file)
        return self.iters

    def run(self):
        """开始收集数据"""
        try:
            while True:
                iters = self.collect_selfplay_data()
                print('batch i: {}, episode_len: {}'.format(iters, self.episode_len))
        except KeyboardInterrupt:
            print('\n\rUser exit')


collecting_pipeline = CollectPipeline(init_model='models/policy.pkl')
collecting_pipeline.run()
