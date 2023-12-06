import copy
import random
from collections import deque
import configparser
import numpy as np
import time
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config = configparser.ConfigParser()
config_file_path = os.path.join(BASE_DIR, 'config.ini')
with open(config_file_path, 'r', encoding='utf-8') as f:
    config.read_file(f)

initial_board_state = [
    ['black_rook', 'black_knight', 'black_bishop', 'black_advisor', 'black_king', 'black_advisor', 'black_bishop',
     'black_knight', 'black_rook'],
    ['', '', '', '', '', '', '', '', ''],
    ['', 'black_cannon', '', '', '', '', '', 'black_cannon', ''],
    ['black_pawn', '', 'black_pawn', '', 'black_pawn', '', 'black_pawn', '', 'black_pawn'],
    ['', '', '', '', '', '', '', '', ''],
    ['', '', '', '', '', '', '', '', ''],
    ['red_pawn', '', 'red_pawn', '', 'red_pawn', '', 'red_pawn', '', 'red_pawn'],
    ['', 'red_cannon', '', '', '', '', '', 'red_cannon', ''],
    ['', '', '', '', '', '', '', '', ''],
    ['red_rook', 'red_knight', 'red_bishop', 'red_advisor', 'red_king', 'red_advisor', 'red_bishop', 'red_knight',
     'red_rook']
]

piece_mapping = {
    'black_rook': np.array([-1, 0, 0, 0, 0, 0, 0]),
    'black_knight': np.array([0, -1, 0, 0, 0, 0, 0]),
    'black_bishop': np.array([0, 0, -1, 0, 0, 0, 0]),
    'black_advisor': np.array([0, 0, 0, -1, 0, 0, 0]),
    'black_king': np.array([0, 0, 0, 0, -1, 0, 0]),
    'black_cannon': np.array([0, 0, 0, 0, 0, -1, 0]),
    'black_pawn': np.array([0, 0, 0, 0, 0, 0, -1]),
    'red_rook': np.array([1, 0, 0, 0, 0, 0, 0]),
    'red_knight': np.array([0, 1, 0, 0, 0, 0, 0]),
    'red_bishop': np.array([0, 0, 1, 0, 0, 0, 0]),
    'red_advisor': np.array([0, 0, 0, 1, 0, 0, 0]),
    'red_king': np.array([0, 0, 0, 0, 1, 0, 0]),
    'red_cannon': np.array([0, 0, 0, 0, 0, 1, 0]),
    'red_pawn': np.array([0, 0, 0, 0, 0, 0, 1]),
    '': np.array([0, 0, 0, 0, 0, 0, 0])
}

# 初始化一个长度为4的队列，用于存储初始状态
initial_deque_state = deque(maxlen=4)
# 遍历4次，将初始棋盘状态复制一份，并添加到队列中
for _ in range(4):
    initial_deque_state.append(copy.deepcopy(initial_board_state))


def find_matching_string(array):
    """
    寻找与array匹配的字符串
    :param array:
    :type array:
    :return:
    :rtype: str or None
    """
    matching_strings = [string for string, value in piece_mapping.items() if np.array_equal(value, array)]
    return matching_strings[0] if matching_strings else None


def change_state(state_list, move):
    """
    根据给定的移动更新状态列表
    :param state_list: 当前状态列表，表示棋盘的二维列表
    :type state_list: list[list[str]]
    :param move: 移动的坐标，格式为 y, x, toy, tox
    :type move: str
    :return: 更新后的状态列表
    :rtype: list[list[str]]
    """
    y, x, toy, tox = map(int, move)
    new_list = [row[:] for row in state_list]  # 使用列表切片复制二维列表

    new_list[toy][tox] = new_list[y][x]
    new_list[y][x] = ''

    return new_list


# test1 = change_state(initial_board_state, '0040')

def state_list_to_state_array(state_list):
    """
    将盘面状态列表转换为状态数组
    :param state_list: 盘面状态列表，表示棋盘的二维列表
    :type state_list: list[list[str]]
    :return: 转换后的状态数组
    :rtype: numpy.ndarray
    """
    _state_array = np.zeros([10, 9, 7])  # 七种棋子
    for i in range(10):
        for j in range(9):
            _state_array[i][j] = piece_mapping[state_list[i][j]]
    return _state_array


# test2 = state_list_to_state_array(initial_board_state)
# print(test2)

def generate_legal_moves_mapping():
    """
    创建所有合法走子,也是神经网络预测的走子概率向量的长度 2086
    :return: 一个包含两个字典的元组，第一个字典将唯一标识符映射到合法走法，第二个字典将合法走法映射到唯一标识符
    :rtype: tuple
    """
    _id_to_action = {}
    _action_to_id = {}

    # 初始化棋盘
    # rows = '012345678'
    # columns = '0123456789'
    row = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
    column = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # 士的全部走法
    advisor_labels = ['0314', '1403', '0514', '1405', '2314', '1423', '2514', '1425',
                      '9384', '8493', '9584', '8495', '7384', '8473', '7584', '8475']

    # 象的全部走法
    bishop_labels = ['2002', '0220', '2042', '4220', '0224', '2402', '4224', '2442',
                     '2406', '0624', '2446', '4624', '0628', '2806', '4628', '2846',
                     '7052', '5270', '7092', '9270', '5274', '7452', '9274', '7492',
                     '7456', '5674', '7496', '9674', '5678', '7856', '9678', '7896']

    idx = 0
    # 遍历棋盘的每一个位置
    for l1 in range(10):
        for n1 in range(9):
            destinations = [(t, n1) for t in range(10)] + \
                           [(l1, t) for t in range(9)] + \
                           [(l1 + a, n1 + b) for (a, b) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]  # 马走日
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(10) and n2 in range(9):
                    # 获取当前位置的合法走法
                    action = column[l1] + row[n1] + column[l2] + row[n2]
                    _id_to_action[idx] = action
                    _action_to_id[action] = idx
                    idx += 1

    # 遍历士的走法
    for action in advisor_labels:
        _id_to_action[idx] = action
        _action_to_id[action] = idx
        idx += 1

    # 遍历象的走法
    for action in bishop_labels:
        _id_to_action[idx] = action
        _action_to_id[action] = idx
        idx += 1

    return _id_to_action, _action_to_id


id_to_action, action_to_id = generate_legal_moves_mapping()


def print_board(_state_array):
    """
    打印盘面
    :param _state_array: 表示棋盘状态的二维列表，大小为 10x9
    :type _state_array: list[list[str]]
    :return: None
    :rtype: None
    """
    board_line = []
    for i in range(10):
        for j in range(9):
            board_line.append(find_matching_string(_state_array[i][j]))
        print(board_line)
        board_line.clear()


# 打印成比较好观察的形式，方便调试
def print_boa(state_list):
    mapping = {
        'black_rook': '黑车', 'black_knight': '黑马', 'black_bishop': '黑象',
        'black_advisor': '黑士', 'black_king': '黑帅', 'black_cannon': '黑炮',
        'black_pawn': '黑兵', 'red_rook': '红车', 'red_knight': '红马',
        'red_bishop': '红象', 'red_advisor': '红士', 'red_king': '红帅',
        'red_cannon': '红炮', 'red_pawn': '红兵', '': ''
    }
    for row in state_list:
        print([mapping[piece] for piece in row])


def flip_map(string):
    """
    镜像当前的走子，扩充数据  example:7052 -> 7856
    :param string: 待镜像的走子
    :type string: str
    :return: 镜像后的走子
    :rtype: str
    """
    return ''.join(str(8 - int(char)) if i % 2 != 0 else char for i, char in enumerate(string))


def bounds_checking(y, x):
    """
    越界检查
    :param y: 坐标
    :type y: int
    :param x: 坐标
    :type x: int
    :return: 如果 x 和 y 均在边界范围内，则返回 True
    :rtype: bool
    """
    return 0 <= y < 10 and 0 <= x < 9


def drop_check(current_color, piece):
    """
    检查下一个落点是否有己方棋子
    :param current_color: 玩家颜色
    :type current_color: str
    :param piece: 落点棋子
    :type piece: str
    :return: 合法则返回 True
    :rtype: bool
    """
    return piece == '' or not piece.startswith(current_color)


def get_legal_moves(state_deque, current_color):
    """
    获取当前棋局中合法的移动列表
    :param state_deque: 保存棋局状态的双端队列
    :type state_deque: deque
    :param current_color: 当前玩家控制棋子颜色
    :type current_color: str
    :return: 包含合法移动的列表，每个移动用一个唯一标识符表示
    :rtype: list
    """
    state_list = state_deque[-1]
    old_state_list = state_deque[-4]
    moves = []  # 用来存放所有合法的着法
    face_to_face = False  # 将对帅
    rk_x, rk_y, bk_x, bk_y = None, None, None, None

    def add_move(_y, _x, _to_y, _to_x):
        m = f"{_y}{_x}{_to_y}{_to_x}"
        if change_state(state_list, m) != old_state_list:
            moves.append(m)

    for y in range(10):
        for x in range(9):
            if state_list[y][x] != '':
                piece_color, piece_type = state_list[y][x].split('_')
                if piece_type == '':  # 不是棋子无法移动
                    continue

                if piece_color == current_color:
                    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 左 右 上 下

                    if piece_type == 'rook':
                        for direction in directions:
                            i, j = direction
                            to_y, to_x = y + i, x + j
                            while bounds_checking(to_y, to_x):
                                if state_list[to_y][to_x] != '':
                                    if piece_color == 'red' and state_list[to_y][to_x].startswith('black'):
                                        add_move(y, x, to_y, to_x)
                                    elif piece_color == 'black' and state_list[to_y][to_x].startswith('red'):
                                        add_move(y, x, to_y, to_x)
                                    break
                                else:
                                    add_move(y, x, to_y, to_x)
                                to_y += i
                                to_x += j

                    elif piece_type == 'knight':
                        for i in [-1, 1]:
                            for j in [-1, 1]:
                                to_y, to_x = y + 2 * i, x + j
                                if bounds_checking(to_y, to_x) and drop_check(piece_color, state_list[to_y][to_x]) \
                                        and state_list[int((to_y + y) / 2)][x] == '':
                                    add_move(y, x, to_y, to_x)
                                to_y, to_x = y + i, x + 2 * j
                                if bounds_checking(to_y, to_x) and drop_check(piece_color, state_list[to_y][to_x]) \
                                        and state_list[y][int((to_x + x) / 2)] == '':
                                    add_move(y, x, to_y, to_x)

                    elif piece_type == 'bishop':
                        # 注意象不能过河！
                        for i in [-2, 2]:
                            for j in [-2, 2]:
                                to_y, to_x = y + i, x + j
                                if bounds_checking(to_y, to_x) and drop_check(piece_color, state_list[to_y][to_x]) \
                                        and state_list[int((to_y + y) / 2)][int((to_x + x) / 2)] == '':
                                    if (piece_color == 'red' and to_y >= 5) or (piece_color == 'black' and to_y <= 4):
                                        add_move(y, x, to_y, to_x)

                    elif piece_type == 'advisor':
                        for i in [-1, 1]:
                            for j in [-1, 1]:
                                to_y, to_x = y + i, x + j
                                if bounds_checking(to_y, to_x) and drop_check(piece_color, state_list[to_y][to_x]):
                                    if (piece_color == 'red' and to_y >= 7 and 3 <= to_x <= 5) \
                                            or (piece_color == 'black' and to_y <= 2 and 3 <= to_x <= 5):
                                        add_move(y, x, to_y, to_x)

                    elif piece_type == 'king':
                        # 记录将帅坐标，判断面对面
                        if piece_color == 'red':
                            rk_y, rk_x = y, x
                        else:
                            bk_y, bk_x = y, x
                        for direction in directions:
                            i, j = direction
                            to_y, to_x = y + i, x + j
                            if bounds_checking(to_y, to_x) and drop_check(piece_color, state_list[to_y][to_x]):
                                if (piece_color == 'red' and to_y >= 7 and 3 <= to_x <= 5) \
                                        or (piece_color == 'black' and to_y <= 2 and 3 <= to_x <= 5):
                                    add_move(y, x, to_y, to_x)

                    elif piece_type == 'cannon':
                        for direction in directions:
                            i, j = direction
                            to_y, to_x = y + i, x + j
                            while bounds_checking(to_y, to_x):
                                if state_list[to_y][to_x] != '':
                                    break
                                add_move(y, x, to_y, to_x)
                                to_y += i
                                to_x += j
                            # 跳过第一个遇到的棋子
                            to_y += i
                            to_x += j
                            while bounds_checking(to_y, to_x):
                                if state_list[to_y][to_x] != '':
                                    if not state_list[to_y][to_x].startswith(piece_color):
                                        add_move(y, x, to_y, to_x)
                                    break
                                to_y += i
                                to_x += j

                    elif piece_type == 'pawn':
                        # 红卒上走 黑卒下走
                        directions = [(-1, 0)] if piece_color == 'red' else [(1, 0)]
                        for i, j in directions:
                            to_y, to_x = y + i, x + j
                            if bounds_checking(to_y, to_x) and drop_check(piece_color, state_list[to_y][to_x]):
                                add_move(y, x, to_y, to_x)

                        # 过河就能移动三个方向
                        if piece_color == 'red' and y <= 4:
                            for i, j in [(-1, 0), (0, 1), (0, -1)]:
                                to_y, to_x = y + i, x + j
                                if bounds_checking(to_y, to_x) and drop_check(piece_color, state_list[to_y][to_x]):
                                    add_move(y, x, to_y, to_x)

                        elif piece_color == 'black' and y >= 5:
                            for i, j in [(1, 0), (0, 1), (0, -1)]:
                                to_y, to_x = y + i, x + j
                                if bounds_checking(to_y, to_x) and drop_check(piece_color, state_list[to_y][to_x]):
                                    add_move(y, x, to_y, to_x)

    if rk_x is not None and bk_x is not None and rk_x == bk_x:
        face_to_face = True
        for i in range(bk_y + 1, rk_y, 1):
            if state_list[i][rk_x] != '':
                face_to_face = False
                break

    if face_to_face:
        if current_color == 'red':
            add_move(rk_y, rk_x, bk_y, bk_x)
        else:
            add_move(bk_y, bk_x, rk_y, rk_x)

    moves_id = []
    for move in moves:
        moves_id.append(action_to_id[move])
    return moves_id


class Board(object):

    def __init__(self):
        self.current_player_id = None
        self.last_move = None
        self.action_count = None
        self.current_player_color = None
        self.color_to_id = None
        self.kill_action = None
        self.id_to_color = None
        self.state_list = copy.deepcopy(initial_board_state)
        self.game_start = False
        self.winner = None
        self.state_deque = copy.deepcopy(initial_deque_state)

    # 初始化棋盘的方法
    def init_board(self, start_player=1):  # 传入先手玩家的id
        # 增加一个颜色到id的映射字典，id到颜色的映射字典
        # 永远红方先移动
        if start_player == 1:
            self.id_to_color = {1: 'red', 2: 'black'}
            self.color_to_id = {'red': 1, 'black': 2}
        elif start_player == 2:
            self.id_to_color = {2: 'red', 1: 'black'}
            self.color_to_id = {'red': 2, 'black': 1}
        # 当前手玩家，也就是先手玩家
        self.current_player_color = self.id_to_color[start_player]  # 红
        self.current_player_id = self.color_to_id['red']
        # 初始化棋盘状态
        self.state_list = copy.deepcopy(initial_board_state)
        self.state_deque = copy.deepcopy(initial_deque_state)
        # 初始化最后落子位置
        self.last_move = -1
        # 记录游戏中吃子的回合数
        self.kill_action = 0
        self.game_start = False
        self.action_count = 0  # 游戏动作计数器
        self.winner = None

    @property
    # 获的当前盘面的所有合法走子集合
    def available(self):
        return get_legal_moves(self.state_deque, self.current_player_color)

    # 从当前玩家的视角返回棋盘状态，current_state_array: [9, 10, 9]  CHW
    def current_state(self):
        _current_state = np.zeros([9, 10, 9])
        # 使用9个平面来表示棋盘状态
        # 0-6个平面表示棋子位置，1代表红方棋子，-1代表黑方棋子, 队列最后一个盘面
        # 第7个平面表示对手player最近一步的落子位置，走子之前的位置为-1，走子之后的位置为1，其余全部是0
        # 第8个平面表示的是当前player是不是先手player，如果是先手player则整个平面全部为1，否则全部为0
        if self.game_start:
            _current_state[:7] = state_list_to_state_array(self.state_deque[-1]).transpose([2, 0, 1])  # [7, 10, 9]
            # 解构self.last_move
            move = id_to_action[self.last_move]
            start_position = int(move[0]), int(move[1])
            end_position = int(move[2]), int(move[3])
            _current_state[7][start_position[0]][start_position[1]] = -1
            _current_state[7][end_position[0]][end_position[1]] = 1
        # 指出当前是哪个玩家走子
        if self.action_count % 2 == 0:
            _current_state[8][:, :] = 1.0

        return _current_state

    # 根据move对棋盘状态做出改变
    def do_move(self, move):
        self.game_start = True  # 游戏开始
        self.action_count += 1  # 移动次数加1
        move_action = id_to_action[move]
        start_y, start_x = int(move_action[0]), int(move_action[1])
        end_y, end_x = int(move_action[2]), int(move_action[3])
        state_list = copy.deepcopy(self.state_deque[-1])
        # 判断是否吃子
        if state_list[end_y][end_x] != '':
            # 如果吃掉对方的帅，则返回当前的current_player胜利
            self.kill_action = 0
            if self.current_player_color == 'black' and state_list[end_y][end_x] == 'red_king':
                self.winner = self.color_to_id['black']
            elif self.current_player_color == 'red' and state_list[end_y][end_x] == 'black_king':
                self.winner = self.color_to_id['red']
        else:
            self.kill_action += 1
        # 更改棋盘状态
        state_list[end_y][end_x] = state_list[start_y][start_x]
        state_list[start_y][start_x] = ''
        self.current_player_color = 'black' if self.current_player_color == 'red' else 'red'  # 改变当前玩家
        self.current_player_id = 1 if self.current_player_id == 2 else 2
        # 记录最后一次移动的位置
        self.last_move = move
        self.state_deque.append(state_list)

    # 是否产生赢家
    def has_a_winner(self):
        """一共有三种状态，红方胜，黑方胜，平局"""
        if self.winner is not None:
            return True, self.winner
        elif self.kill_action >= int(config.get('Game', 'kill_action')):  # 平局
            return False, -1
        return False, -1

    # 检查当前棋局是否结束
    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif self.kill_action >= int(config.get('Game', 'kill_action')):  # 平局，没有赢家
            return True, -1
        return False, -1

    def get_current_player_color(self):
        return self.current_player_color

    def get_current_player_id(self):
        return self.current_player_id


class Game(object):

    def __init__(self, board):
        self.board = board

    def start_play(self, player1, player2, start_player=1, is_shown=1):
        if start_player not in (1, 2):
            raise Exception('start_player should be either 1 (player1 first) '
                            'or 2 (player2 first)')
        self.board.init_board(start_player)  # 初始化棋盘
        p1, p2 = 1, 2
        player1.set_player_ind(1)
        player2.set_player_ind(2)
        players = {p1: player1, p2: player2}
        if is_shown:
            graphic(self.board, self.board.id_to_color[player1.player], self.board.id_to_color[player2.player])

        while True:
            current_player = self.board.get_current_player_id()  # 红子对应的玩家id
            player_in_turn = players[current_player]  # 决定当前玩家的代理
            move = player_in_turn.get_action(self.board)  # 当前玩家代理拿到动作
            self.board.do_move(move)  # 棋盘做出改变
            if is_shown:
                graphic(self.board, self.board.id_to_color[player1.player], self.board.id_to_color[player2.player])
            end, winner = self.board.game_end()
            if end:
                if winner != -1:
                    print("Game end. Winner is", players[winner])
                else:
                    print("Game end. Tie")
                return winner

    # 使用蒙特卡洛树搜索开始自我对弈，存储游戏状态（状态，蒙特卡洛落子概率，胜负手）三元组用于神经网络训练
    def start_self_play(self, player, is_shown=False, temp=1e-3):
        self.board.init_board()  # 初始化棋盘, start_player=1
        states, mcts_probs, current_players = [], [], []
        # 开始自我对弈
        _count = 0
        while True:
            _count += 1
            if _count % 20 == 0:
                start_time = time.time()
                move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
                print('one-step cost: ', time.time() - start_time)
            else:
                move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            # 保存自我对弈的数据
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player_id)
            # 执行一步落子
            self.board.do_move(move)
            end, winner = self.board.game_end()
            if end:
                # 从每一个状态state对应的玩家的视角保存胜负信息
                winner_z = np.zeros(len(current_players))
                if winner != -1:
                    winner_z[np.array(current_players) == winner] = 1.0
                    winner_z[np.array(current_players) != winner] = -1.0
                # 重置蒙特卡洛根节点
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is:", winner)
                    else:
                        print('Game end. Tie')

                return winner, zip(states, mcts_probs, winner_z)


def graphic(board, player1_color, player2_color):
    print('player1 take: ', player1_color)
    print('player2 take: ', player2_color)
    print_board(state_list_to_state_array(board.state_deque[-1]))


if __name__ == '__main__':
    # print(id_to_action)
    '''# 测试初始局面走子
    mo = get_legal_moves(initial_deque_state, current_color='black')
    print(len(mo))
    print(mo)
    for z in mo:
        print(id_to_action[z])

    # 测试
    class Human1:
        def __init__(self):
            self.player = None

        @staticmethod
        def get_action(board):
            # print(board.current_player_color)
            # move = action_to_id[input('请输入')]
            # if move in board.available:
            #     print("合法")
            # else:
            #     print("不合法")
            move = random.choice(board.available)
            return move

        def set_player_ind(self, p):
            self.player = p


    class Human2:
        def __init__(self):
            self.player = None

        @staticmethod
        def get_action(board):
            # print(board.current_player_color)
            # move = action_to_id[input('请输入')]
            move = random.choice(board.available)
            return move

        def set_player_ind(self, p):
            self.player = p


    human1 = Human1()
    human2 = Human2()
    game = Game(board=Board())
    for _ in range(10):
        game.start_play(human1, human2, start_player=1, is_shown=0)'''
