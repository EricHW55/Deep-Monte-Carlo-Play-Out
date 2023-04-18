# import numpy as np
import tensorflow as tf
from model import Model
from MCPO import MCTF
from env import connect4
import numpy as np
from typing import Tuple
from copy import deepcopy
from ReplayMemory import Memory

from reward_model import DQN
# reward_model = DQN()

m = Model()
# action_model = m.action_model()
reward_model_1 = DQN()
reward_model_2 = DQN()
# reward_model_1.summary(); reward_model_2.summary()
# model.summary(); reward_model.summary()
# replay_memory = []
length = 100
reward_memory_1 = Memory(length)  # 선공
reward_memory_2 = Memory(length)  # 후공

mctf = MCTF()
mctf(1, False)

env = connect4(main_page=False)
node_id = (0,)
num = 500 
# reward = mctf.mcpo(node_id, num, False)
# state = deepcopy(env.state)
# state = tf.reshape(state, (1,42))
# reward_memory.append([node_id, state, reward])
# m.reward_train(reward_model, state, reward, epochs=8)

def action(node_id, first) -> Tuple[tuple, np.array, float]:
    action_num = mctf.select_action(node_id, num=500, first=first)
    env.action(action_num)
    node_id += (action_num, )
    state = deepcopy(env.state)
    state = tf.reshape(state, (1,42))
    reward = mctf.mcpo(node_id, num, False)
    return node_id, state, reward

player = True # True : agent, False : player
game_done = False
for i in range(16):
    while not game_done:
        # print(i)
        node_id, state, reward = action(node_id, first=player)
        if player:
            reward_memory_1.append([node_id, state, reward])
            # print('1')
        else:
            reward_memory_2.append([node_id, state, reward])
            # print('2')
        # env.main()
        player = not(player)
        game_done = True if not env.check_finish() == None else False  # if 게임이 끝 True else False

    
    batch_size = 8
    batch_data_1 = reward_memory_1.sample(batch_size)
    batch_data_2 = reward_memory_2.sample(batch_size)

    for node_id, state, reward in batch_data_1:
        m.reward_train(reward_model_1, state, reward, epochs=2)

    for node_id, state, reward in batch_data_2:
        m.reward_train(reward_model_2, state, reward, epochs=2)

    

    env = connect4(main_page=False)
    node_id = (0,)
    game_done = False
    player = True # True : agent, False : player




batch_size = 8
# batch_data = random.sample(replay_memory, batch_size)
batch_data_1 = reward_memory_1.sample(batch_size)
batch_data_2 = reward_memory_2.sample(batch_size)

for node_id, state, reward in batch_data_1:
    m.reward_train(reward_model_1, state, reward, epochs=2)

for node_id, state, reward in batch_data_2:
    m.reward_train(reward_model_2, state, reward, epochs=2)
        # print(reward)

# node_id, state, reward = action(node_id, player)
# env.main()
# player = not(player)

# # print(state)
# print('reward : ', reward)
# print(reward_model_1(state))
# print(';;;')
node_id, state, reward = reward_memory_1.sample(1)[0]
print(reward_model_1(state))
print(state, reward)
print((reward-float(reward_model_1(state)))/reward*100,'% : Model1\n')

node_id, state, reward = reward_memory_2.sample(1)[0]
print(reward_model_2(state))
print(state, reward)
print((reward-float(reward_model_2(state)))/reward*100,'% : Model2')

