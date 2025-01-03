import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PPO import TransformerPolicyNetwork
import os
import sys

STRA = 'avgDepth'
BOX_SIZE = (10, 10, 10)  # 箱子的尺寸
ITEMS = [[2, 2, 2, 5], [3, 1, 4, 5], [1, 3, 3, 1], [5, 3, 4, 3]]  # 物体的状态表示
ROLL = 3 # 随机模拟分支数

# 计算箱子的状态表示向量中，后6个位置的属性值
def calculate_attributes(state, x, y):
    for i in range(x):
        for j in range(y):
            height = state[i, j, 0]

            # 1号位置：向右高度不变的距离
            r_distance = 0
            while i + r_distance < x and state[i + r_distance, j, 0] == height:
                r_distance += 1
            state[i, j, 1] = r_distance

            # 2号位置：向左高度不变的距离
            l_distance = 0
            while i - l_distance - 1 >= 0 and state[i - l_distance - 1, j, 0] == height:
                l_distance += 1
            state[i, j, 2] = l_distance

            # 3号位置：向下高度不变的距离
            d_distance = 0
            while j - d_distance - 1 >= 0 and state[i, j - d_distance - 1, 0] == height:
                d_distance += 1
            state[i, j, 3] = d_distance

            # 4号位置：向上高度不变的距离
            u_distance = 0
            while j + u_distance < y and state[i, j + u_distance, 0] == height:
                u_distance += 1
            state[i, j, 4] = u_distance

            # 5号位置：向右比它高的点的距离
            vertical_distance = 0
            while i + vertical_distance < x and state[i + vertical_distance, j, 0] <= height:
                vertical_distance += 1
            state[i, j, 5] = vertical_distance

            # 6号位置：向上比它高的点的距离
            horizontal_distance = 0
            while j + horizontal_distance < y and state[i, j + horizontal_distance, 0] <= height:
                horizontal_distance += 1
            state[i, j, 6] = horizontal_distance

    return state

# 遍历搜索合法动作
def get_legal_actions(state, items, x, y, z, init = False):
    # 初始化合法动作列表
    legal_actions = []
    
    if len(items) > 0:
        # 第一步中优先选择大物体和左下角
        if init:
            x_step, y_step = 1, 1
            item = max(items, key=lambda item: item[0] * item[1] * item[2])
        # 模拟步中优先选择小物体，并在一定步长的基础上遍历
        else:
            x_step, y_step = 5, 5
            item = min(items, key=lambda item: item[0] * item[1] * item[2])
        
        length, width, height, _ = item
        # 生成所选物体的所有旋转可能
        rotate_items = [[length, width, height],[width, length, height],[height, width, length],[length, height, width],[width, height, length],[height, length, width]]
        

        # 遍历位置和旋转情况
        for i in range(0, x, x//x_step):
            for j in range(0, y, y//y_step):
                for l in range(6):
                    action = dict()
                    length, width, height = rotate_items[l]

                    # 检查是否超出长、宽
                    out_of_range = False
                    for p in range(0, length):
                        if width > state[min(i + p, x - 1), j, 6]:
                            out_of_range = True

                    for q in range(0, width):
                        if length > state[i, min(j + q, y - 1), 5]:
                            out_of_range = True
                                 
                    if not out_of_range:
                        current_height = state[i: i + length, j: j + width, 0]
                        # 检查是否超出高度
                        if (current_height + height).max() <= z :
                            # 记录合法动作细节并加入列表
                            action['position'] = [i, j, state[i, j, 0]]
                            action['item'] = item
                            action['rotation'] = l
                            legal_actions.append(action)

    return legal_actions

# 应用动作并更新状态表示
def apply_action(box_state, items, action):

    state = box_state.copy()
    available_items = np.copy(items)
    position = action['position']
    item = action['item']
    rotation = action['rotation']

    # 根据rotation旋转物体
    length, width, height, _ = item
    if rotation == 1:
         length, width = width, length
    elif rotation == 2:
        length, height = height, length
    elif rotation == 3:
        width, height = height, width
    elif rotation == 4:
        length, width, height = width, height, length
    elif rotation == 5:
        length, width, height = height, length, width
    
    # 放入物体，更新箱子状态表示中的高度项
    state[position[0]: position[0] + length, position[1]: position[1] + width, 0] = height + state[position[0], position[1], 0]

    # 更新剩余物品，当数量降为0时删除表项
    for available_item in available_items:
        if np.array_equal(available_item, item):
            available_item[3] = available_item[3] - 1
            break
    rest_items = [available_item for available_item in available_items if available_item[3] > 0]
    return state, rest_items  # 返回新的状态

# 计算数组的平均
def average(lst):
    return sum(lst) / len(lst) if lst else 0

# 随机模拟后续过程
def perform_simulation(state, items, x, y, z):
    depth = 0 # 初始化深度
    new_state = state
    rest_items = items
    # 获取合法动作列表
    legal_actions = get_legal_actions(new_state, rest_items, x, y, z)
    while (len(legal_actions) > 0 and len(rest_items) > 0):
        # 随机一个合法动作并执行，更新深度和状态
        action = random.choice(legal_actions)
        raw_state, rest_items = apply_action(new_state, rest_items, action)
        new_state = calculate_attributes(raw_state, x, y)
        depth += 1
        # 根据新的状态获取新的合法动作列表
        if len(rest_items) > 0:
            legal_actions = get_legal_actions(new_state, rest_items, x, y, z)
    return depth

# 使用策略网络生成合法动作
def generate_legal_actions(policy_net, box_state, rest_items, x, y, z):
    try_count = 0
    legal_actions = []
    while try_count < 10:
        try_count = try_count + 1
        action = dict()
        position_index, item_index, rotation_index, _, _, _ = policy_net.forward(box_state, [row[:3] for row in rest_items]) # 让网络根据当前状态生成动作
        length, width, height, _ = rest_items[item_index]
        rotate_items = [[length, width, height],[width, length, height],[height, width, length],[length, height, width],[width, height, length],[height, length, width]]
        length, width, height = rotate_items[rotation_index]
        i, j = position_index // y, position_index % y

        # 检查是否超出长、宽
        out_of_range = False
        for p in range(0, length):
            if width > box_state[min(i + p, x - 1), j, 6]:
                out_of_range = True

        for q in range(0, width):
            if length > box_state[i, min(j + q, y - 1), 5]:
                out_of_range = True
                                 
        if not out_of_range:
            current_height = box_state[i: i + length, j: j + width, 0]
            # 检查是否超出高度
            if (current_height + height).max() <= z :
                # 记录合法动作细节并加入列表
                action['position'] = [i, j, box_state[i, j, 0]]
                action['item'] = rest_items[item_index]
                action['rotation'] = rotation_index
                legal_actions.append(action)
    
    return legal_actions

def main(strategy, box_size, items_list, n_roll, policy_net):
    # 初始化状态
    raw_state = np.zeros([box_size[0],box_size[1],7], dtype=int)
    box_state = calculate_attributes(raw_state, box_size[0], box_size[1])
    items = items_list
    action_history = []

    # 生成初始动作
    if len(items) > 0:
        legal_actions = get_legal_actions(box_state, items, box_size[0], box_size[1], box_size[2], True)

    # 在有合法动作时循环
    while len(legal_actions) > 0:
        best_action = None
        best_max_depth, best_avg_depth = 0, 0
        
        # 执行所有合法动作
        for legal_action in legal_actions:
            new_raw_state, rest_items = apply_action(box_state, items, legal_action)
            new_box_state = calculate_attributes(new_raw_state, box_size[0], box_size[1])
            action_depths = []
            
            # 分别执行随机合法动作，记录深度
            for _ in range(n_roll):
                depth = perform_simulation(new_box_state, rest_items, box_size[0], box_size[1], box_size[2])
                action_depths.append(depth)
            max_depth = max(action_depths)
            avg_depth = average(action_depths)

            # 根据深度和策略选择当前最佳合法动作
            if strategy == 'maxDepth' and max_depth >= best_max_depth:
                best_max_depth = max_depth
                best_action = legal_action
            elif strategy == 'avgDepth' and avg_depth >= best_avg_depth:
                best_avg_depth = avg_depth
                best_action = legal_action
        
        # 执行当前最佳合法动作，更新状态并记录历史
        raw_state, items = apply_action(box_state, items, best_action)
        box_state = calculate_attributes(raw_state, box_size[0],box_size[1])
        action_history.append(best_action)

        # 若还有未装入物体，调用策略网络生成一批后续合法动作
        if len(items) > 0:
            legal_actions = generate_legal_actions(policy_net, box_state, items, box_size[0], box_size[1], box_size[2]) + get_legal_actions(box_state, items, box_size[0], box_size[1], box_size[2], True)
        else:
            break

    solution_found = len(items) == 0

    return solution_found, action_history, items

# 打印可视化结果
def plot_items(action_history):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for action in action_history:
        position = action['position']
        item = action['item']
        rotation = action['rotation']
        
        x, y, z = position[0], position[1], position[2]
        length, width, height = item[0], item[1], item[2]

        if rotation == 1:
            length, width = width, length
        elif rotation == 2:
            length, height = height, length
        elif rotation == 3:
            width, height = height, width
        elif rotation == 4:
            length, width, height = width, height, length
        elif rotation == 5:
            length, width, height = height, length, width
        
        # Create a 3D box
        ax.bar3d(x, y, z, length, width, height, alpha=0.5)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(script_dir)
    policy_net = TransformerPolicyNetwork()
    state_dict = torch.load("policy_net_epoch_10.pth")
    policy_net.load_state_dict(state_dict)
    solution_found, action_history, _ = main(STRA, BOX_SIZE, ITEMS, ROLL, policy_net)
    plot_items(action_history)