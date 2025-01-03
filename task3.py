import pandas as pd
import os
import sys
import arrange
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PPO import TransformerPolicyNetwork
from tqdm import tqdm
import math

CONTAINERS = (35, 23, 13), (37, 26, 13), (38, 26, 13), (40, 28, 16), (42, 30, 18), (42, 40, 30), (52, 40, 17), (54, 45, 36)

def select_container(items, containers):
    # 找出最大的长、宽、高
    max_length = max(item[0] for item in items)  # 最大长度
    max_width = max(item[1] for item in items)   # 最大宽度
    max_height = max(item[2] for item in items)  # 最大高度

    # 计算最大物体的体积
    thr_volume = max(item[0] * item[1] * item[2] for item in items)

    suitable_containers = []
    for container in containers:

        # 检查容器是否比所有 items 中最大的边长都大，作为合适的容器
        if (container[0] >= max_length and
            container[1] >= max_width and
            container[2] >= max_height):
            suitable_containers.append(container)

    # 如果找到合适的容器，返回与最大物体体积最接近的那个
    if suitable_containers:
        return min(suitable_containers, key=lambda c: c[0] * c[1] * c[2] - thr_volume)

    return (54, 45, 36)  # 若未找到合适的容器则返回最大的

def check_item(item, containers):
    # 提取物体边长并排序
    edges = sorted(item[:3], reverse=True)

    # 遍历所有容器，检查是否可以放下
    for container in containers:
        if (edges[0] <= container[0] and
            edges[1] <= container[1] and
            edges[2] <= container[2]):
            return True
    return False
    
# 更改当前工作目录
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_dir)

# 读取CSV文件
df = pd.read_csv('task3.csv')

# 创建一个字典来存储统计信息
statistics = dict()

# 遍历表格中的每一行数据
for index, row in df.iterrows():
    sta_code = row['sta_code']
    sku_code = row['sku_code']

    # 如果订单号不在字典中，进行初始化
    if sta_code not in statistics:
        statistics[sta_code] = {}

    # 如果物品号不在相应订单号的字典中，进行初始化，加入物体的长、宽、高、数量
    if sku_code not in statistics[sta_code]:
        statistics[sta_code][sku_code] = [row['长(CM)'], row['宽(CM)'], row['高(CM)'], row['qty']]
    else:
        # 如果物品号 已存在，更新其数量
        statistics[sta_code][sku_code][3] += row['qty']

# 引入策略网络，初始化容器和物体总体积
policy_net = TransformerPolicyNetwork()
state_dict = torch.load("policy_net_epoch_100.pth")
policy_net.load_state_dict(state_dict)
Volume_of_container = 0
Volume_of_items = 0

# 遍历所有订单
for sta_code, sku_dict in tqdm(statistics.items(), desc="Processing"):
    items = []
    legal_item = True
    # 检查每个订单下是否有物体超出范围，如果合法则更新物体的体积，将一个订单下所有的物体加入列表
    for sku_code, item in sku_dict.items():
        if check_item(item, CONTAINERS) == False:
            legal_item = False
            print(item, "can't be put into any container!")
            break
        Volume_of_items += np.prod(item)
        items.append([math.ceil(num) for num in item])
    if not legal_item:
        continue
    # 针对一个订单下所有的物体，执行循环装箱
    done = False
    while not done:
        # 选择箱子
        container = select_container(items, CONTAINERS)
        # 调用模型进行一次装箱
        done, action_history, items = arrange.main('avgDepth', container, items, 5, policy_net)
        # 输出本次装箱的动作
        if action_history:
            Volume_of_container += np.prod(container)
            ratio = Volume_of_items / Volume_of_container
            print("Pick a container", container, "for task", sta_code, "and do", action_history)
    # 输出完成每个订单后，累积的空间利用率
    print("The ratio now is: ", ratio)

# 输出最后累积的空间利用率
ratio = Volume_of_items / Volume_of_container
print("The ratio now is: ", ratio)

