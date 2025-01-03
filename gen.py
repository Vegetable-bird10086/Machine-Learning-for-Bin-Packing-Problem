import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
def split_item(item):
    # 以边长为权重随机选择一个轴
    lengths = item[3:6]
    axis = random.choices([0, 1, 2], weights=lengths, k=1)[0]
    
    # 使用正态分布生成切割点，均值为物体边长的一半，标准差为边长的1/6
    mean = item[3 + axis] / 2
    std_dev = item[3 + axis] / 6
    position = int(np.clip(np.random.normal(mean, std_dev), 1, item[3 + axis] - 1))  # 确保位置在有效范围内

    # 防止切割后尺寸为零
    if position <= 0 or position >= item[3 + axis]:
        position = item[3 + axis] // 2  # 默认为物品的中点切割
    
    # 计算新物品的尺寸
    new_item_1 = list(item)
    new_item_2 = list(item)
    
    # 切割物品
    new_item_1[3 + axis] = position
    new_item_2[3 + axis] = item[3 + axis] - position
    
    # 更新位置
    if axis == 0:  # x轴切割
        new_item_2[0] += position
    elif axis == 1:  # y轴切割
        new_item_2[1] += position
    elif axis == 2:  # z轴切割
        new_item_2[2] += position
    
    # 检查新物品尺寸是否为零，如果为零则舍弃
    if new_item_1[3] == 0 or new_item_1[4] == 0 or new_item_1[5] == 0:
        return None
    if new_item_2[3] == 0 or new_item_2[4] == 0 or new_item_2[5] == 0:
        return None
    
    return new_item_1, new_item_2


def generate_bin_packing_data():
    # 初始化物品列表，格式为 (x, y, z, length, width, height)
    items = [(0, 0, 0, 10, 10, 10)]  # 初始大箱

    # 根据边长决定被选中的概率
    weights = [item[3] * item[4] * item[5] for item in items]  # 体积作为权重

    # 从范围中随机抽取切割数
    N = random.randint(15, 25)

    while len(items) < N:
        # 根据权重选择一个物品
        item_to_split = random.choices(items, weights=weights, k=1)[0]

        # 从物品列表中移除该物品
        items.remove(item_to_split)
        weights.remove(item_to_split[3] * item_to_split[4] * item_to_split[5])

        # 切割物品
        result = split_item(item_to_split)
        
        # 如果 split_item 返回 None，则跳过当前切割操作
        if result is None:
            continue

        new_item_1, new_item_2 = result

        # 添加新物品到列表
        items.append(tuple(new_item_1))
        items.append(tuple(new_item_2))

        # 更新权重
        weights.append(new_item_1[3] * new_item_1[4] * new_item_1[5])
        weights.append(new_item_2[3] * new_item_2[4] * new_item_2[5])

    # 返回仅包含尺寸信息的物品列表
    return [[item[3], item[4], item[5]] for item in items]



def plot_items(items):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for item in items:
        x, y, z = item[0], item[1], item[2]
        l, w, h = item[3], item[4], item[5]
        
        # Create a 3D box
        ax.bar3d(x, y, z, l, w, h, alpha=0.5)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()


 # 批量生成和保存数据集时用
def save_data_sets(train_count=10, test_count=10):
    # 保存训练集
    train_data = []
    box_size=(10,10,10)
    for _ in range(train_count):
        items = generate_bin_packing_data()
        train_data.append({"box_size": box_size, "items": items})
    
    with open('train_sets.json', 'w') as f:
        json.dump(train_data, f, indent=4)
    
    # 保存测试集
    test_data = []
    for _ in range(test_count):
        items = generate_bin_packing_data()
        test_data.append({"box_size": box_size, "items": items})
    
    with open('test_sets.json', 'w') as f:
        json.dump(test_data, f, indent=4)

if __name__ == "__main__":
    save_data_sets()
