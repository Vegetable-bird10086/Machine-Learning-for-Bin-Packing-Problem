import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 加载数据集
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# 加载训练集
train_data = load_data('train_sets.json')
test_data = load_data('test1.json')
LEARNING_RATE = 0.001
NUM_EPISODES = 20
EPSILON = 0.01


    
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
            while j + horizontal_distance < x and state[i, j + horizontal_distance, 0] <= height:
                horizontal_distance += 1
            state[i, j, 6] = horizontal_distance

    return state

# 添加位置编码
def positional_encoding(seq_len, d_model):
    pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)  # 形状 [seq_len, 1]
    idx = torch.arange(d_model, dtype=torch.float).unsqueeze(0)  # 形状 [1, d_model]
    angle_rates = 1 / torch.pow(10000, (2 * (idx // 2)) / d_model)  # 计算角度
    angle_rads = pos * angle_rates  # 计算角度

    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])  # 偶数维度
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])  # 奇数维度

    return angle_rads

# 状态空间
class BoxEnvironment:
    def __init__(self, box_size, init_items):
        self.box_size = box_size
        self.init_items = init_items
        raw_state = np.zeros([self.box_size[0],self.box_size[1],7], dtype=int)
        self.state = calculate_attributes(raw_state, self.box_size[0], self.box_size[1]) # 初始化箱子状态
        self.items = init_items  # 存储未放入的物品

    def reset(self):
        raw_state = np.zeros([self.box_size[0],self.box_size[1],7], dtype=int)
        self.state = calculate_attributes(raw_state, self.box_size[0], self.box_size[1]) # 初始化箱子状态
        self.items = self.init_items.copy()  # 初始化物品列表
        return self.state

    def step(self, position, item, rotation):
        # 获取上一轮操作后的评分
        last_score = self.get_score()
        end_now = False

        # 获取物品的尺寸
        length, width, height = item
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

        action = dict()
        action['position'] = [position[0], position[1], self.state[position[0], position[1]][0]]
        action['item'] = item
        action['rotation'] = rotation

        out_of_range = False
        for p in range(0, length):
            if width > self.state[min(position[0] + p, self.box_size[0] - 1), position[1], 6]:
                out_of_range = True

        for q in range(0, width):
            if length > self.state[position[0], min(position[1] + q, self.box_size[1] - 1), 5]:
                out_of_range = True

        # 检查物品是否可以放入箱子
        if out_of_range:
            return self.state, -10, False, False, action  # 重叠或超出箱子边界，返回惩罚，继续

        # 放入物品
        current_values = self.state[position[0]: position[0] + length, position[1]: position[1] + width, 0]
        # 检查是否达到高度
        if (current_values + height).max() <= self.box_size[2]:
            # 更新高度
            self.state[position[0]: position[0] + length, position[1]: position[1] + width, 0] = height + self.state[position[0],position[1],0]
        else:
            end_now = True
            #return self.state, -10, False, False, action

        self.state = calculate_attributes(self.state, self.box_size[0], self.box_size[1]) # 更新箱子状态

        self.items.remove(item)  # 从剩余物品中移除已放入的物品
        if not self.items:
            end_now = True

        #前后分差为每一步奖励
        reward = last_score - self.get_score()

        if end_now:
            return self.state, reward, True, False, action  # 返回新的状态，奖励，结束
        return self.state, reward, False, True, action  # 返回新的状态，奖励，继续

    def get_score(self):
        # 计算当前装入的物体体积总和
        volume_used = np.sum(self.state[:, :, 0])  # 计算已装入的体积
        box_volume = np.prod(self.box_size)  # 箱子的总体积

        # 计算最大装入高度
        max_height = np.max(self.state[:, :, 0])

        # 分数 = 最大高度 * 箱子长 * 箱子宽 - 实际装入体积总和 = 浪费体积
        score = max_height * self.box_size[0] * self.box_size[1] - volume_used
        return score

# 策略网络
class TransformerPolicyNetwork(nn.Module):
    def __init__(self, d_model=128):
        super(TransformerPolicyNetwork, self).__init__()
        self.d_model = d_model

        self.axis_embedding = nn.Linear(1, d_model)
        self.state_embedding = nn.Linear(7, d_model)
        self.state_embedding = nn.Linear(7, d_model)
        self.rotation_embedding = nn.Linear(3, d_model)
        
        # 定义Transformer层
        self.transformer_box_encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=4)
        self.transformer_item_encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=4)
        self.transformer_layer_position = nn.TransformerDecoderLayer(d_model=d_model, nhead=4)
        self.transformer_layer_item = nn.TransformerDecoderLayer(d_model=d_model, nhead=4)
        self.transformer_layer_rotation = nn.TransformerDecoderLayer(d_model=d_model, nhead=4)
        
        # 输出层
        self.fc_position = nn.Linear(d_model, 1)  # 输出位置选择的概率
        self.fc_item = nn.Linear(d_model, 1)      # 输出物品选择的概率
        self.fc_rotation = nn.Linear(d_model, 1)   # 输出旋转选择的概率

    def forward(self, container_state, remaining_items):

        # 进行嵌入
        reshaped_container_state = torch.tensor(container_state, dtype=torch.float32).to(device).view(-1, 7)
        embedded_container_state = self.state_embedding(reshaped_container_state)
        pos_enc = positional_encoding(embedded_container_state.shape[0], self.d_model).to(device)
        positional_state_tensor = embedded_container_state + pos_enc
        
        # 编码箱子状态
        container_encoding = self.transformer_box_encoder(positional_state_tensor)

        # 编码每个物品的长、宽、高
        item_embeddings = []
        for item in remaining_items:
            # 假设 item 的形状为 (batch_size, item_length, 3)，分别代表长、宽、高
            length_encoded = self.axis_embedding(torch.tensor(item[0], dtype=torch.float32,device=device).unsqueeze(0))  # 编码长
            width_encoded = self.axis_embedding(torch.tensor(item[1], dtype=torch.float32,device=device).unsqueeze(0))   # 编码宽
            height_encoded = self.axis_embedding(torch.tensor(item[2], dtype=torch.float32,device=device).unsqueeze(0)) # 编码高
            # 对每个物品的编码进行平均
            embed_item = (length_encoded + width_encoded + height_encoded) / 3
            item_embeddings.append(embed_item)

        # 将所有物品合并编码
        remaining_items_encoded = self.transformer_item_encoder(torch.stack(item_embeddings))

        # 第一步：选择位置
        position_q = container_encoding
        position_kv = remaining_items_encoded
        position_encoded = self.transformer_layer_position(position_q, position_kv)
        position_probs = self.fc_position(position_encoded)
        position_probs = torch.softmax(position_probs, dim=0)
        # ε-贪婪策略选择位置
        if torch.rand(1).item() < EPSILON:  # 探索
            position_index = torch.randint(0, position_probs.size(0), (1,))
        else:  # 利用
            position_index = torch.multinomial(position_probs.view(-1), num_samples=1)
        position_embedding = container_encoding[position_index]

        # 第二步：选择物品
        item_q = remaining_items_encoded
        item_kv = position_embedding  # 使用位置的输出作为键和值
        item_encoded = self.transformer_layer_item(item_q, item_kv)
        item_probs = self.fc_item(item_encoded)
        item_probs = torch.softmax(item_probs, dim=0)

        # ε-贪婪策略选择物品
        if torch.rand(1).item() < EPSILON:  # 探索
            item_index = torch.randint(0, item_probs.size(0), (1,))
        else:  # 利用
            item_index = torch.multinomial(item_probs.view(-1), num_samples=1)

        length, width, height = remaining_items[item_index]
        rotation_items = [[length, width, height],[width, length, height],[height, width, length],[length, height, width],[width, height, length],[height, length, width]]
        rotation_items_embedding = self.rotation_embedding(torch.tensor(rotation_items, dtype=torch.float32,device=device))

        # 第三步：选择旋转
        rotation_q = rotation_items_embedding  # 使用物品选择的输出作为查询
        rotation_kv = position_encoded  # 位置选择的输出作为键和值
        rotation_encoded = self.transformer_layer_rotation(rotation_q, rotation_kv)
        rotation_probs = self.fc_rotation(rotation_encoded)
        rotation_probs = torch.softmax(rotation_probs, dim=0)

        # ε-贪婪策略选择旋转
        if torch.rand(1).item() < EPSILON:  # 探索
            rotation_index = torch.randint(0, rotation_probs.size(0), (1,))
        else:  # 利用
            rotation_index = torch.multinomial(rotation_probs.view(-1), num_samples=1)

        return position_index.item(), item_index.item(), rotation_index.item(), position_probs, item_probs, rotation_probs

class ValueNetwork(nn.Module):
    def __init__(self, input_size=7, hidden_size=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # 输出一个标量，表示状态值

    def forward(self, x):
        x = x.to(device)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)  # 输出状态值
        return value
    
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    使用 GAE 计算优势函数
    Args:
        rewards: 每一步的即时奖励 (tensor)
        values: 每一步的状态值函数估计 (tensor)
        gamma: 折扣因子
        lam: GAE 的衰减系数
    Returns:
        advantages: 每一步的优势函数估计 (tensor)
    """
    advantages = torch.zeros_like(rewards)  # 创建与 rewards 相同形状的张量，用来存储优势函数
    gae = 0  # 初始化 GAE
    for t in reversed(range(len(rewards))):
        # 计算 TD 残差 (delta)
        delta = rewards[t] + gamma * (values[t + 1] if t < len(rewards) - 1 else 0) - values[t]
        gae = delta + gamma * lam * gae  # 更新 GAE
        advantages[t] = gae  # 保存优势函数值

    return advantages

# 假设在每个 episode 开始时计算旧策略的对数概率
old_log_probs_list = []


def train_one_epoch(BOX_SIZE, ITEMS, epoch):
    env = BoxEnvironment(BOX_SIZE, ITEMS)
    policy_net = TransformerPolicyNetwork().to(device)
    value_net = ValueNetwork().to(device) # 假设你有一个值网络来估计状态的值
    optimizer = optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=LEARNING_RATE)
    if epoch > 1:
        policy_net.load_state_dict(torch.load(f'policy_net_epoch_{epoch - 1}.pth'))
        value_net.load_state_dict(torch.load(f'value_net_epoch_{epoch - 1}.pth'))
    epsilon = 0.2  # PPO裁剪范围
    beta = 0.01  # 熵正则化系数
    c1 = 1.0  # 值损失权重
    c2 = 0.01  # PPO策略损失权重

    for episode in tqdm(range(NUM_EPISODES), desc="Training Progress", unit="episode"):
        state = env.reset()
        done = False
        episode_loss = 0
        rewards = []
        values = []
        dones = []
        position_probs_list = []
        item_probs_list = []
        rotation_probs_list = []
        old_log_probs_list = []
        action_history = []

        while not done:
            remaining_items = env.items

            # 使用策略网络选择位置、物品和旋转
            position_index, item_index, rotation_index, position_probs, item_probs, rotation_probs = policy_net.forward(state, remaining_items)

            # 保存旧策略的对数概率
            old_log_probs = torch.log(position_probs[position_index]) + torch.log(item_probs[item_index]) + torch.log(rotation_probs[rotation_index])
            old_log_probs_list.append(old_log_probs)

            position = [position_index // env.box_size[1], position_index % env.box_size[1]]
            item = remaining_items[item_index]
            rotation = rotation_index

            legal = False
            # 执行动作
            new_state, reward, done, legal, action = env.step(position, item, rotation)
            if legal:
                action_history.append(action)

            # 保存当前的奖励、状态值和动作概率
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            value = value_net(state_tensor).squeeze()  # 当前状态的价值
            rewards.append(reward)
            values.append(value)
            dones.append(1.0 if done else 0.0)
            position_probs_list.append(position_probs)
            item_probs_list.append(item_probs)
            rotation_probs_list.append(rotation_probs)
            del position_probs, item_probs, rotation_probs
            #中间结果用完后立即释放
            torch.cuda.empty_cache()
            state = new_state  # 更新状态
        
        if episode == NUM_EPISODES-1:
            save_model(policy_net, value_net, epoch)
            #plot_items(action_history)

        # 计算GAE优势函数
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32,device=device)
        values_flattened = torch.cat([v.flatten() for v in values])

        # 计算 GAE 优势函数
        advantages = compute_gae(rewards_tensor, values_flattened)

        # 更新策略和价值网络
        loss = torch.tensor(0.0, dtype=torch.float32,device=device)
        for t in range(len(rewards)):
            # 使用正确的动作索引
            position_probs = position_probs_list[t]
            item_probs = item_probs_list[t]
            rotation_probs = rotation_probs_list[t]

            # 从概率分布中选择索引（假设你想选择最大概率的动作）
            position_index = torch.argmax(position_probs)  # 从 position_probs 选择最大值的索引
            item_index = torch.argmax(item_probs)          # 从 item_probs 选择最大值的索引
            rotation_index = torch.argmax(rotation_probs)  # 从 rotation_probs 选择最大值的索引

            # 计算当前策略的对数概率
            position_log_prob = torch.log(position_probs[position_index])
            item_log_prob = torch.log(item_probs[item_index])
            rotation_log_prob = torch.log(rotation_probs[rotation_index])

            # 确保所有对数概率是标量，然后相加
            new_log_probs = position_log_prob + item_log_prob + rotation_log_prob

            # 计算重要性采样比率r_t(θ)
            ratio = torch.exp(new_log_probs - old_log_probs_list[t])

            # 计算PPO的裁剪策略损失
            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            policy_loss = -torch.min(ratio * advantages[t], clipped_ratio * advantages[t]).mean()

            # 计算价值损失
            #value_loss = F.mse_loss(value, rewards_tensor[t])
            #print("Value shape:", value.shape)
            #print("A shape:", advantages[t])
            #print("Reward tensor shape:", rewards_tensor[t].shape)
            # 将优势值扩展为与 value 相同的形状
            expanded_advantages = advantages[t].expand_as(value)

            # 计算目标值
            target = expanded_advantages + value

            # 计算 MSE 损失
            value_loss = F.mse_loss(value, target)
            # 熵正则化
            entropy_position = -(position_probs * torch.log(position_probs + 1e-10)).mean()
            entropy_item = -(item_probs * torch.log(item_probs + 1e-10)).mean()
            entropy_rotation = -(rotation_probs * torch.log(rotation_probs + 1e-10)).mean()

            entropy_loss = entropy_position + entropy_item + entropy_rotation

            # 总损失
            total_loss = policy_loss + c1 * value_loss - beta * entropy_loss
            loss += total_loss

            # 累加每个时间步的损失
            episode_loss += total_loss.item()
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Episode {episode + 1}/{NUM_EPISODES}, Loss: {episode_loss:.4f}")
        del rewards,values,dones,position_probs_list,item_probs_list,rotation_probs_list,old_log_probs_list,action_history
        torch.cuda.empty_cache()


def save_model(policy_net, value_net, epoch):
    # 保存模型状态字典
    torch.save(policy_net.state_dict(), f"policy_net_epoch_{epoch}.pth")
    torch.save(value_net.state_dict(), f"value_net_epoch_{epoch}.pth")
    print(f"Models saved at epoch {epoch}.")
    
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
    torch.autograd.set_detect_anomaly(True)
    epoch_num = 0
    torch.cuda.empty_cache()
    for sample in train_data:
        epoch_num +=1
        box_size = sample['box_size']
        items = sample['items']
        train_one_epoch(box_size,items,epoch_num)
        torch.cuda.empty_cache()
        '''
    for sample in test_data:
        box_size = sample['box_size']
        items = sample['items']
        train_one_epoch(box_size,items,11)
    '''