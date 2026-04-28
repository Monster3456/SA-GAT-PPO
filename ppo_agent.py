
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


class DynamicGraphTransformerBlock(nn.Module):
    """
    大模型级联基座：融合动态Query/Key与残差连接的 Graph Transformer 块
    用于在 240Hz 快衰落下动态捕捉、对齐不确定性的干扰拓扑
    """

    def __init__(self, embed_dim=128, num_heads=4):
        super(DynamicGraphTransformerBlock, self).__init__()
        self.embed_dim = embed_dim

        # 1. 大模型基石：LayerNormalization 放在 Attention 前面 (Pre-LN 结构更稳定)
        self.layernorm1 = nn.LayerNorm(embed_dim)

        # 2. 核心大杀器：用于生成输入依赖（动态）Q, K 的 MLP
        # 允许网络根据当前的 SNR 和误差界动态改变“我看谁”的角度
        self.dynamic_qk = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim * 2)  # 输出 Q 和 K 的投影向量
        )

        # 用于 Value (V) 的标准静态投影 (V 是内容，不需要总是动态改变)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 标准的多头注意力内核 (它接收动态投影出的 Q, K)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        # 3. 级联 MLP + LN：增强非线性处理能力
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, x):
        batch_size, K, _ = x.shape

        # ==========================================
        # 核心一：输入依赖的动态 Q, K 生成 (Dynamic Attention)
        # ==========================================
        # 先归一化输入
        norm_x = self.layernorm1(x)

        # 根据 norm_x 动态生成 Q 和 K 的投影矩阵
        # (这步极大地增强了网络应对信道剧变和非完美 CSI 的不确定性的能力)
        qk_proj = self.dynamic_qk(norm_x)
        dynamic_q = qk_proj[:, :, :self.embed_dim]
        dynamic_k = qk_proj[:, :, self.embed_dim:]

        # 静态投影生成 V (Value)
        static_v = self.v_proj(norm_x)

        # 放入 Attention 内核计算
        attn_out, _ = self.attention(dynamic_q, dynamic_k, static_v)

        # ------------------------------------------
        # 核心二：残差连接 (与大模型结构完全对齐)
        # 将 Attention 输出加回原始输入 x (保证深层网络不退化)
        # ------------------------------------------
        x = x + attn_out

        # ------------------------------------------
        # 核心三：级联 MLP 与 LN (FeedForward 块)
        # ------------------------------------------
        norm_x2 = self.layernorm2(x)
        ff_out = self.feedforward(norm_x2)
        x = x + ff_out  # 第二次残差连接

        return x


class HeavyweightDynamicAttentionActorCritic(nn.Module):
    """
    进化版单智能体 PPO：真正具备“置换不变性”的 Graph Transformer 架构
    完美解决了展平(Flatten)带来的拓扑失真问题
    """

    def __init__(self, K, Nt, Nr, action_dim):
        super(HeavyweightDynamicAttentionActorCritic, self).__init__()
        self.K = K
        self.feature_dim = 2 * Nr * Nt + 2
        self.embed_dim = 128
        self.action_dim_per_user = action_dim // K  # 每个用户需要 5 个动作维度

        self.node_embed = nn.Sequential(
            nn.Linear(self.feature_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU()
        )

        self.graph_transformer_block = DynamicGraphTransformerBlock(self.embed_dim, num_heads=4)

        # ==========================================
        # 核心改动 1：具有等变性 (Equivariance) 的 Actor 头
        # 不再是一个巨大的全连接层，而是一个处理单个节点的轻量级网络
        # ==========================================
        self.actor_head = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim_per_user),
            nn.Tanh()
        )
        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), -0.5))

        # ==========================================
        # 核心改动 2：具有置换不变性 (Invariance) 的 Critic
        # 输入维度降回 embed_dim，不再是 embed_dim * K
        # ==========================================
        self.critic = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def _extract_graph_features(self, state):
        batch_size = state.shape[0]
        nodes = state.view(batch_size, self.K, self.feature_dim)
        embedded_nodes = self.node_embed(nodes)
        transformed_nodes = self.graph_transformer_block(embedded_nodes)

        # 直接返回未展平的图节点特征 [batch_size, K, embed_dim]
        return transformed_nodes

    def forward(self, state):
        batch_size = state.shape[0]
        # 获取图节点特征：[batch_size, K, embed_dim]
        transformed_nodes = self._extract_graph_features(state)

        # ------------------------------------------
        # Critic 前向传播：使用全局平均池化 (Mean Pooling)
        # ------------------------------------------
        # 沿着 K 维度取平均，得到 [batch_size, embed_dim]
        # 这样无论用户顺序怎么打乱，求和平均的结果绝对一致！
        global_feature = transformed_nodes.mean(dim=1)
        value = self.critic(global_feature)

        # ------------------------------------------
        # Actor 前向传播：独立处理与转置对齐
        # ------------------------------------------
        # 1. 独立计算每个节点的动作 -> [batch_size, K, 5]
        node_actions = self.actor_head(transformed_nodes)

        # 2. 张量转置 -> [batch_size, 5, K]
        # 这一步极其关键！转置后再 view，输出的顺序就会变成：
        # [所有用户的动作1, 所有用户的动作2...], 完美对齐 rsma_env.py 的解包逻辑
        aligned_actions = node_actions.transpose(1, 2).contiguous()

        # 3. 展平为最终的一维动作向量 -> [batch_size, 5 * K]
        mean = aligned_actions.view(batch_size, -1)

        std = self.actor_logstd.exp().expand_as(mean)
        dist = Normal(mean, std)

        return dist, value


class PureMLPActorCritic(nn.Module):
    """
    纯 MLP 网络 (消融实验基线)
    剥离了图注意力机制和动态 Query/Key 设计
    直接将所有用户的状态特征展平后丢入全连接层
    """

    def __init__(self, K, Nt, Nr, action_dim):
        super(PureMLPActorCritic, self).__init__()
        self.K = K
        self.feature_dim = 2 * Nr * Nt + 2

        # 直接展平后的总输入维度
        input_dim = self.K * self.feature_dim

        # 虽然去掉了 Transformer，但为了公平对比异质 SNR 的泛化能力，
        # 我们依然保留 LayerNorm 来稳定梯度。
        self.actor_mean = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), -0.5))

        self.critic = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )


    def forward(self, state):
        batch_size = state.shape[0]
        # 不做任何节点特征提取，直接展平 (Flatten)
        feat = state.view(batch_size, -1)

        value = self.critic(feat)
        mean = self.actor_mean(feat)
        std = self.actor_logstd.exp().expand_as(mean)
        dist = Normal(mean, std)
        return dist, value


class PPOAgent:
    # 核心修改：增加 model_type 参数，用于灵活切换网络架构
    def __init__(self, K, Nt, Nr, action_dim, lr=1e-4, gamma=0.99, clip_ratio=0.2, model_type='transformer'):
        self.model_type = model_type

        # 根据传入的参数实例化不同的网络
        if model_type == 'transformer':
            self.policy = HeavyweightDynamicAttentionActorCritic(K, Nt, Nr, action_dim)
        elif model_type == 'mlp':
            self.policy = PureMLPActorCritic(K, Nt, Nr, action_dim)
        else:
            raise ValueError("Unsupported model_type! Use 'transformer' or 'mlp'.")

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio


    def get_action(self, state):
        state_ts = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            dist, value = self.policy(state_ts)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action.squeeze(0).numpy(), log_prob.item(), value.item()

    # 替换 ppo_agent.py 中 PPOAgent 类的 update 函数
    def update(self, rollouts, ent_coef=0.01, batch_size=256):
        states = torch.FloatTensor(np.array(rollouts['states']))
        actions = torch.FloatTensor(np.array(rollouts['actions']))
        old_log_probs = torch.FloatTensor(np.array(rollouts['log_probs']))
        returns = torch.FloatTensor(np.array(rollouts['returns']))
        advantages = torch.FloatTensor(np.array(rollouts['advantages']))

        # 全局优势归一化，稳定方差
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = states.size(0)
        indices = np.arange(dataset_size)

        for _ in range(10):  # 10 个 Epochs
            np.random.shuffle(indices)

            # 引入 Mini-batch 机制，打碎数据增加更新频次，极大提升高维空间收敛速度
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_idx = indices[start_idx:end_idx]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                dist, values = self.policy(batch_states)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.MSELoss()(values.squeeze(-1), batch_returns)

                # 核心修复：使用外部传入的动态 ent_coef，而不是写死的 0.01
                loss = actor_loss + 0.5 * critic_loss - ent_coef * dist.entropy().mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()