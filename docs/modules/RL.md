# RL (Reinforcement Learning) 模組詳解

## 目錄

1. [什麼是強化學習？](#什麼是強化學習)
2. [為什麼需要強化學習？](#為什麼需要強化學習)
3. [PPO 算法詳解](#ppo-算法詳解)
4. [辯論環境設計](#辯論環境設計)
5. [訓練過程](#訓練過程)
6. [網路架構](#網路架構)
7. [獎勵機制](#獎勵機制)
8. [實際應用](#實際應用)
9. [程式碼範例](#程式碼範例)
10. [常見問題](#常見問題)

## 什麼是強化學習？

### 基本概念

想像你在學習下棋：
- 你嘗試不同的走法（**動作**）
- 看到棋盤的變化（**狀態**）
- 贏棋得分，輸棋扣分（**獎勵**）
- 慢慢學會什麼時候該怎麼走（**策略**）

**強化學習（RL）**就是讓 AI 通過不斷嘗試和獲得反饋來學習最佳策略。

### 核心要素

```python
# RL 的基本循環
while not done:
    action = agent.choose_action(state)  # 根據當前狀態選擇動作
    next_state, reward = env.step(action)  # 執行動作，獲得獎勵
    agent.learn(state, action, reward, next_state)  # 從經驗中學習
    state = next_state
```

### 與其他學習方法的區別

| 學習方法 | 特點 | 例子 |
|---------|------|------|
| 監督學習 | 有標準答案 | 看圖識貓 |
| 無監督學習 | 找規律 | 客戶分群 |
| 強化學習 | 試錯學習 | 學習下棋 |

## 為什麼需要強化學習？

### 辯論的動態特性

辯論不是靜態的問答，而是動態的博弈：
1. **對手會反應**：你的策略會影響對手的回應
2. **長期效果**：當前的論點可能在後續產生影響
3. **策略適應**：需要根據對手風格調整策略

### RL 的優勢

```python
# 傳統方法：固定策略
if opponent_aggressive:
    return "defensive"
else:
    return "analytical"

# RL 方法：學習最優策略
state = encode_debate_state(history, opponent_style, topic)
action = policy_network(state)  # 根據學習到的策略選擇
```

## PPO 算法詳解

### 什麼是 PPO？

**PPO (Proximal Policy Optimization)** 是一種先進的策略梯度算法，特點是：
- **穩定**：避免策略更新過大
- **高效**：樣本利用率高
- **簡單**：實現相對簡單

### PPO 的核心思想

想像你在學習騎腳踏車：
1. **不要改變太快**：每次只做小調整，避免摔倒
2. **記住好的經驗**：成功的動作要多練習
3. **限制更新幅度**：確保新策略不會偏離太遠

### PPO 的數學原理（簡化版）

```python
# 傳統策略梯度
loss = -log(π(a|s)) * advantage

# PPO 的改進：加入限制
ratio = π_new(a|s) / π_old(a|s)
loss = -min(
    ratio * advantage,
    clip(ratio, 1-ε, 1+ε) * advantage
)
```

其中：
- `π(a|s)`：在狀態 s 下選擇動作 a 的概率
- `advantage`：這個動作比平均好多少
- `clip`：限制更新幅度在 [1-ε, 1+ε] 範圍內

## 辯論環境設計

### 環境組成

```python
class DebateEnvironment:
    def __init__(self):
        self.state = {
            'round': 0,              # 當前回合
            'history': [],           # 對話歷史
            'stances': {},          # 各方立場
            'persuasion_scores': {} # 說服分數
        }
```

### 狀態空間（State Space）

狀態包含辯論的所有相關信息：

```python
state = {
    # 1. 文本特徵（768維 BERT 嵌入）
    'text_embedding': [...],
    
    # 2. 立場信息（3維）
    'my_stance': 0.8,        # 我的立場強度
    'opponent_stance': -0.6, # 對手立場強度
    'stance_distance': 1.4,  # 立場差距
    
    # 3. 歷史信息（10維）
    'round_number': 3,       # 當前回合
    'my_wins': 2,           # 我贏的回合數
    'opponent_wins': 1,     # 對手贏的回合數
    'momentum': 0.3,        # 動量（誰佔優勢）
    
    # 4. 策略使用歷史（4維）
    'strategy_counts': [2, 1, 0, 0]  # 各策略使用次數
}
```

### 動作空間（Action Space）

四種辯論策略：

```python
actions = {
    0: 'aggressive',   # 攻擊型：直接反駁，挑戰對方
    1: 'defensive',    # 防守型：鞏固立場，回應質疑
    2: 'analytical',   # 分析型：理性分析，提供證據
    3: 'empathetic'    # 同理型：理解對方，尋求共識
}
```

### 狀態轉移

```python
def step(self, action):
    # 1. 根據動作生成回應
    response = generate_response(action, self.state)
    
    # 2. 對手回應
    opponent_response = opponent.respond(response)
    
    # 3. 更新狀態
    self.state['history'].append({
        'agent': response,
        'opponent': opponent_response
    })
    self.state['round'] += 1
    
    # 4. 計算立場變化
    stance_change = calculate_stance_change(response, opponent_response)
    self.state['stances'] = update_stances(stance_change)
    
    # 5. 計算獎勵
    reward = calculate_reward(stance_change, action)
    
    return self.state, reward, done
```

## 訓練過程

### 1. 數據收集

```python
def collect_experience(env, policy, num_episodes):
    experiences = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_data = []
        
        while not done:
            # 選擇動作
            action, log_prob = policy.select_action(state)
            
            # 執行動作
            next_state, reward, done = env.step(action)
            
            # 儲存經驗
            episode_data.append({
                'state': state,
                'action': action,
                'reward': reward,
                'log_prob': log_prob
            })
            
            state = next_state
        
        experiences.extend(episode_data)
    
    return experiences
```

### 2. 計算優勢（Advantage）

優勢函數告訴我們一個動作比平均好多少：

```python
def compute_advantages(rewards, values, gamma=0.99, lambda_=0.95):
    advantages = []
    advantage = 0
    
    for t in reversed(range(len(rewards))):
        # TD 誤差
        td_error = rewards[t] + gamma * values[t+1] - values[t]
        
        # GAE (Generalized Advantage Estimation)
        advantage = td_error + gamma * lambda_ * advantage
        advantages.insert(0, advantage)
    
    return advantages
```

### 3. PPO 更新

```python
def ppo_update(policy, old_policy, experiences, epochs=10):
    for epoch in range(epochs):
        for batch in create_batches(experiences):
            # 計算新舊策略的比率
            ratio = policy.log_prob(batch) / old_policy.log_prob(batch)
            
            # PPO 的 clipped objective
            surr1 = ratio * batch.advantages
            surr2 = torch.clamp(ratio, 1-eps, 1+eps) * batch.advantages
            
            # 策略損失
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 價值損失
            value_loss = F.mse_loss(policy.value(batch.states), batch.returns)
            
            # 總損失
            loss = policy_loss + 0.5 * value_loss
            
            # 更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### 4. 訓練流程

```python
def train_ppo():
    env = DebateEnvironment()
    policy = PPONetwork()
    
    for iteration in range(1000):
        # 1. 收集經驗
        experiences = collect_experience(env, policy, num_episodes=10)
        
        # 2. 計算優勢和回報
        advantages = compute_advantages(experiences)
        returns = compute_returns(experiences)
        
        # 3. PPO 更新
        old_policy = copy.deepcopy(policy)
        ppo_update(policy, old_policy, experiences)
        
        # 4. 評估
        if iteration % 10 == 0:
            eval_reward = evaluate_policy(policy, env)
            print(f"Iteration {iteration}: Reward = {eval_reward}")
```

## 網路架構

### PPO 網路結構

```python
class PPONetwork(nn.Module):
    def __init__(self, state_dim=785, action_dim=4):
        super().__init__()
        
        # 共享層
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 策略頭（Actor）
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # 價值頭（Critic）
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        # 共享特徵
        features = self.shared(state)
        
        # 策略輸出（動作概率）
        action_probs = self.policy_head(features)
        
        # 價值輸出（狀態價值）
        state_value = self.value_head(features)
        
        return action_probs, state_value
```

### 網路設計考量

1. **共享層**：Actor 和 Critic 共享底層特徵
2. **Dropout**：防止過擬合
3. **適當的寬度**：512→256→128 逐層縮減
4. **激活函數**：ReLU 提供非線性

## 獎勵機制

### 獎勵設計原則

好的獎勵函數應該：
1. **引導正確行為**：鼓勵有效的辯論策略
2. **避免退化**：防止重複或無意義的回應
3. **平衡多目標**：說服力、多樣性、質量

### 獎勵組成

```python
def calculate_reward(state, action, next_state):
    reward = 0
    
    # 1. 說服獎勵（主要目標）
    stance_change = next_state['opponent_stance'] - state['opponent_stance']
    if stance_change > 0:  # 對手向我方立場移動
        reward += stance_change * 10
    
    # 2. 策略效果獎勵
    strategy_effectiveness = {
        'aggressive': 0.8 if opponent_defensive else 0.3,
        'defensive': 0.7 if opponent_aggressive else 0.4,
        'analytical': 0.9 if topic_complex else 0.5,
        'empathetic': 0.8 if opponent_emotional else 0.4
    }
    reward += strategy_effectiveness[action]
    
    # 3. 多樣性獎勵（避免重複使用同一策略）
    strategy_count = state['strategy_counts'][action]
    diversity_bonus = 1.0 / (1 + strategy_count)
    reward += diversity_bonus * 0.5
    
    # 4. 回合懲罰（鼓勵快速說服）
    reward -= 0.1  # 每回合小懲罰
    
    # 5. 勝利獎勵
    if next_state['opponent_stance'] * state['my_stance'] > 0:
        reward += 20  # 成功說服
    
    return reward
```

### 獎勵塑形（Reward Shaping）

為了加速學習，提供中間獎勵：

```python
# 基礎獎勵
base_reward = calculate_base_reward()

# 塑形獎勵（引導但不改變最優策略）
shaping_reward = 0

# 鼓勵使用證據
if uses_evidence(response):
    shaping_reward += 0.2

# 鼓勵回應對方論點
if addresses_opponent_points(response):
    shaping_reward += 0.3

# 懲罰過於激進
if too_aggressive(response):
    shaping_reward -= 0.5

total_reward = base_reward + shaping_reward
```

## 實際應用

### 1. 策略選擇

```python
def select_debate_strategy(debate_state):
    # 將辯論狀態編碼
    state_vector = encode_state(debate_state)
    
    # 使用訓練好的 PPO 網路
    with torch.no_grad():
        action_probs, _ = ppo_network(state_vector)
    
    # 選擇策略
    if training:
        # 訓練時：根據概率採樣
        action = torch.multinomial(action_probs, 1).item()
    else:
        # 推理時：選擇最佳策略
        action = torch.argmax(action_probs).item()
    
    return STRATEGIES[action]
```

### 2. 自適應學習

```python
class AdaptiveDebater:
    def __init__(self):
        self.ppo_network = load_pretrained_ppo()
        self.experience_buffer = []
    
    def debate(self, topic, opponent):
        state = self.initialize_debate(topic)
        
        while not self.debate_ended():
            # 使用 PPO 選擇策略
            strategy = self.select_strategy(state)
            
            # 生成回應
            response = self.generate_response(strategy, state)
            
            # 獲得對手回應
            opponent_response = opponent.respond(response)
            
            # 更新狀態
            next_state = self.update_state(state, response, opponent_response)
            
            # 計算獎勵
            reward = self.calculate_reward(state, strategy, next_state)
            
            # 儲存經驗（用於後續學習）
            self.experience_buffer.append({
                'state': state,
                'action': strategy,
                'reward': reward,
                'next_state': next_state
            })
            
            state = next_state
        
        # 辯論結束後可以繼續學習
        if self.should_update():
            self.update_policy()
```

### 3. 策略組合

```python
def get_mixed_strategy(state, temperature=1.0):
    """
    根據狀態返回混合策略（概率分布）
    temperature 控制探索程度
    """
    with torch.no_grad():
        logits, _ = ppo_network(state)
        
        # 溫度調節
        scaled_logits = logits / temperature
        
        # 轉換為概率
        probs = F.softmax(scaled_logits, dim=-1)
    
    return {
        'aggressive': probs[0].item(),
        'defensive': probs[1].item(),
        'analytical': probs[2].item(),
        'empathetic': probs[3].item()
    }
```

## 程式碼範例

### 完整的訓練腳本

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class DebateEnvironment:
    """辯論環境"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.round = 0
        self.history = []
        self.agent_stance = np.random.uniform(0.5, 1.0)
        self.opponent_stance = np.random.uniform(-1.0, -0.5)
        return self._get_state()
    
    def _get_state(self):
        # 編碼當前狀態
        text_features = self._encode_history()
        stance_features = [
            self.agent_stance,
            self.opponent_stance,
            self.agent_stance - self.opponent_stance
        ]
        round_features = [self.round / 10]
        
        return np.concatenate([
            text_features,
            stance_features,
            round_features
        ])
    
    def step(self, action):
        # 執行動作
        self.round += 1
        
        # 模擬對手反應
        opponent_reaction = self._simulate_opponent(action)
        
        # 更新立場
        stance_change = self._calculate_stance_change(action, opponent_reaction)
        self.opponent_stance += stance_change
        
        # 計算獎勵
        reward = self._calculate_reward(stance_change, action)
        
        # 檢查結束條件
        done = (
            self.round >= 10 or 
            abs(self.opponent_stance) < 0.1 or
            self.opponent_stance * self.agent_stance > 0
        )
        
        return self._get_state(), reward, done, {}

class PPOTrainer:
    """PPO 訓練器"""
    def __init__(self, state_dim, action_dim):
        self.policy = PPONetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        
        # 超參數
        self.gamma = 0.99
        self.lambda_ = 0.95
        self.epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
    
    def train(self, env, num_iterations=1000):
        for iteration in range(num_iterations):
            # 收集軌跡
            trajectories = self.collect_trajectories(env, num_episodes=32)
            
            # 計算優勢
            self.compute_advantages(trajectories)
            
            # PPO 更新
            for epoch in range(10):
                self.ppo_update(trajectories)
            
            # 記錄進度
            if iteration % 10 == 0:
                avg_reward = np.mean([t['total_reward'] for t in trajectories])
                print(f"Iteration {iteration}: Avg Reward = {avg_reward:.2f}")
    
    def collect_trajectories(self, env, num_episodes):
        trajectories = []
        
        for _ in range(num_episodes):
            trajectory = {
                'states': [],
                'actions': [],
                'rewards': [],
                'log_probs': [],
                'values': [],
                'total_reward': 0
            }
            
            state = env.reset()
            done = False
            
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                # 獲取動作和價值
                with torch.no_grad():
                    action_probs, value = self.policy(state_tensor)
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                
                # 執行動作
                next_state, reward, done, _ = env.step(action.item())
                
                # 儲存數據
                trajectory['states'].append(state)
                trajectory['actions'].append(action.item())
                trajectory['rewards'].append(reward)
                trajectory['log_probs'].append(log_prob.item())
                trajectory['values'].append(value.item())
                trajectory['total_reward'] += reward
                
                state = next_state
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def compute_advantages(self, trajectories):
        for trajectory in trajectories:
            rewards = trajectory['rewards']
            values = trajectory['values']
            
            # 計算回報和優勢
            returns = []
            advantages = []
            
            G = 0
            A = 0
            
            for t in reversed(range(len(rewards))):
                G = rewards[t] + self.gamma * G
                returns.insert(0, G)
                
                if t < len(rewards) - 1:
                    td_error = rewards[t] + self.gamma * values[t+1] - values[t]
                else:
                    td_error = rewards[t] - values[t]
                
                A = td_error + self.gamma * self.lambda_ * A
                advantages.insert(0, A)
            
            trajectory['returns'] = returns
            trajectory['advantages'] = advantages
    
    def ppo_update(self, trajectories):
        # 準備批次數據
        states = []
        actions = []
        old_log_probs = []
        returns = []
        advantages = []
        
        for trajectory in trajectories:
            states.extend(trajectory['states'])
            actions.extend(trajectory['actions'])
            old_log_probs.extend(trajectory['log_probs'])
            returns.extend(trajectory['returns'])
            advantages.extend(trajectory['advantages'])
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # 標準化優勢
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 計算新的動作概率和價值
        action_probs, values = self.policy(states)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        
        # 計算比率
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO 損失
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 價值損失
        value_loss = self.value_coef * F.mse_loss(values.squeeze(), returns)
        
        # 熵獎勵（鼓勵探索）
        entropy = -self.entropy_coef * dist.entropy().mean()
        
        # 總損失
        loss = policy_loss + value_loss + entropy
        
        # 更新
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

# 主訓練程式
if __name__ == "__main__":
    # 創建環境和訓練器
    env = DebateEnvironment()
    state_dim = 785  # 768 (BERT) + 其他特徵
    action_dim = 4   # 4種策略
    
    trainer = PPOTrainer(state_dim, action_dim)
    
    # 訓練
    trainer.train(env, num_iterations=1000)
    
    # 保存模型
    torch.save(trainer.policy.state_dict(), 'ppo_debate_policy.pt')
    
    # 測試
    test_policy(trainer.policy, env)
```

### 推理使用範例

```python
# 載入訓練好的模型
policy = PPONetwork(state_dim=785, action_dim=4)
policy.load_state_dict(torch.load('ppo_debate_policy.pt'))
policy.eval()

def get_debate_strategy(debate_context):
    """
    根據辯論上下文選擇最佳策略
    """
    # 1. 編碼狀態
    state = encode_debate_state(debate_context)
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    
    # 2. 獲取策略分布
    with torch.no_grad():
        action_probs, state_value = policy(state_tensor)
    
    # 3. 選擇策略
    strategies = ['aggressive', 'defensive', 'analytical', 'empathetic']
    
    # 貪婪選擇（部署時）
    best_action = torch.argmax(action_probs).item()
    best_strategy = strategies[best_action]
    
    # 或者根據概率分布（保持一定隨機性）
    dist = torch.distributions.Categorical(action_probs)
    sampled_action = dist.sample().item()
    sampled_strategy = strategies[sampled_action]
    
    return {
        'best_strategy': best_strategy,
        'strategy_distribution': {
            strategies[i]: prob.item() 
            for i, prob in enumerate(action_probs[0])
        },
        'expected_value': state_value.item()
    }

# 實際使用
debate_context = {
    'topic': '是否應該徵收碳稅',
    'my_stance': '支持',
    'opponent_stance': '反對',
    'round': 3,
    'history': [...]
}

result = get_debate_strategy(debate_context)
print(f"建議策略: {result['best_strategy']}")
print(f"策略分布: {result['strategy_distribution']}")
print(f"預期價值: {result['expected_value']:.2f}")
```

## 常見問題

### Q1: 為什麼選擇 PPO 而不是其他 RL 算法？

**A**: PPO 的優勢：
- **穩定性**：相比 TRPO 更簡單，相比 A3C 更穩定
- **樣本效率**：比 DQN 等 off-policy 方法更高效
- **實現簡單**：不需要複雜的 trust region 計算
- **性能好**：在許多任務上達到 SOTA

### Q2: 如何處理文本輸入？

**A**: 
1. 使用預訓練語言模型（如 BERT）編碼文本
2. 提取固定維度的特徵向量（768維）
3. 結合其他數值特徵（立場、回合等）
4. 輸入到 PPO 網路

### Q3: 訓練需要多長時間？

**A**: 
- **小規模**：100 個 episodes，約 30 分鐘
- **中規模**：1000 個 episodes，約 5 小時
- **大規模**：10000 個 episodes，約 2 天
- 使用 GPU 可以顯著加速

### Q4: 如何避免策略退化？

**A**: 
1. **熵正則化**：鼓勵探索，避免過早收斂
2. **多樣性獎勵**：懲罰重複使用同一策略
3. **對手池**：訓練時使用多種對手
4. **課程學習**：從簡單到困難逐步訓練

### Q5: 如何調試 RL 訓練？

**A**: 
1. **監控指標**：
   - 平均獎勵
   - 策略熵
   - 價值損失
   - KL 散度

2. **視覺化**：
   - 獎勵曲線
   - 策略分布變化
   - 動作選擇熱圖

3. **診斷問題**：
   - 獎勵不增長 → 檢查獎勵函數
   - 策略崩潰 → 降低學習率
   - 高方差 → 增加批次大小

### Q6: PPO 的局限性？

**A**: 
- **樣本需求**：仍需要大量交互
- **超參數敏感**：需要仔細調參
- **局部最優**：可能陷入次優策略
- **計算成本**：訓練成本較高

## 總結

PPO 在辯論系統中的作用：
1. **動態適應**：根據對手調整策略
2. **長期規劃**：考慮整場辯論的勝利
3. **策略學習**：從經驗中學習最優策略
4. **靈活決策**：平衡多個目標

通過強化學習，我們讓 AI 不僅知道「怎麼說」，更學會「何時說什麼」，真正掌握辯論的藝術。 