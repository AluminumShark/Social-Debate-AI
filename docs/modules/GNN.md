# GNN (Graph Neural Network) 模組詳解

## 目錄

1. [什麼是 GNN？](#什麼是-gnn)
2. [為什麼需要 GNN？](#為什麼需要-gnn)
3. [訓練資料詳解](#訓練資料詳解)
4. [模型架構](#模型架構)
5. [訓練過程](#訓練過程)
6. [推理過程](#推理過程)
7. [實際應用](#實際應用)
8. [程式碼範例](#程式碼範例)
9. [常見問題](#常見問題)

## 什麼是 GNN？

### 基本概念

想像你在社交網路上，每個人（節點）都和其他人（節點）有連接（邊）。**圖神經網路（GNN）**就是專門處理這種網路結構數據的深度學習模型。

```
    A --- B
    |     |
    C --- D
```

在辯論系統中：
- **節點**：代表參與者或評論
- **邊**：代表互動關係（回覆、引用、說服）
- **特徵**：每個節點的內容和屬性

### 為什麼不用普通神經網路？

普通神經網路只看單一輸入，但 GNN 能理解關係：
- **普通 NN**：只看評論內容 → 判斷好壞
- **GNN**：看評論內容 + 誰說的 + 對誰說 + 上下文 → 判斷說服效果

## 為什麼需要 GNN？

### 辯論的社交特性

辯論不是孤立的，而是互動的過程：
1. **社會影響力**：有些人天生更有說服力
2. **關係網路**：誰回覆誰很重要
3. **上下文依賴**：同樣的話在不同情境效果不同

### GNN 的優勢

```python
# 普通方法
說服力 = f(評論內容)

# GNN 方法
說服力 = f(評論內容, 發言者特徵, 對象特徵, 互動歷史)
```

## 訓練資料詳解

### 資料來源：Reddit ChangeMyView (CMV)

CMV 是一個專門用於改變觀點的討論版，特點：
- 用戶發表觀點，其他人嘗試改變其想法
- 如果成功被說服，會頒發 "Delta" (Δ) 標記
- 提供了天然的「說服成功」標籤

### 資料格式

```json
{
  "comment_id": "abc123",
  "text": "我認為你的觀點忽略了一個重要因素...",
  "author": "user456",
  "parent_id": "xyz789",
  "delta_awarded": true,  // 是否成功說服
  "score": 42,            // 社群評分
  "created_utc": 1634567890
}
```

### 資料預處理

1. **文本編碼**：使用 BERT 將文本轉為 768 維向量
2. **特徵提取**：
   - 論證強度（evidence markers 數量）
   - 文本長度
   - 情感分數
3. **圖構建**：
   - 節點：每個評論
   - 邊：回覆關係

```python
# 構建圖的過程
nodes = []
edges = []

for comment in comments:
    # 創建節點
    node_features = bert_encode(comment.text)  # 768維
    nodes.append(node_features)
    
    # 創建邊（如果有回覆關係）
    if comment.parent_id:
        edges.append([comment.id, comment.parent_id])
```

### 標籤定義

我們的 GNN 進行多任務學習，有三種標籤：

1. **Delta 標籤**（二分類）
   - 1：成功說服（獲得 Delta）
   - 0：未成功說服

2. **品質分數**（回歸）
   - 基於社群評分和互動度
   - 範圍：0-100

3. **策略標籤**（多分類）
   - 0：aggressive（攻擊型）
   - 1：defensive（防守型）
   - 2：analytical（分析型）
   - 3：empathetic（同理型）

## 模型架構

### 1. 輸入層（Input Layer）

```python
輸入維度 = 770
= 768 (BERT 文本嵌入) + 2 (論證特徵)
```

每個節點的特徵包含：
- **文本嵌入**：BERT 編碼的語義信息
- **論證特徵**：
  - 證據強度（0-1）
  - 文本長度（標準化）

### 2. 圖卷積層（Graph Convolution）

使用 **GraphSAGE**（Graph Sample and Aggregate）：

```python
# 第一層：770 → 256
conv1 = GraphSAGE(input_dim=770, output_dim=256)

# 第二層：256 → 256  
conv2 = GraphSAGE(input_dim=256, output_dim=256)

# 第三層：256 → 128
conv3 = GraphSAGE(input_dim=256, output_dim=128)
```

**GraphSAGE 的工作原理**：
1. **採樣鄰居**：選擇周圍的節點
2. **聚合信息**：整合鄰居的特徵
3. **更新表示**：結合自身和鄰居信息

```
原始節點 A：[特徵向量]
     ↓
聚合鄰居 B, C, D 的信息
     ↓
更新後的 A：[包含鄰居信息的新特徵]
```

### 3. 注意力機制（Attention）

使用 **GAT**（Graph Attention Network）：

```python
attention = GAT(
    in_channels=128,
    out_channels=128,
    heads=4,  # 4個注意力頭
    concat=False
)
```

**注意力的作用**：
- 不是所有鄰居都同等重要
- 學習給不同鄰居分配不同權重
- 例如：被成功說服的評論應該更受關注

### 4. 任務特定輸出層

```python
# Delta 預測頭（二分類）
delta_head = Linear(128 → 64) → ReLU → Dropout → Linear(64 → 1) → Sigmoid

# 品質評分頭（回歸）
quality_head = Linear(128 → 64) → ReLU → Dropout → Linear(64 → 1)

# 策略分類頭（4分類）
strategy_head = Linear(128 → 64) → ReLU → Dropout → Linear(64 → 4) → Softmax
```

### 完整架構圖

```
輸入 (770維)
    ↓
GraphSAGE 層1 (→256維)
    ↓
GraphSAGE 層2 (→256維)
    ↓
GraphSAGE 層3 (→128維)
    ↓
GAT 注意力層 (128維)
    ↓
    ├─→ Delta 預測
    ├─→ 品質評分
    └─→ 策略分類
```

## 訓練過程

### 1. 損失函數

多任務學習使用組合損失：

```python
total_loss = w1 * delta_loss + w2 * quality_loss + w3 * strategy_loss

其中：
- delta_loss: 二元交叉熵（BCE）
- quality_loss: 均方誤差（MSE）
- strategy_loss: 交叉熵（CE）
- w1=0.5, w2=0.3, w3=0.2（權重）
```

### 2. 訓練步驟

```python
for epoch in range(50):
    for batch in dataloader:
        # 1. 前向傳播
        outputs = model(batch.x, batch.edge_index)
        
        # 2. 計算損失
        loss = compute_multi_task_loss(outputs, batch.y)
        
        # 3. 反向傳播
        loss.backward()
        
        # 4. 更新參數
        optimizer.step()
```

### 3. 訓練技巧

- **批次處理**：將大圖切分成子圖
- **Dropout**：防止過擬合（0.2）
- **早停**：驗證集性能不提升時停止
- **學習率調度**：逐步降低學習率

## 推理過程

### 什麼是推理？

訓練完成後，使用模型預測新數據的過程。

### 推理步驟

1. **準備輸入**
```python
# 獲取文本的 BERT 嵌入
text = "我認為這個政策會帶來負面影響..."
text_embedding = bert_model.encode(text)  # 768維

# 添加論證特徵
evidence_score = count_evidence_markers(text) / len(text.split())
length_feature = len(text) / 1000  # 標準化
features = np.concatenate([text_embedding, [evidence_score, length_feature]])
```

2. **構建圖結構**
```python
# 如果是單一預測，創建單節點圖
x = torch.tensor(features).unsqueeze(0)  # [1, 770]
edge_index = torch.tensor([[0], [0]])    # 自環
```

3. **模型預測**
```python
with torch.no_grad():
    outputs = model(x, edge_index)
    
    delta_prob = torch.sigmoid(outputs['delta']).item()
    quality_score = outputs['quality'].item()
    strategy_probs = torch.softmax(outputs['strategy'], dim=1)
```

4. **解釋結果**
```python
result = {
    'delta_probability': 0.73,      # 73% 可能成功說服
    'quality_score': 82.5,          # 品質分數 82.5/100
    'best_strategy': 'analytical',  # 建議使用分析型策略
    'strategy_scores': {
        'aggressive': 0.15,
        'defensive': 0.20,
        'analytical': 0.45,  # 最高
        'empathetic': 0.20
    }
}
```

## 實際應用

### 1. 辯論中的策略建議

```python
def get_strategy_recommendation(topic, current_stance, opponent_stance):
    # 準備特徵
    features = prepare_features(topic, current_stance)
    
    # GNN 預測
    prediction = gnn_model.predict(features)
    
    # 根據預測調整策略
    if prediction['delta_probability'] > 0.7:
        return prediction['best_strategy']
    else:
        # 如果預測效果不好，嘗試其他策略
        return select_alternative_strategy(prediction['strategy_scores'])
```

### 2. 評估發言效果

```python
def evaluate_response(response_text, context):
    # 使用 GNN 預測這個回應的說服力
    features = extract_features(response_text, context)
    prediction = gnn_model.predict(features)
    
    return {
        'persuasiveness': prediction['delta_probability'],
        'quality': prediction['quality_score'],
        'suggested_improvement': get_improvement_tips(prediction)
    }
```

### 3. 社會影響力計算

```python
def calculate_social_influence(agent_id, interaction_history):
    # 構建局部社交圖
    local_graph = build_interaction_graph(agent_id, interaction_history)
    
    # 使用 GNN 編碼
    agent_embedding = gnn_model.encode(local_graph, agent_id)
    
    # 計算影響力分數
    influence_score = compute_influence(agent_embedding)
    return influence_score
```

## 程式碼範例

### 完整的訓練腳本

```python
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import Data, DataLoader

class PersuasionGNN(nn.Module):
    def __init__(self, input_dim=770, hidden_dim=256, num_strategies=4):
        super().__init__()
        
        # 圖卷積層
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim // 2)
        
        # 注意力層
        self.attention = GATConv(
            hidden_dim // 2, 
            hidden_dim // 2, 
            heads=4, 
            concat=False
        )
        
        # 任務頭
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        self.strategy_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_strategies)
        )
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, edge_index, batch=None):
        # 圖卷積
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x, edge_index))
        
        # 注意力
        x = self.attention(x, edge_index)
        x = self.dropout(x)
        
        # 如果有批次信息，進行池化
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # 多任務輸出
        return {
            'delta': self.delta_head(x),
            'quality': self.quality_head(x),
            'strategy': self.strategy_head(x),
            'embeddings': x
        }

# 訓練函數
def train_gnn(model, train_loader, val_loader, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # 前向傳播
            outputs = model(batch.x, batch.edge_index, batch.batch)
            
            # 計算多任務損失
            delta_loss = F.binary_cross_entropy_with_logits(
                outputs['delta'], 
                batch.delta_labels
            )
            quality_loss = F.mse_loss(
                outputs['quality'], 
                batch.quality_labels
            )
            strategy_loss = F.cross_entropy(
                outputs['strategy'], 
                batch.strategy_labels
            )
            
            # 組合損失
            loss = 0.5 * delta_loss + 0.3 * quality_loss + 0.2 * strategy_loss
            
            # 反向傳播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 驗證
        val_metrics = evaluate(model, val_loader)
        print(f"Epoch {epoch}: Loss={total_loss:.4f}, "
              f"Delta_Acc={val_metrics['delta_acc']:.4f}")

# 使用範例
if __name__ == "__main__":
    # 載入數據
    train_data = load_cmv_data('train')
    val_data = load_cmv_data('val')
    
    # 創建數據載入器
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    
    # 初始化模型
    model = PersuasionGNN()
    
    # 訓練
    train_gnn(model, train_loader, val_loader)
    
    # 保存模型
    torch.save(model.state_dict(), 'gnn_persuasion.pt')
```

### 推理使用範例

```python
# 載入訓練好的模型
model = PersuasionGNN()
model.load_state_dict(torch.load('gnn_persuasion.pt'))
model.eval()

# 預測函數
def predict_persuasion(text, context=None):
    # 1. 文本編碼
    text_features = bert_model.encode(text)
    
    # 2. 提取論證特徵
    evidence_markers = ['because', 'therefore', 'research shows']
    evidence_score = sum(1 for marker in evidence_markers if marker in text.lower())
    evidence_score = evidence_score / len(text.split())
    
    length_feature = min(len(text.split()) / 100, 1.0)
    
    # 3. 組合特徵
    features = torch.tensor(
        np.concatenate([text_features, [evidence_score, length_feature]]),
        dtype=torch.float32
    ).unsqueeze(0)
    
    # 4. 創建圖（單節點）
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    
    # 5. 預測
    with torch.no_grad():
        outputs = model(features, edge_index)
        
        delta_prob = torch.sigmoid(outputs['delta']).item()
        quality = outputs['quality'].item()
        strategies = torch.softmax(outputs['strategy'], dim=1).squeeze().tolist()
    
    # 6. 返回結果
    strategy_names = ['aggressive', 'defensive', 'analytical', 'empathetic']
    best_strategy_idx = strategies.index(max(strategies))
    
    return {
        'delta_probability': delta_prob,
        'quality_score': quality,
        'best_strategy': strategy_names[best_strategy_idx],
        'strategy_scores': dict(zip(strategy_names, strategies))
    }

# 使用範例
text = "我理解你的觀點，但研究顯示這個政策在其他國家已經成功實施..."
result = predict_persuasion(text)

print(f"說服成功率: {result['delta_probability']:.2%}")
print(f"品質分數: {result['quality_score']:.1f}")
print(f"建議策略: {result['best_strategy']}")
```

## 常見問題

### Q1: 為什麼要用圖結構？

**A**: 辯論是互動的過程，不能孤立看待每個發言。圖結構能捕捉：
- 誰對誰說話（方向性）
- 說話的順序（時序性）
- 參與者之間的關係（社交性）

### Q2: GraphSAGE 和普通卷積有什麼區別？

**A**: 
- **普通卷積**：在固定網格上操作（如圖像）
- **GraphSAGE**：在不規則圖結構上操作，能處理不同大小的鄰居

### Q3: 為什麼要多任務學習？

**A**: 三個任務相互關聯：
- 預測是否能說服 → 幫助理解什麼樣的論述有效
- 評估品質分數 → 提供更細緻的效果評估
- 分類策略類型 → 指導策略選擇

### Q4: 訓練需要多少數據？

**A**: 
- 最少：~5,000 個標註的評論
- 建議：~20,000 個評論以上
- CMV 數據集：~35,000 個評論

### Q5: 如何提升模型效果？

**A**: 
1. **更多數據**：收集更多高質量的標註數據
2. **特徵工程**：添加更多論證特徵（如情感、修辭手法）
3. **模型調優**：調整層數、維度、注意力頭數
4. **預訓練**：使用更大的辯論語料預訓練

### Q6: GNN 的限制是什麼？

**A**: 
- 需要構建圖結構（計算開銷）
- 對圖的大小敏感（太大會記憶體不足）
- 需要足夠的連接才能發揮優勢

## 總結

GNN 在辯論系統中的作用：
1. **理解關係**：不只看內容，還看互動
2. **預測效果**：評估論述的說服力
3. **策略建議**：推薦最佳辯論策略
4. **社會建模**：理解參與者的影響力

通過圖神經網路，我們讓 AI 不僅理解「說什麼」，更理解「對誰說」和「怎麼說」，這正是成功辯論的關鍵。 