# Social Debate AI API Reference

*English | [中文](#chinese-version)*

This document provides detailed specifications for all Flask Web API endpoints and usage methods.

## Basic Information

- **Base URL**: `http://localhost:5000`
- **Content Type**: `application/json`
- **Authentication**: None (will be added in future versions)

## API Endpoint List

### System Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/init` | Initialize system |
| POST | `/api/reset` | Reset debate |

### Debate Control

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/set_topic` | Set debate topic |
| POST | `/api/debate_round` | Execute one debate round |
| GET | `/api/debate_history` | Get debate history |
| GET | `/api/debate_summary` | Get debate summary |

### Data Export

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/export` | Export debate records |

## Detailed Endpoint Specifications

### 1. Initialize System

**Endpoint**: `POST /api/init`

**Description**: Initialize the debate system and load all necessary modules.

**Request Example**:
```bash
curl -X POST http://localhost:5000/api/init \
  -H "Content-Type: application/json"
```

**Success Response**:
```json
{
  "success": true,
  "message": "System initialized successfully",
  "debate_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Error Response**:
```json
{
  "success": false,
  "message": "System initialization failed: [error details]"
}
```

### 2. Set Debate Topic

**Endpoint**: `POST /api/set_topic`

**Description**: Set the debate topic and reset agent states.

**Request Parameters**:
```json
{
  "topic": "Should artificial intelligence be regulated by government?"
}
```

**Request Example**:
```bash
curl -X POST http://localhost:5000/api/set_topic \
  -H "Content-Type: application/json" \
  -d '{"topic": "Should artificial intelligence be regulated by government?"}'
```

**Success Response**:
```json
{
  "success": true,
  "topic": "Should artificial intelligence be regulated by government?",
  "message": "Debate topic set: Should artificial intelligence be regulated by government?"
}
```

### 3. Execute Debate Round

**Endpoint**: `POST /api/debate_round`

**Description**: Execute one debate round with all agents speaking in sequence.

**Request Example**:
```bash
curl -X POST http://localhost:5000/api/debate_round \
  -H "Content-Type: application/json"
```

**Success Response**:
```json
{
  "success": true,
  "round": 1,
  "responses": [
    {
      "agent_id": "Agent_A",
      "content": "I believe artificial intelligence needs government regulation...",
      "effects": {
        "persuasion_score": 0.3,
        "attack_score": 0.1,
        "evidence_score": 0.4,
        "length_score": 0.8
      },
      "timestamp": 1642123456.789
    }
    // ... other agents' responses
  ],
  "agent_states": {
    "Agent_A": {
      "stance": 0.8,
      "conviction": 0.7,
      "has_surrendered": false,
      "persuasion_avg": 0.15
    }
    // ... other agents' states
  },
  "debate_ended": false,
  "message": "Round 1 completed"
}
```

**Debate End Response**:
```json
{
  "success": true,
  "round": 5,
  "debate_ended": true,
  "summary": {
    "winner": "Agent_A",
    "scores": {
      "Agent_A": 82.5,
      "Agent_B": 65.3,
      "Agent_C": 71.2
    },
    "verdict": "Agent_A won with stable performance and strong arguments.",
    "surrendered_agents": ["Agent_B"],
    "final_states": {
      "Agent_A": {
        "stance": 0.75,
        "conviction": 0.65,
        "final_position": "Strongly supportive"
      }
      // ... other agents' final states
    },
    "total_rounds": 5
  },
  "message": "Debate ended!"
}
```

### 4. Get Debate History

**Endpoint**: `GET /api/debate_history`

**Description**: Get complete history of the current debate.

**Request Example**:
```bash
curl http://localhost:5000/api/debate_history
```

**Success Response**:
```json
{
  "success": true,
  "topic": "Should artificial intelligence be regulated by government?",
  "current_round": 3,
  "history": [
    {
      "round": 1,
      "responses": [
        {
          "agent_id": "Agent_A",
          "content": "...",
          "effects": { /* ... */ }
        }
        // ...
      ]
    }
    // ... other rounds
  ]
}
```

### 5. Get Debate Summary

**Endpoint**: `GET /api/debate_summary`

**Description**: Get summary and victory determination of current debate.

**Request Example**:
```bash
curl http://localhost:5000/api/debate_summary
```

**Success Response**:
```json
{
  "success": true,
  "summary": {
    "winner": "Agent_A",
    "scores": { /* ... */ },
    "verdict": "...",
    "surrendered_agents": [],
    "final_states": { /* ... */ },
    "total_rounds": 5
  }
}
```

### 6. Reset Debate

**Endpoint**: `POST /api/reset`

**Description**: Reset the entire debate system and clear all states.

**Request Example**:
```bash
curl -X POST http://localhost:5000/api/reset \
  -H "Content-Type: application/json"
```

**Success Response**:
```json
{
  "success": true,
  "message": "Debate reset",
  "debate_id": "new-debate-id"
}
```

### 7. Export Debate Records

**Endpoint**: `GET /api/export`

**Description**: Export complete debate records in JSON format.

**Request Example**:
```bash
curl http://localhost:5000/api/export -o debate_export.json
```

**Success Response**:
```json
{
  "success": true,
  "data": {
    "debate_id": "550e8400-e29b-41d4-a716-446655440000",
    "topic": "Should artificial intelligence be regulated by government?",
    "total_rounds": 5,
    "history": [ /* ... */ ],
    "exported_at": "2024-01-20T10:30:00.000Z"
  }
}
```

## Error Handling

All API endpoints use a unified error format:

```json
{
  "success": false,
  "message": "Error description"
}
```

Common HTTP Status Codes:
- `200 OK` - Request successful
- `400 Bad Request` - Invalid request parameters
- `500 Internal Server Error` - Server internal error

## Usage Examples

### Python Example

```python
import requests
import json

# Base URL
BASE_URL = "http://localhost:5000"

# 1. Initialize system
response = requests.post(f"{BASE_URL}/api/init")
print(response.json())

# 2. Set topic
topic_data = {"topic": "Should artificial intelligence be regulated by government?"}
response = requests.post(f"{BASE_URL}/api/set_topic", json=topic_data)
print(response.json())

# 3. Execute debate
for i in range(5):
    response = requests.post(f"{BASE_URL}/api/debate_round")
    result = response.json()
    print(f"Round {result['round']} completed")
    
    if result.get('debate_ended'):
        print("Debate ended!")
        print(f"Winner: {result['summary']['winner']}")
        break

# 4. Export results
response = requests.get(f"{BASE_URL}/api/export")
with open("debate_result.json", "w", encoding="utf-8") as f:
    json.dump(response.json()['data'], f, ensure_ascii=False, indent=2)
```

### JavaScript Example

```javascript
// Using Fetch API
const BASE_URL = 'http://localhost:5000';

// Initialize system
async function initSystem() {
  const response = await fetch(`${BASE_URL}/api/init`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'}
  });
  return await response.json();
}

// Set topic
async function setTopic(topic) {
  const response = await fetch(`${BASE_URL}/api/set_topic`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({topic})
  });
  return await response.json();
}

// Execute debate round
async function runDebateRound() {
  const response = await fetch(`${BASE_URL}/api/debate_round`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'}
  });
  return await response.json();
}

// Usage example
async function runDebate() {
  await initSystem();
  await setTopic('Should artificial intelligence be regulated by government?');
  
  let debateEnded = false;
  while (!debateEnded) {
    const result = await runDebateRound();
    console.log(`Round ${result.round} completed`);
    debateEnded = result.debate_ended;
  }
}
```

## Security Considerations

1. **Cross-Origin Requests (CORS)**
   - Default allows all origins
   - Production should configure specific origins

2. **Input Validation**
   - Topic length limit: 500 characters
   - Special characters are filtered

3. **Rate Limiting**
   - Currently no limits
   - Recommended to add in production

## Future Plans

- Add WebSocket support for real-time updates
- Implement user authentication and authorization
- Support multiple concurrent debates
- Add debate replay functionality
- Provide more statistical analysis APIs

---

**Note**: This API documentation corresponds to version 1.0, future versions may change.

---

## Chinese Version

# Social Debate AI API Reference

*[English](#social-debate-ai-api-reference) | 中文*

本文檔詳細說明 Flask Web API 的所有端點和使用方法。

## 基礎信息

- **基礎 URL**: `http://localhost:5000`
- **內容類型**: `application/json`
- **認證**: 暫無（未來版本將添加）

## API 端點列表

### 系統管理

| 方法 | 端點 | 說明 |
|------|------|------|
| POST | `/api/init` | 初始化系統 |
| POST | `/api/reset` | 重置辯論 |

### 辯論控制

| 方法 | 端點 | 說明 |
|------|------|------|
| POST | `/api/set_topic` | 設置辯論主題 |
| POST | `/api/debate_round` | 執行一輪辯論 |
| GET | `/api/debate_history` | 獲取辯論歷史 |
| GET | `/api/debate_summary` | 獲取辯論總結 |

### 數據導出

| 方法 | 端點 | 說明 |
|------|------|------|
| GET | `/api/export` | 導出辯論記錄 |

## 詳細端點說明

### 1. 初始化系統

**端點**: `POST /api/init`

**說明**: 初始化辯論系統，載入所有必要的模組。

**請求範例**:
```bash
curl -X POST http://localhost:5000/api/init \
  -H "Content-Type: application/json"
```

**成功響應**:
```json
{
  "success": true,
  "message": "系統初始化成功",
  "debate_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**錯誤響應**:
```json
{
  "success": false,
  "message": "系統初始化失敗: [錯誤詳情]"
}
```

### 2. 設置辯論主題

**端點**: `POST /api/set_topic`

**說明**: 設置辯論主題並重置 Agent 狀態。

**請求參數**:
```json
{
  "topic": "人工智慧是否應該由政府監管？"
}
```

**請求範例**:
```bash
curl -X POST http://localhost:5000/api/set_topic \
  -H "Content-Type: application/json" \
  -d '{"topic": "人工智慧是否應該由政府監管？"}'
```

**成功響應**:
```json
{
  "success": true,
  "topic": "人工智慧是否應該由政府監管？",
  "message": "辯論主題已設置: 人工智慧是否應該由政府監管？"
}
```

### 3. 執行辯論回合

**端點**: `POST /api/debate_round`

**說明**: 執行一輪辯論，所有 Agent 依序發言。

**請求範例**:
```bash
curl -X POST http://localhost:5000/api/debate_round \
  -H "Content-Type: application/json"
```

**成功響應**:
```json
{
  "success": true,
  "round": 1,
  "responses": [
    {
      "agent_id": "Agent_A",
      "content": "我認為人工智慧需要政府監管...",
      "effects": {
        "persuasion_score": 0.3,
        "attack_score": 0.1,
        "evidence_score": 0.4,
        "length_score": 0.8
      },
      "timestamp": 1642123456.789
    }
    // ... 其他 Agent 的回應
  ],
  "agent_states": {
    "Agent_A": {
      "stance": 0.8,
      "conviction": 0.7,
      "has_surrendered": false,
      "persuasion_avg": 0.15
    }
    // ... 其他 Agent 的狀態
  },
  "debate_ended": false,
  "message": "第 1 輪辯論完成"
}
```

**辯論結束響應**:
```json
{
  "success": true,
  "round": 5,
  "debate_ended": true,
  "summary": {
    "winner": "Agent_A",
    "scores": {
      "Agent_A": 82.5,
      "Agent_B": 65.3,
      "Agent_C": 71.2
    },
    "verdict": "Agent_A 憑藉穩定的表現和有力的論證獲得勝利。",
    "surrendered_agents": ["Agent_B"],
    "final_states": {
      "Agent_A": {
        "stance": 0.75,
        "conviction": 0.65,
        "final_position": "堅定支持"
      }
      // ... 其他 Agent 的最終狀態
    },
    "total_rounds": 5
  },
  "message": "辯論已結束！"
}
```

### 4. 獲取辯論歷史

**端點**: `GET /api/debate_history`

**說明**: 獲取當前辯論的完整歷史記錄。

**請求範例**:
```bash
curl http://localhost:5000/api/debate_history
```

**成功響應**:
```json
{
  "success": true,
  "topic": "人工智慧是否應該由政府監管？",
  "current_round": 3,
  "history": [
    {
      "round": 1,
      "responses": [
        {
          "agent_id": "Agent_A",
          "content": "...",
          "effects": { /* ... */ }
        }
        // ...
      ]
    }
    // ... 其他回合
  ]
}
```

### 5. 獲取辯論總結

**端點**: `GET /api/debate_summary`

**說明**: 獲取當前辯論的總結和勝負判定。

**請求範例**:
```bash
curl http://localhost:5000/api/debate_summary
```

**成功響應**:
```json
{
  "success": true,
  "summary": {
    "winner": "Agent_A",
    "scores": { /* ... */ },
    "verdict": "...",
    "surrendered_agents": [],
    "final_states": { /* ... */ },
    "total_rounds": 5
  }
}
```

### 6. 重置辯論

**端點**: `POST /api/reset`

**說明**: 重置整個辯論系統，清空所有狀態。

**請求範例**:
```bash
curl -X POST http://localhost:5000/api/reset \
  -H "Content-Type: application/json"
```

**成功響應**:
```json
{
  "success": true,
  "message": "辯論已重置",
  "debate_id": "new-debate-id"
}
```

### 7. 導出辯論記錄

**端點**: `GET /api/export`

**說明**: 導出完整的辯論記錄為 JSON 格式。

**請求範例**:
```bash
curl http://localhost:5000/api/export -o debate_export.json
```

**成功響應**:
```json
{
  "success": true,
  "data": {
    "debate_id": "550e8400-e29b-41d4-a716-446655440000",
    "topic": "人工智慧是否應該由政府監管？",
    "total_rounds": 5,
    "history": [ /* ... */ ],
    "exported_at": "2024-01-20T10:30:00.000Z"
  }
}
```

## 錯誤處理

所有 API 端點都使用統一的錯誤格式：

```json
{
  "success": false,
  "message": "錯誤描述"
}
```

常見 HTTP 狀態碼：
- `200 OK` - 請求成功
- `400 Bad Request` - 請求參數錯誤
- `500 Internal Server Error` - 服務器內部錯誤

## 使用範例

### Python 範例

```python
import requests
import json

# 基礎 URL
BASE_URL = "http://localhost:5000"

# 1. 初始化系統
response = requests.post(f"{BASE_URL}/api/init")
print(response.json())

# 2. 設置主題
topic_data = {"topic": "人工智慧是否應該由政府監管？"}
response = requests.post(f"{BASE_URL}/api/set_topic", json=topic_data)
print(response.json())

# 3. 執行辯論
for i in range(5):
    response = requests.post(f"{BASE_URL}/api/debate_round")
    result = response.json()
    print(f"第 {result['round']} 輪完成")
    
    if result.get('debate_ended'):
        print("辯論結束！")
        print(f"獲勝者: {result['summary']['winner']}")
        break

# 4. 導出結果
response = requests.get(f"{BASE_URL}/api/export")
with open("debate_result.json", "w", encoding="utf-8") as f:
    json.dump(response.json()['data'], f, ensure_ascii=False, indent=2)
```

### JavaScript 範例

```javascript
// 使用 Fetch API
const BASE_URL = 'http://localhost:5000';

// 初始化系統
async function initSystem() {
  const response = await fetch(`${BASE_URL}/api/init`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'}
  });
  return await response.json();
}

// 設置主題
async function setTopic(topic) {
  const response = await fetch(`${BASE_URL}/api/set_topic`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({topic})
  });
  return await response.json();
}

// 執行辯論回合
async function runDebateRound() {
  const response = await fetch(`${BASE_URL}/api/debate_round`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'}
  });
  return await response.json();
}

// 使用範例
async function runDebate() {
  await initSystem();
  await setTopic('人工智慧是否應該由政府監管？');
  
  let debateEnded = false;
  while (!debateEnded) {
    const result = await runDebateRound();
    console.log(`第 ${result.round} 輪完成`);
    debateEnded = result.debate_ended;
  }
}
```

## 安全注意事項

1. **跨域請求 (CORS)**
   - 預設允許所有來源
   - 生產環境應配置特定來源

2. **輸入驗證**
   - 主題長度限制：500 字符
   - 特殊字符會被過濾

3. **速率限制**
   - 目前無限制
   - 建議生產環境添加

## 未來計劃

- 添加 WebSocket 支援實時更新
- 實現用戶認證和授權
- 支援多場並行辯論
- 添加辯論回放功能
- 提供更多統計分析 API

---

**注意**：本 API 文檔對應版本 1.0，後續版本可能有所變更。 