# Social Debate AI API Reference

*English | [](#chinese-version)*

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

*[English](#social-debate-ai-api-reference) | *

 Flask Web API 

## 

- ** URL**: `http://localhost:5000`
- ****: `application/json`
- ****: 

## API 

### 

|  |  |  |
|------|------|------|
| POST | `/api/init` |  |
| POST | `/api/reset` |  |

### 

|  |  |  |
|------|------|------|
| POST | `/api/set_topic` |  |
| POST | `/api/debate_round` |  |
| GET | `/api/debate_history` |  |
| GET | `/api/debate_summary` |  |

### 

|  |  |  |
|------|------|------|
| GET | `/api/export` |  |

## 

### 1. 

****: `POST /api/init`

****: 

****:
```bash
curl -X POST http://localhost:5000/api/init \
  -H "Content-Type: application/json"
```

****:
```json
{
  "success": true,
  "message": "",
  "debate_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

****:
```json
{
  "success": false,
  "message": ": []"
}
```

### 2. 

****: `POST /api/set_topic`

****:  Agent 

****:
```json
{
  "topic": ""
}
```

****:
```bash
curl -X POST http://localhost:5000/api/set_topic \
  -H "Content-Type: application/json" \
  -d '{"topic": ""}'
```

****:
```json
{
  "success": true,
  "topic": "",
  "message": ": "
}
```

### 3. 

****: `POST /api/debate_round`

****:  Agent 

****:
```bash
curl -X POST http://localhost:5000/api/debate_round \
  -H "Content-Type: application/json"
```

****:
```json
{
  "success": true,
  "round": 1,
  "responses": [
    {
      "agent_id": "Agent_A",
      "content": "...",
      "effects": {
        "persuasion_score": 0.3,
        "attack_score": 0.1,
        "evidence_score": 0.4,
        "length_score": 0.8
      },
      "timestamp": 1642123456.789
    }
    // ...  Agent 
  ],
  "agent_states": {
    "Agent_A": {
      "stance": 0.8,
      "conviction": 0.7,
      "has_surrendered": false,
      "persuasion_avg": 0.15
    }
    // ...  Agent 
  },
  "debate_ended": false,
  "message": " 1 "
}
```

****:
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
    "verdict": "Agent_A ",
    "surrendered_agents": ["Agent_B"],
    "final_states": {
      "Agent_A": {
        "stance": 0.75,
        "conviction": 0.65,
        "final_position": ""
      }
      // ...  Agent 
    },
    "total_rounds": 5
  },
  "message": ""
}
```

### 4. 

****: `GET /api/debate_history`

****: 

****:
```bash
curl http://localhost:5000/api/debate_history
```

****:
```json
{
  "success": true,
  "topic": "",
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
    // ... 
  ]
}
```

### 5. 

****: `GET /api/debate_summary`

****: 

****:
```bash
curl http://localhost:5000/api/debate_summary
```

****:
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

### 6. 

****: `POST /api/reset`

****: 

****:
```bash
curl -X POST http://localhost:5000/api/reset \
  -H "Content-Type: application/json"
```

****:
```json
{
  "success": true,
  "message": "",
  "debate_id": "new-debate-id"
}
```

### 7. 

****: `GET /api/export`

****:  JSON 

****:
```bash
curl http://localhost:5000/api/export -o debate_export.json
```

****:
```json
{
  "success": true,
  "data": {
    "debate_id": "550e8400-e29b-41d4-a716-446655440000",
    "topic": "",
    "total_rounds": 5,
    "history": [ /* ... */ ],
    "exported_at": "2024-01-20T10:30:00.000Z"
  }
}
```

## 

 API 

```json
{
  "success": false,
  "message": ""
}
```

 HTTP 
- `200 OK` - 
- `400 Bad Request` - 
- `500 Internal Server Error` - 

## 

### Python 

```python
import requests
import json

#  URL
BASE_URL = "http://localhost:5000"

# 1. 
response = requests.post(f"{BASE_URL}/api/init")
print(response.json())

# 2. 
topic_data = {"topic": ""}
response = requests.post(f"{BASE_URL}/api/set_topic", json=topic_data)
print(response.json())

# 3. 
for i in range(5):
    response = requests.post(f"{BASE_URL}/api/debate_round")
    result = response.json()
    print(f" {result['round']} ")
    
    if result.get('debate_ended'):
        print("")
        print(f": {result['summary']['winner']}")
        break

# 4. 
response = requests.get(f"{BASE_URL}/api/export")
with open("debate_result.json", "w", encoding="utf-8") as f:
    json.dump(response.json()['data'], f, ensure_ascii=False, indent=2)
```

### JavaScript 

```javascript
//  Fetch API
const BASE_URL = 'http://localhost:5000';

// 
async function initSystem() {
  const response = await fetch(`${BASE_URL}/api/init`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'}
  });
  return await response.json();
}

// 
async function setTopic(topic) {
  const response = await fetch(`${BASE_URL}/api/set_topic`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({topic})
  });
  return await response.json();
}

// 
async function runDebateRound() {
  const response = await fetch(`${BASE_URL}/api/debate_round`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'}
  });
  return await response.json();
}

// 
async function runDebate() {
  await initSystem();
  await setTopic('');
  
  let debateEnded = false;
  while (!debateEnded) {
    const result = await runDebateRound();
    console.log(` ${result.round} `);
    debateEnded = result.debate_ended;
  }
}
```

## 

1. ** (CORS)**
   - 
   - 

2. ****
   - 500 
   - 

3. ****
   - 
   - 

## 

-  WebSocket 
- 
- 
- 
-  API

---

**** API  1.0 