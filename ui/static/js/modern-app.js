// Modern Social Debate AI - 簡化且穩定的前端邏輯

// 全局狀態
let state = {
    initialized: false,
    topic: '',
    currentRound: 0,
    debating: false,
    loading: false
};

// DOM 元素緩存
let elements = {};

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log('初始化 Social Debate AI...');
    
    // 緩存 DOM 元素
    elements = {
        topicInput: document.getElementById('topicInput'),
        topicDisplay: document.getElementById('topicDisplay'),
        currentRound: document.getElementById('currentRound'),
        debateStatus: document.getElementById('debateStatus'),
        debateContent: document.getElementById('debateContent'),
        loadingOverlay: document.getElementById('loadingOverlay'),
        loadingText: document.getElementById('loadingText'),
        startBtn: document.getElementById('startBtn'),
        nextBtn: document.getElementById('nextBtn')
    };
    
    // 綁定 Enter 鍵
    elements.topicInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            setTopic();
        }
    });
    
    // 初始化系統
    initSystem();
});

// 顯示/隱藏載入動畫（簡化版）
function showLoading(text = '處理中...') {
    console.log('顯示載入:', text);
    elements.loadingText.textContent = text;
    elements.loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    console.log('隱藏載入');
    elements.loadingOverlay.style.display = 'none';
}

// 顯示訊息
function showMessage(message, type = 'info') {
    console.log(`[${type}] ${message}`);
    
    // 創建訊息元素
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show`;
    alertDiv.style.cssText = 'position: fixed; top: 80px; right: 20px; z-index: 1050; min-width: 300px;';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // 自動移除
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

// 初始化系統
async function initSystem() {
    try {
        showLoading('正在初始化系統...');
        
        const response = await fetch('/api/init', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'}
        });
        
        const data = await response.json();
        
        if (data.success) {
            state.initialized = true;
            showMessage('系統初始化成功！', 'success');
            updateUI();
        } else {
            throw new Error(data.message || '初始化失敗');
        }
    } catch (error) {
        console.error('初始化錯誤:', error);
        showMessage('系統初始化失敗: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// 設置主題
async function setTopic() {
    const topic = elements.topicInput.value.trim();
    
    if (!topic) {
        showMessage('請輸入辯論主題', 'warning');
        return;
    }
    
    if (!state.initialized) {
        showMessage('系統尚未初始化', 'warning');
        return;
    }
    
    try {
        showLoading('正在設置主題...');
        
        const response = await fetch('/api/set_topic', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ topic: topic })
        });
        
        const data = await response.json();
        
        if (data.success) {
            state.topic = topic;
            state.currentRound = 0;
            elements.topicDisplay.textContent = topic;
            showMessage('主題設置成功！', 'success');
            
            // 清空辯論內容，顯示開始按鈕
            elements.debateContent.innerHTML = `
                <div class="text-center py-5">
                    <h4>主題已設置：${topic}</h4>
                    <p class="text-muted">點擊"開始辯論"按鈕開始第一輪辯論</p>
                </div>
            `;
            
            updateUI();
        } else {
            throw new Error(data.message || '設置失敗');
        }
    } catch (error) {
        console.error('設置主題錯誤:', error);
        showMessage('設置主題失敗: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// 開始辯論
async function startDebate() {
    if (!state.topic) {
        showMessage('請先設置辯論主題', 'warning');
        return;
    }
    
    state.currentRound = 0;
    elements.debateContent.innerHTML = '';
    await runDebateRound();
}

// 下一回合
async function nextRound() {
    await runDebateRound();
}

// 執行辯論回合
async function runDebateRound() {
    if (!state.initialized || !state.topic) {
        showMessage('請先初始化系統並設置主題', 'warning');
        return;
    }
    
    if (state.loading) {
        showMessage('請等待當前操作完成', 'info');
        return;
    }
    
    try {
        state.loading = true;
        state.debating = true;
        updateUI();
        
        showLoading(state.currentRound === 0 ? 
            '首次載入模型，請稍候（10-30秒）...' : 
            '智能體正在思考中...'
        );
        
        const response = await fetch('/api/debate_round', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            // 添加超時控制
            signal: AbortSignal.timeout(60000)  // 60秒超時
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            state.currentRound = data.round;
            elements.currentRound.textContent = data.round;
            
            // 顯示辯論內容
            displayDebateRound(data);
            
            // 更新 Agent 狀態
            if (data.agent_states) {
                updateAgentStates(data.agent_states);
            }
            
            // 檢查是否結束
            if (data.debate_ended) {
                state.debating = false;
                if (data.summary) {
                    showDebateResult(data.summary);
                }
                showMessage('辯論已結束！', 'info');
            } else {
                showMessage(`第 ${data.round} 輪辯論完成！`, 'success');
            }
        } else {
            throw new Error(data.message || '執行失敗');
        }
    } catch (error) {
        console.error('辯論執行錯誤:', error);
        
        // 根據錯誤類型顯示不同訊息
        if (error.name === 'AbortError') {
            showMessage('請求超時，請重試', 'error');
        } else if (error.message.includes('fetch')) {
            showMessage('網絡連接錯誤，請檢查連接', 'error');
        } else {
            showMessage('辯論執行失敗: ' + error.message, 'error');
        }
        
        // 如果是第一輪就失敗，重置辯論狀態
        if (state.currentRound === 0) {
            state.debating = false;
        }
    } finally {
        state.loading = false;
        updateUI();
        hideLoading();
    }
}

// 顯示辯論回合
function displayDebateRound(data) {
    // 檢查數據完整性
    if (!data || !data.round) {
        console.error('辯論數據不完整:', data);
        showMessage('辯論數據錯誤', 'error');
        return;
    }
    
    const roundDiv = document.createElement('div');
    roundDiv.className = 'debate-round';
    roundDiv.innerHTML = `
        <div class="round-header">
            <div class="round-number">${data.round}</div>
            <h4>第 ${data.round} 回合</h4>
        </div>
    `;
    
    // 檢查 responses 是否存在且為數組
    if (data.responses && Array.isArray(data.responses) && data.responses.length > 0) {
        // 添加每個 AI 的回應
        data.responses.forEach(response => {
            if (!response || !response.agent_id) {
                console.warn('跳過無效的回應:', response);
                return;
            }
            
            const agentType = response.agent_id === 'Agent_A' ? 'support' : 
                             response.agent_id === 'Agent_B' ? 'oppose' : 'neutral';
            const agentName = response.agent_id === 'Agent_A' ? '支持者 A' :
                             response.agent_id === 'Agent_B' ? '反對者 B' : '中立者 C';
            
            const responseDiv = document.createElement('div');
            responseDiv.className = `ai-response ${agentType}`;
            
            // 安全地獲取效果數據
            const persuasion = response.effects?.persuasion_score || 0;
            const attack = response.effects?.attack_score || 0;
            
            responseDiv.innerHTML = `
                <div class="response-header">
                    <div class="agent-avatar ${agentType}">
                        <i class="fas ${agentType === 'support' ? 'fa-user-tie' : 
                                       agentType === 'oppose' ? 'fa-user-shield' : 'fa-user-graduate'}"></i>
                    </div>
                    <div>
                        <h5>${agentName}</h5>
                        <small class="text-muted">
                            說服力: ${(persuasion * 100).toFixed(0)}% | 
                            攻擊力: ${(attack * 100).toFixed(0)}%
                        </small>
                    </div>
                </div>
                <div class="response-content">
                    ${response.content || '(無內容)'}
                </div>
            `;
            
            roundDiv.appendChild(responseDiv);
        });
    } else {
        // 如果沒有回應數據，顯示錯誤訊息
        const errorDiv = document.createElement('div');
        errorDiv.className = 'alert alert-warning';
        errorDiv.textContent = '本回合沒有收到有效的回應數據';
        roundDiv.appendChild(errorDiv);
    }
    
    elements.debateContent.appendChild(roundDiv);
    
    // 滾動到最新內容
    elements.debateContent.scrollTop = elements.debateContent.scrollHeight;
}

// 更新 Agent 狀態
function updateAgentStates(states) {
    // 檢查 states 是否存在
    if (!states || typeof states !== 'object') {
        console.warn('Agent 狀態數據無效:', states);
        return;
    }
    
    Object.entries(states).forEach(([agentId, state]) => {
        try {
            const suffix = agentId.split('_')[1];
            
            // 更新立場進度條
            const stanceBar = document.getElementById(`stance${suffix}`);
            if (stanceBar && state.stance !== undefined) {
                const stancePercent = ((state.stance + 1) / 2 * 100).toFixed(0);
                stanceBar.style.width = stancePercent + '%';
                const stanceSpan = stanceBar.querySelector('span');
                if (stanceSpan) {
                    stanceSpan.textContent = 
                        state.stance > 0 ? `+${state.stance.toFixed(2)}` : state.stance.toFixed(2);
                }
            }
            
            // 更新信念進度條
            const convictionBar = document.getElementById(`conviction${suffix}`);
            if (convictionBar && state.conviction !== undefined) {
                const convictionPercent = (state.conviction * 100).toFixed(0);
                convictionBar.style.width = convictionPercent + '%';
                const convictionSpan = convictionBar.querySelector('span');
                if (convictionSpan) {
                    convictionSpan.textContent = state.conviction.toFixed(2);
                }
            }
            
            // 檢查投降狀態
            if (state.has_surrendered) {
                const agentCard = document.getElementById(`agent${suffix}`);
                if (agentCard) {
                    agentCard.style.opacity = '0.6';
                    showMessage(`${agentId.replace('_', ' ')} 已投降！`, 'warning');
                }
            }
        } catch (error) {
            console.error(`更新 ${agentId} 狀態時出錯:`, error);
        }
    });
}

// 顯示辯論結果
function showDebateResult(summary) {
    // 檢查 summary 是否存在
    if (!summary) {
        console.error('辯論結果數據不存在');
        return;
    }
    
    const resultDiv = document.createElement('div');
    resultDiv.className = 'debate-result text-center py-5';
    
    // 安全地構建結果 HTML
    let scoresHtml = '';
    if (summary.scores && typeof summary.scores === 'object') {
        scoresHtml = Object.entries(summary.scores)
            .sort(([,a], [,b]) => b - a)
            .map(([agent, score]) => `
                <div class="mb-2">
                    ${agent}: ${(score || 0).toFixed(1)} 分
                    ${agent === summary.winner ? '<span class="badge bg-warning ms-2">獲勝者</span>' : ''}
                </div>
            `).join('');
    }
    
    resultDiv.innerHTML = `
        <h3>辯論結束</h3>
        <p class="lead">${summary.verdict || '辯論已完成'}</p>
        <div class="mt-4">
            <h5>最終得分</h5>
            ${scoresHtml || '<p class="text-muted">無評分數據</p>'}
        </div>
    `;
    
    elements.debateContent.appendChild(resultDiv);
}

// 重置辯論
async function resetDebate() {
    if (state.loading) {
        showMessage('請等待當前操作完成', 'info');
        return;
    }
    
    if (!confirm('確定要重置辯論嗎？')) {
        return;
    }
    
    try {
        showLoading('正在重置...');
        
        const response = await fetch('/api/reset', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'}
        });
        
        const data = await response.json();
        
        if (data.success) {
            // 重置狀態 - 保持 initialized 為 true
            state.topic = '';
            state.currentRound = 0;
            state.debating = false;
            state.loading = false;  // 確保 loading 狀態被重置
            
            // 重置 UI
            elements.topicInput.value = '';
            elements.topicDisplay.textContent = '請設置辯論主題以開始';
            elements.currentRound.textContent = '0';
            elements.debateContent.innerHTML = `
                <div class="welcome-screen">
                    <div class="welcome-icon">
                        <i class="fas fa-robot"></i>
                    </div>
                    <h3>歡迎使用 Social Debate AI</h3>
                    <p>這是一個多智能體辯論系統，三個 AI 代理將從不同角度討論你提出的主題</p>
                </div>
            `;
            
            // 重置進度條
            resetAgentStates();
            
            showMessage('辯論已重置', 'success');
            updateUI();
        } else {
            throw new Error(data.message || '重置失敗');
        }
    } catch (error) {
        console.error('重置錯誤:', error);
        showMessage('重置失敗: ' + error.message, 'error');
        // 即使失敗也要重置 loading 狀態
        state.loading = false;
        updateUI();
    } finally {
        hideLoading();
    }
}

// 重置 Agent 狀態
function resetAgentStates() {
    // Agent A
    document.getElementById('stanceA').style.width = '80%';
    document.getElementById('stanceA').querySelector('span').textContent = '+0.8';
    document.getElementById('convictionA').style.width = '70%';
    document.getElementById('convictionA').querySelector('span').textContent = '0.7';
    
    // Agent B
    document.getElementById('stanceB').style.width = '30%';
    document.getElementById('stanceB').querySelector('span').textContent = '-0.6';
    document.getElementById('convictionB').style.width = '60%';
    document.getElementById('convictionB').querySelector('span').textContent = '0.6';
    
    // Agent C
    document.getElementById('stanceC').style.width = '50%';
    document.getElementById('stanceC').querySelector('span').textContent = '0.0';
    document.getElementById('convictionC').style.width = '50%';
    document.getElementById('convictionC').querySelector('span').textContent = '0.5';
    
    // 移除投降狀態
    document.querySelectorAll('.agent-card').forEach(card => {
        card.style.opacity = '1';
    });
}

// 更新 UI 狀態
function updateUI() {
    // 更新按鈕狀態
    elements.startBtn.disabled = !state.initialized || !state.topic || state.loading || state.debating;
    elements.nextBtn.disabled = !state.initialized || !state.topic || state.loading || !state.debating || state.currentRound === 0;
    
    // 更新狀態顯示
    if (state.debating) {
        elements.debateStatus.textContent = '進行中';
        elements.debateStatus.style.color = 'var(--success-color)';
    } else if (state.topic) {
        elements.debateStatus.textContent = '準備就緒';
        elements.debateStatus.style.color = 'var(--info-color)';
    } else {
        elements.debateStatus.textContent = '等待設置';
        elements.debateStatus.style.color = 'var(--warning-color)';
    }
}

// 導出辯論記錄
async function exportDebate() {
    if (state.currentRound === 0) {
        showMessage('沒有可導出的辯論記錄', 'warning');
        return;
    }
    
    try {
        showLoading('正在導出...');
        
        const response = await fetch('/api/export');
        const data = await response.json();
        
        if (data.success) {
            // 創建下載
            const blob = new Blob([JSON.stringify(data.data, null, 2)], 
                                 { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `debate_${new Date().toISOString().slice(0, 10)}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            showMessage('辯論記錄已導出', 'success');
        } else {
            throw new Error(data.message || '導出失敗');
        }
    } catch (error) {
        console.error('導出錯誤:', error);
        showMessage('導出失敗: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// 顯示統計
function showStats() {
    showMessage('統計功能開發中...', 'info');
}

// 顯示關於
function showAbout() {
    showMessage('Social Debate AI - 多智能體辯論系統 v1.0', 'info');
}

// 切換主題
function toggleTheme() {
    document.body.classList.toggle('dark-theme');
    const icon = document.getElementById('themeIcon');
    icon.className = document.body.classList.contains('dark-theme') ? 
                     'fas fa-sun' : 'fas fa-moon';
} 