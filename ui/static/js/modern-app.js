// Modern Social Debate AI - 

// 
let state = {
    initialized: false,
    topic: '',
    currentRound: 0,
    debating: false,
    loading: false
};

// DOM 
let elements = {};

// 
document.addEventListener('DOMContentLoaded', function() {
    console.log(' Social Debate AI...');
    
    //  DOM 
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
    
    //  Enter 
    elements.topicInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            setTopic();
        }
    });
    
    // 
    initSystem();
});

// /
function showLoading(text = '...') {
    console.log(':', text);
    elements.loadingText.textContent = text;
    elements.loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    console.log('');
    elements.loadingOverlay.style.display = 'none';
}

// 
function showMessage(message, type = 'info') {
    console.log(`[${type}] ${message}`);
    
    // 
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show`;
    alertDiv.style.cssText = 'position: fixed; top: 80px; right: 20px; z-index: 1050; min-width: 300px;';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // 
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

// 
async function initSystem() {
    try {
        showLoading('...');
        
        const response = await fetch('/api/init', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'}
        });
        
        const data = await response.json();
        
        if (data.success) {
            state.initialized = true;
            showMessage('', 'success');
            updateUI();
        } else {
            throw new Error(data.message || '');
        }
    } catch (error) {
        console.error(':', error);
        showMessage(': ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// 
async function setTopic() {
    const topic = elements.topicInput.value.trim();
    
    if (!topic) {
        showMessage('', 'warning');
        return;
    }
    
    if (!state.initialized) {
        showMessage('', 'warning');
        return;
    }
    
    try {
        showLoading('...');
        
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
            showMessage('', 'success');
            
            // 
            elements.debateContent.innerHTML = `
                <div class="text-center py-5">
                    <h4>${topic}</h4>
                    <p class="text-muted">""</p>
                </div>
            `;
            
            updateUI();
        } else {
            throw new Error(data.message || '');
        }
    } catch (error) {
        console.error(':', error);
        showMessage(': ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// 
async function startDebate() {
    if (!state.topic) {
        showMessage('', 'warning');
        return;
    }
    
    state.currentRound = 0;
    elements.debateContent.innerHTML = '';
    await runDebateRound();
}

// 
async function nextRound() {
    await runDebateRound();
}

// 
async function runDebateRound() {
    if (!state.initialized || !state.topic) {
        showMessage('', 'warning');
        return;
    }
    
    if (state.loading) {
        showMessage('', 'info');
        return;
    }
    
    try {
        state.loading = true;
        state.debating = true;
        updateUI();
        
        showLoading(state.currentRound === 0 ? 
            '10-30...' : 
            '...'
        );
        
        const response = await fetch('/api/debate_round', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                topic: state.topic || ''
            }),
            // 
            signal: AbortSignal.timeout(60000)  // 60
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            state.currentRound = data.round;
            elements.currentRound.textContent = data.round;
            
            // 
            displayDebateRound(data);
            
            //  Agent 
            if (data.agent_states) {
                updateAgentStates(data.agent_states);
            }
            
            // 
            if (data.debate_ended) {
                state.debating = false;
                console.log('Debate ended, summary:', data.summary);
                if (data.summary) {
                    showDebateResult(data.summary);
                } else {
                    // 
                    const endDiv = document.createElement('div');
                    endDiv.className = 'debate-result text-center py-5';
                    endDiv.innerHTML = `
                        <h3></h3>
                        <p class="lead"></p>
                    `;
                    elements.debateContent.appendChild(endDiv);
                }
                showMessage('', 'info');
            } else {
                showMessage(` ${data.round} `, 'success');
            }
        } else {
            throw new Error(data.message || '');
        }
    } catch (error) {
        console.error(':', error);
        
        // 
        if (error.name === 'AbortError') {
            showMessage('', 'error');
        } else if (error.message.includes('fetch')) {
            showMessage('', 'error');
        } else {
            showMessage(': ' + error.message, 'error');
        }
        
        // 
        if (state.currentRound === 0) {
            state.debating = false;
        }
    } finally {
        state.loading = false;
        updateUI();
        hideLoading();
    }
}

// 
function displayDebateRound(data) {
    // 
    if (!data || !data.round) {
        console.error(':', data);
        showMessage('', 'error');
        return;
    }
    
    const roundDiv = document.createElement('div');
    roundDiv.className = 'debate-round';
    roundDiv.innerHTML = `
        <div class="round-header">
            <div class="round-number">${data.round}</div>
            <h4> ${data.round} </h4>
        </div>
    `;
    
    //  responses 
    if (data.responses && Array.isArray(data.responses) && data.responses.length > 0) {
        //  AI 
        data.responses.forEach(response => {
            if (!response || !response.agent_id) {
                console.warn(':', response);
                return;
            }
            
            const agentType = response.agent_id === 'Agent_A' ? 'support' : 
                             response.agent_id === 'Agent_B' ? 'oppose' : 'neutral';
            const agentName = response.agent_id === 'Agent_A' ? ' A' :
                             response.agent_id === 'Agent_B' ? ' B' : ' C';
            
            const responseDiv = document.createElement('div');
            responseDiv.className = `ai-response ${agentType}`;
            
            // 
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
                            : ${(persuasion * 100).toFixed(0)}% | 
                            : ${(attack * 100).toFixed(0)}%
                        </small>
                    </div>
                </div>
                <div class="response-content">
                    ${response.content || '()'}
                </div>
            `;
            
            roundDiv.appendChild(responseDiv);
        });
    } else {
        // 
        const errorDiv = document.createElement('div');
        errorDiv.className = 'alert alert-warning';
        errorDiv.textContent = '';
        roundDiv.appendChild(errorDiv);
    }
    
    elements.debateContent.appendChild(roundDiv);
    
    // 
    elements.debateContent.scrollTop = elements.debateContent.scrollHeight;
}

//  Agent 
function updateAgentStates(states) {
    //  states 
    if (!states || typeof states !== 'object') {
        console.warn('Agent :', states);
        return;
    }
    
    Object.entries(states).forEach(([agentId, state]) => {
        try {
            const suffix = agentId.split('_')[1];
            
            // 
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
            
            // 
            const convictionBar = document.getElementById(`conviction${suffix}`);
            if (convictionBar && state.conviction !== undefined) {
                const convictionPercent = (state.conviction * 100).toFixed(0);
                convictionBar.style.width = convictionPercent + '%';
                const convictionSpan = convictionBar.querySelector('span');
                if (convictionSpan) {
                    convictionSpan.textContent = state.conviction.toFixed(2);
                }
            }
            
            // 
            if (state.has_surrendered) {
                const agentCard = document.getElementById(`agent${suffix}`);
                if (agentCard) {
                    agentCard.style.opacity = '0.6';
                    showMessage(`${agentId.replace('_', ' ')} `, 'warning');
                }
            }
        } catch (error) {
            console.error(` ${agentId} :`, error);
        }
    });
}

// 
function showDebateResult(summary) {
    //  summary 
    if (!summary) {
        console.error('');
        return;
    }
    
    const resultDiv = document.createElement('div');
    resultDiv.className = 'debate-result text-center py-5';
    
    //  HTML
    let scoresHtml = '';
    if (summary.scores && typeof summary.scores === 'object') {
        scoresHtml = Object.entries(summary.scores)
            .sort(([,a], [,b]) => b - a)
            .map(([agent, score]) => `
                <div class="mb-2">
                    ${agent}: ${(score || 0).toFixed(1)} 
                    ${agent === summary.winner ? '<span class="badge bg-warning ms-2"></span>' : ''}
                </div>
            `).join('');
    }
    
    resultDiv.innerHTML = `
        <h3></h3>
        <p class="lead">${summary.verdict || ''}</p>
        <div class="mt-4">
            <h5></h5>
            ${scoresHtml || '<p class="text-muted"></p>'}
        </div>
    `;
    
    elements.debateContent.appendChild(resultDiv);
}

// 
async function resetDebate() {
    if (state.loading) {
        showMessage('', 'info');
        return;
    }
    
    if (!confirm('')) {
        return;
    }
    
    try {
        showLoading('...');
        
        const response = await fetch('/api/reset', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'}
        });
        
        const data = await response.json();
        
        if (data.success) {
            //  -  initialized  true
            state.topic = '';
            state.currentRound = 0;
            state.debating = false;
            state.loading = false;  //  loading 
            
            //  UI
            elements.topicInput.value = '';
            elements.topicDisplay.textContent = '';
            elements.currentRound.textContent = '0';
            elements.debateContent.innerHTML = `
                <div class="welcome-screen">
                    <div class="welcome-icon">
                        <i class="fas fa-robot"></i>
                    </div>
                    <h3> Social Debate AI</h3>
                    <p> AI </p>
                </div>
            `;
            
            // 
            resetAgentStates();
            
            showMessage('', 'success');
            updateUI();
        } else {
            throw new Error(data.message || '');
        }
    } catch (error) {
        console.error(':', error);
        showMessage(': ' + error.message, 'error');
        //  loading 
        state.loading = false;
        updateUI();
    } finally {
        hideLoading();
    }
}

//  Agent 
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
    
    // 
    document.querySelectorAll('.agent-card').forEach(card => {
        card.style.opacity = '1';
    });
}

//  UI 
function updateUI() {
    // 
    elements.startBtn.disabled = !state.initialized || !state.topic || state.loading || state.debating;
    elements.nextBtn.disabled = !state.initialized || !state.topic || state.loading || !state.debating || state.currentRound === 0;
    
    // 
    if (state.debating) {
        elements.debateStatus.textContent = '';
        elements.debateStatus.style.color = 'var(--success-color)';
    } else if (state.topic) {
        elements.debateStatus.textContent = '';
        elements.debateStatus.style.color = 'var(--info-color)';
    } else {
        elements.debateStatus.textContent = '';
        elements.debateStatus.style.color = 'var(--warning-color)';
    }
}

// 
async function exportDebate() {
    if (state.currentRound === 0) {
        showMessage('', 'warning');
        return;
    }
    
    try {
        showLoading('...');
        
        const response = await fetch('/api/export');
        const data = await response.json();
        
        if (data.success) {
            // 
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
            
            showMessage('', 'success');
        } else {
            throw new Error(data.message || '');
        }
    } catch (error) {
        console.error(':', error);
        showMessage(': ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// 
function showStats() {
    showMessage('...', 'info');
}

// 
function showAbout() {
    showMessage('Social Debate AI -  v1.0', 'info');
}

// 
function toggleTheme() {
    document.body.classList.toggle('dark-theme');
    const icon = document.getElementById('themeIcon');
    icon.className = document.body.classList.contains('dark-theme') ? 
                     'fas fa-sun' : 'fas fa-moon';
} 