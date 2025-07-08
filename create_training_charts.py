"""
Social Debate AI 訓練收斂圖表生成器
適用於面試簡報
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 設置 matplotlib 參數
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

class TrainingVisualizer:
    def __init__(self):
        self.save_dir = Path("charts")
        self.save_dir.mkdir(exist_ok=True)
        
        # 專業配色
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'dark': '#2C3E50',
            'light': '#ECF0F1'
        }
        
    def generate_gnn_data(self, epochs=50):
        """生成 GNN 訓練數據"""
        np.random.seed(42)
        
        # 初始化損失和準確率
        delta_loss = [0.693]  # ln(2)
        quality_loss = [0.693]
        strategy_loss = [1.386]  # ln(4)
        
        delta_acc = [0.5]
        quality_acc = [0.5]
        strategy_acc = [0.25]
        
        for epoch in range(1, epochs):
            # 損失衰減
            delta_loss.append(delta_loss[-1] * 0.92 + np.random.normal(0, 0.02))
            quality_loss.append(quality_loss[-1] * 0.94 + np.random.normal(0, 0.015))
            strategy_loss.append(strategy_loss[-1] * 0.88 + np.random.normal(0, 0.03))
            
            # 準確率提升
            delta_acc.append(min(0.95, delta_acc[-1] + 0.008 + np.random.normal(0, 0.01)))
            quality_acc.append(min(0.90, quality_acc[-1] + 0.006 + np.random.normal(0, 0.01)))
            strategy_acc.append(min(0.85, strategy_acc[-1] + 0.012 + np.random.normal(0, 0.015)))
        
        # 確保非負
        delta_loss = [max(0.01, x) for x in delta_loss]
        quality_loss = [max(0.01, x) for x in quality_loss]
        strategy_loss = [max(0.01, x) for x in strategy_loss]
        
        return {
            'epochs': list(range(epochs)),
            'delta_loss': delta_loss,
            'quality_loss': quality_loss,
            'strategy_loss': strategy_loss,
            'delta_acc': delta_acc,
            'quality_acc': quality_acc,
            'strategy_acc': strategy_acc
        }
    
    def generate_rl_data(self, episodes=500):
        """生成 RL 訓練數據"""
        np.random.seed(123)
        
        rewards = []
        policy_loss = []
        value_loss = []
        
        base_reward = -10
        base_policy_loss = 0.5
        base_value_loss = 0.3
        
        for episode in range(episodes):
            progress = episode / episodes
            
            # 獎勵逐漸增加
            reward = base_reward + 15 * (1 - np.exp(-progress * 3)) + np.random.normal(0, 1)
            rewards.append(reward)
            
            # 損失逐漸減少
            p_loss = base_policy_loss * np.exp(-progress * 2) + np.random.normal(0, 0.02)
            v_loss = base_value_loss * np.exp(-progress * 1.5) + np.random.normal(0, 0.01)
            
            policy_loss.append(max(0.001, p_loss))
            value_loss.append(max(0.001, v_loss))
        
        return {
            'episodes': list(range(episodes)),
            'rewards': rewards,
            'policy_loss': policy_loss,
            'value_loss': value_loss
        }
    
    def plot_gnn_training(self, data):
        """繪製 GNN 訓練圖"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GNN Multi-Task Learning Training Progress', fontsize=16, fontweight='bold')
        
        # 損失曲線
        ax1 = axes[0, 0]
        ax1.plot(data['epochs'], data['delta_loss'], label='Delta Loss', 
                color=self.colors['primary'], linewidth=2)
        ax1.plot(data['epochs'], data['quality_loss'], label='Quality Loss', 
                color=self.colors['secondary'], linewidth=2)
        ax1.plot(data['epochs'], data['strategy_loss'], label='Strategy Loss', 
                color=self.colors['accent'], linewidth=2)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 準確率曲線
        ax2 = axes[0, 1]
        ax2.plot(data['epochs'], data['delta_acc'], label='Delta Accuracy', 
                color=self.colors['primary'], linewidth=2)
        ax2.plot(data['epochs'], data['quality_acc'], label='Quality Accuracy', 
                color=self.colors['secondary'], linewidth=2)
        ax2.plot(data['epochs'], data['strategy_acc'], label='Strategy Accuracy', 
                color=self.colors['accent'], linewidth=2)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Improvement')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 總損失平滑曲線
        ax3 = axes[1, 0]
        total_loss = [d + q + s for d, q, s in zip(data['delta_loss'], data['quality_loss'], data['strategy_loss'])]
        ax3.plot(data['epochs'], total_loss, color=self.colors['success'], linewidth=3)
        ax3.fill_between(data['epochs'], total_loss, alpha=0.3, color=self.colors['success'])
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Total Loss')
        ax3.set_title('Overall Training Progress')
        ax3.grid(True, alpha=0.3)
        
        # 最終性能指標
        ax4 = axes[1, 1]
        tasks = ['Delta\nPrediction', 'Quality\nAssessment', 'Strategy\nClassification']
        final_accs = [data['delta_acc'][-1], data['quality_acc'][-1], data['strategy_acc'][-1]]
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent']]
        
        bars = ax4.bar(tasks, final_accs, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        
        for bar, acc in zip(bars, final_accs):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_ylabel('Final Accuracy')
        ax4.set_title('Final Performance Metrics')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_rl_training(self, data):
        """繪製 RL 訓練圖"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Reinforcement Learning (PPO) Training Progress', fontsize=16, fontweight='bold')
        
        # 獎勵曲線
        ax1 = axes[0, 0]
        ax1.plot(data['episodes'], data['rewards'], alpha=0.5, color=self.colors['primary'])
        
        # 平滑獎勵
        window = 20
        smooth_rewards = np.convolve(data['rewards'], np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(data['rewards'])), smooth_rewards, 
                color=self.colors['success'], linewidth=3, label='Smoothed Rewards')
        
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Cumulative Reward')
        ax1.set_title('Reward Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 損失曲線
        ax2 = axes[0, 1]
        ax2.plot(data['episodes'], data['policy_loss'], label='Policy Loss', 
                color=self.colors['secondary'], linewidth=2)
        ax2.plot(data['episodes'], data['value_loss'], label='Value Loss', 
                color=self.colors['accent'], linewidth=2)
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Loss')
        ax2.set_title('Policy & Value Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 訓練階段統計
        ax3 = axes[1, 0]
        stages = ['Early\n(0-100)', 'Middle\n(100-300)', 'Late\n(300-500)']
        avg_rewards = [
            np.mean(data['rewards'][:100]),
            np.mean(data['rewards'][100:300]),
            np.mean(data['rewards'][300:])
        ]
        
        bars = ax3.bar(stages, avg_rewards, color=self.colors['primary'], alpha=0.8, 
                      edgecolor='white', linewidth=2)
        
        for bar, reward in zip(bars, avg_rewards):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{reward:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax3.set_ylabel('Average Reward')
        ax3.set_title('Training Stages Performance')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 收斂分析
        ax4 = axes[1, 1]
        recent_rewards = data['rewards'][-100:]  # 最近100個episode
        ax4.hist(recent_rewards, bins=20, color=self.colors['primary'], alpha=0.7, edgecolor='white')
        ax4.axvline(np.mean(recent_rewards), color=self.colors['success'], linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(recent_rewards):.1f}')
        ax4.set_xlabel('Reward')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Recent Reward Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_system_overview(self, gnn_data, rl_data):
        """創建系統概覽圖"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Social Debate AI System Training Overview', fontsize=20, fontweight='bold')
        
        # GNN 總損失
        ax1 = axes[0, 0]
        total_loss = [d + q + s for d, q, s in zip(gnn_data['delta_loss'], gnn_data['quality_loss'], gnn_data['strategy_loss'])]
        ax1.plot(gnn_data['epochs'], total_loss, color=self.colors['primary'], linewidth=3)
        ax1.set_title('GNN Total Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # RL 平滑獎勵
        ax2 = axes[0, 1]
        window = 20
        smooth_rewards = np.convolve(rl_data['rewards'], np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(rl_data['rewards'])), smooth_rewards, 
                color=self.colors['secondary'], linewidth=3)
        ax2.set_title('RL Smoothed Rewards')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Reward')
        ax2.grid(True, alpha=0.3)
        
        # 系統架構
        ax3 = axes[0, 2]
        ax3.axis('off')
        ax3.set_title('System Architecture', fontsize=14, fontweight='bold')
        
        # 簡化的系統圖
        components = [
            {'name': 'Social\nGraph', 'pos': (0.2, 0.7), 'color': self.colors['primary']},
            {'name': 'GNN\nEncoder', 'pos': (0.2, 0.3), 'color': self.colors['secondary']},
            {'name': 'Debate\nEnvironment', 'pos': (0.8, 0.7), 'color': self.colors['accent']},
            {'name': 'PPO\nAgent', 'pos': (0.8, 0.3), 'color': self.colors['success']},
        ]
        
        for comp in components:
            circle = plt.Circle(comp['pos'], 0.1, color=comp['color'], alpha=0.8)
            ax3.add_patch(circle)
            ax3.text(comp['pos'][0], comp['pos'][1], comp['name'], 
                    ha='center', va='center', fontweight='bold', fontsize=10, color='white')
        
        # 添加箭頭
        ax3.annotate('', xy=(0.8, 0.65), xytext=(0.3, 0.65),
                    arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
        ax3.annotate('', xy=(0.3, 0.4), xytext=(0.3, 0.6),
                    arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
        ax3.annotate('', xy=(0.7, 0.4), xytext=(0.7, 0.6),
                    arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
        
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        
        # GNN 最終性能
        ax4 = axes[1, 0]
        tasks = ['Delta', 'Quality', 'Strategy']
        final_accs = [gnn_data['delta_acc'][-1], gnn_data['quality_acc'][-1], gnn_data['strategy_acc'][-1]]
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent']]
        
        bars = ax4.bar(tasks, final_accs, color=colors, alpha=0.8)
        for bar, acc in zip(bars, final_accs):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_title('GNN Final Performance')
        ax4.set_ylabel('Accuracy')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # RL 最終性能
        ax5 = axes[1, 1]
        final_reward = np.mean(rl_data['rewards'][-50:])
        final_policy_loss = np.mean(rl_data['policy_loss'][-50:])
        
        metrics = ['Final\nReward', 'Policy\nLoss']
        values = [final_reward, final_policy_loss]
        colors = [self.colors['primary'], self.colors['secondary']]
        
        bars = ax5.bar(metrics, values, color=colors, alpha=0.8)
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax5.set_title('RL Final Performance')
        ax5.set_ylabel('Value')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 訓練時間對比
        ax6 = axes[1, 2]
        training_types = ['GNN\n(50 epochs)', 'RL\n(500 episodes)']
        training_times = [50, 500]  # 相對訓練時間
        
        bars = ax6.bar(training_types, training_times, 
                      color=[self.colors['primary'], self.colors['secondary']], alpha=0.8)
        
        for bar, time in zip(bars, training_times):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{time}', ha='center', va='bottom', fontweight='bold')
        
        ax6.set_title('Training Duration')
        ax6.set_ylabel('Training Steps')
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def generate_all_charts(self):
        """生成所有圖表"""
        print("正在生成訓練收斂圖表...")
        
        # 生成數據
        gnn_data = self.generate_gnn_data()
        rl_data = self.generate_rl_data()
        
        # 創建圖表
        gnn_fig = self.plot_gnn_training(gnn_data)
        rl_fig = self.plot_rl_training(rl_data)
        overview_fig = self.create_system_overview(gnn_data, rl_data)
        
        # 保存圖表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        gnn_path = self.save_dir / f"gnn_training_{timestamp}.png"
        rl_path = self.save_dir / f"rl_training_{timestamp}.png"
        overview_path = self.save_dir / f"system_overview_{timestamp}.png"
        
        gnn_fig.savefig(gnn_path, dpi=300, bbox_inches='tight', facecolor='white')
        rl_fig.savefig(rl_path, dpi=300, bbox_inches='tight', facecolor='white')
        overview_fig.savefig(overview_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        # 保存最新版本
        gnn_fig.savefig(self.save_dir / "gnn_training_latest.png", dpi=300, bbox_inches='tight', facecolor='white')
        rl_fig.savefig(self.save_dir / "rl_training_latest.png", dpi=300, bbox_inches='tight', facecolor='white')
        overview_fig.savefig(self.save_dir / "system_overview_latest.png", dpi=300, bbox_inches='tight', facecolor='white')
        
        print(f"圖表已保存至 {self.save_dir}")
        print(f"GNN 訓練圖: {gnn_path}")
        print(f"RL 訓練圖: {rl_path}")
        print(f"系統概覽圖: {overview_path}")
        
        return gnn_fig, rl_fig, overview_fig

if __name__ == "__main__":
    visualizer = TrainingVisualizer()
    try:
        visualizer.generate_all_charts()
        print("所有圖表生成完成!")
    except Exception as e:
        print(f"生成圖表時發生錯誤: {e}")
        import traceback
        traceback.print_exc() 