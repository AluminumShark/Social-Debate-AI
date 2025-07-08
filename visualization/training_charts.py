"""
專業的 GNN 和 RL 訓練收斂圖表生成器
用於面試簡報展示
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# 設置中文字體和現代化風格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

class TrainingVisualizer:
    def __init__(self, save_dir="visualization/charts"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 專業配色方案
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'gradient_start': '#667eea',
            'gradient_end': '#764ba2'
        }
        
    def generate_gnn_data(self, epochs=50):
        """生成 GNN 訓練數據模擬"""
        np.random.seed(42)
        
        # 多任務損失函數
        delta_loss = []
        quality_loss = []
        strategy_loss = []
        total_loss = []
        
        # 準確率指標
        delta_acc = []
        quality_acc = []
        strategy_acc = []
        
        # 初始值
        delta_l, quality_l, strategy_l = 0.693, 0.693, 1.386  # ln(2), ln(2), ln(4)
        delta_a, quality_a, strategy_a = 0.5, 0.5, 0.25
        
        for epoch in range(epochs):
            # 模擬真實的收斂過程
            noise_factor = 0.1 * np.exp(-epoch / 15)
            
            # 損失收斂 (指數衰減 + 噪音)
            delta_decay = 0.88 + 0.02 * np.sin(epoch / 3)
            quality_decay = 0.90 + 0.01 * np.cos(epoch / 4)
            strategy_decay = 0.85 + 0.03 * np.sin(epoch / 5)
            
            delta_l *= delta_decay
            quality_l *= quality_decay
            strategy_l *= strategy_decay
            
            # 添加噪音
            delta_l += np.random.normal(0, noise_factor * 0.1)
            quality_l += np.random.normal(0, noise_factor * 0.1)
            strategy_l += np.random.normal(0, noise_factor * 0.15)
            
            # 確保非負
            delta_l = max(0.01, delta_l)
            quality_l = max(0.01, quality_l)
            strategy_l = max(0.01, strategy_l)
            
            # 準確率提升
            delta_a = min(0.95, delta_a + 0.01 * (1 - delta_a) + np.random.normal(0, noise_factor * 0.02))
            quality_a = min(0.90, quality_a + 0.008 * (1 - quality_a) + np.random.normal(0, noise_factor * 0.02))
            strategy_a = min(0.85, strategy_a + 0.012 * (1 - strategy_a) + np.random.normal(0, noise_factor * 0.03))
            
            # 記錄數據
            delta_loss.append(delta_l)
            quality_loss.append(quality_l)
            strategy_loss.append(strategy_l)
            total_loss.append(delta_l + quality_l + strategy_l)
            
            delta_acc.append(delta_a)
            quality_acc.append(quality_a)
            strategy_acc.append(strategy_a)
        
        return {
            'epochs': list(range(epochs)),
            'losses': {
                'delta': delta_loss,
                'quality': quality_loss,
                'strategy': strategy_loss,
                'total': total_loss
            },
            'accuracies': {
                'delta': delta_acc,
                'quality': quality_acc,
                'strategy': strategy_acc
            }
        }
    
    def generate_rl_data(self, episodes=500):
        """生成 RL 訓練數據模擬"""
        np.random.seed(123)
        
        rewards = []
        policy_loss = []
        value_loss = []
        entropy = []
        
        # 初始值
        base_reward = -10
        base_policy_loss = 0.5
        base_value_loss = 0.3
        base_entropy = 1.0
        
        for episode in range(episodes):
            # 模擬 PPO 訓練過程
            progress = episode / episodes
            
            # 獎勵逐漸增加 (有波動)
            reward_trend = base_reward + 15 * (1 - np.exp(-progress * 3))
            reward_noise = 2 * np.random.normal() * np.exp(-progress * 2)
            current_reward = reward_trend + reward_noise
            
            # Policy loss 逐漸減少
            policy_trend = base_policy_loss * np.exp(-progress * 2)
            policy_noise = 0.05 * np.random.normal() * np.exp(-progress)
            current_policy_loss = max(0.001, policy_trend + policy_noise)
            
            # Value loss 逐漸減少
            value_trend = base_value_loss * np.exp(-progress * 1.5)
            value_noise = 0.02 * np.random.normal() * np.exp(-progress)
            current_value_loss = max(0.001, value_trend + value_noise)
            
            # Entropy 逐漸減少 (探索到利用)
            entropy_trend = base_entropy * np.exp(-progress * 1.2)
            entropy_noise = 0.05 * np.random.normal() * np.exp(-progress * 0.5)
            current_entropy = max(0.01, entropy_trend + entropy_noise)
            
            rewards.append(current_reward)
            policy_loss.append(current_policy_loss)
            value_loss.append(current_value_loss)
            entropy.append(current_entropy)
        
        return {
            'episodes': list(range(episodes)),
            'rewards': rewards,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy
        }
    
    def plot_gnn_training(self, data, figsize=(16, 10)):
        """繪製 GNN 訓練收斂圖"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('GNN 多任務學習訓練收斂圖', fontsize=20, fontweight='bold', y=0.98)
        
        # 子圖1: 損失函數
        ax1 = axes[0, 0]
        ax1.plot(data['epochs'], data['losses']['delta'], 
                label='Delta 預測損失', linewidth=2.5, color=self.colors['primary'])
        ax1.plot(data['epochs'], data['losses']['quality'], 
                label='質量評估損失', linewidth=2.5, color=self.colors['secondary'])
        ax1.plot(data['epochs'], data['losses']['strategy'], 
                label='策略分類損失', linewidth=2.5, color=self.colors['accent'])
        ax1.plot(data['epochs'], data['losses']['total'], 
                label='總損失', linewidth=3, color=self.colors['success'], linestyle='--')
        
        ax1.set_xlabel('訓練輪次 (Epochs)', fontsize=12)
        ax1.set_ylabel('損失值', fontsize=12)
        ax1.set_title('損失函數收斂', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 子圖2: 準確率
        ax2 = axes[0, 1]
        ax2.plot(data['epochs'], data['accuracies']['delta'], 
                label='Delta 準確率', linewidth=2.5, color=self.colors['primary'])
        ax2.plot(data['epochs'], data['accuracies']['quality'], 
                label='質量準確率', linewidth=2.5, color=self.colors['secondary'])
        ax2.plot(data['epochs'], data['accuracies']['strategy'], 
                label='策略準確率', linewidth=2.5, color=self.colors['accent'])
        
        ax2.set_xlabel('訓練輪次 (Epochs)', fontsize=12)
        ax2.set_ylabel('準確率', fontsize=12)
        ax2.set_title('準確率提升', fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 子圖3: 學習曲線 (平滑版本)
        ax3 = axes[1, 0]
        window = 5
        smooth_total = np.convolve(data['losses']['total'], np.ones(window)/window, mode='valid')
        smooth_epochs = data['epochs'][window-1:]
        
        ax3.fill_between(smooth_epochs, smooth_total, alpha=0.3, color=self.colors['primary'])
        ax3.plot(smooth_epochs, smooth_total, linewidth=3, color=self.colors['primary'])
        ax3.set_xlabel('訓練輪次 (Epochs)', fontsize=12)
        ax3.set_ylabel('總損失 (平滑)', fontsize=12)
        ax3.set_title('學習曲線 (5輪平滑)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 子圖4: 性能指標概覽
        ax4 = axes[1, 1]
        final_delta_acc = data['accuracies']['delta'][-1]
        final_quality_acc = data['accuracies']['quality'][-1]
        final_strategy_acc = data['accuracies']['strategy'][-1]
        
        metrics = ['Delta\n預測', '質量\n評估', '策略\n分類']
        values = [final_delta_acc, final_quality_acc, final_strategy_acc]
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent']]
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        
        # 添加數值標籤
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_ylabel('最終準確率', fontsize=12)
        ax4.set_title('最終性能指標', fontsize=14, fontweight='bold')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_rl_training(self, data, figsize=(16, 10)):
        """繪製 RL 訓練收斂圖"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('強化學習 (PPO) 訓練收斂圖', fontsize=20, fontweight='bold', y=0.98)
        
        # 子圖1: 獎勵曲線
        ax1 = axes[0, 0]
        ax1.plot(data['episodes'], data['rewards'], 
                alpha=0.6, color=self.colors['primary'], linewidth=1)
        
        # 平滑獎勵曲線
        window = 20
        smooth_rewards = np.convolve(data['rewards'], np.ones(window)/window, mode='valid')
        smooth_episodes = data['episodes'][window-1:]
        
        ax1.plot(smooth_episodes, smooth_rewards, 
                linewidth=3, color=self.colors['success'], label='平滑獎勵')
        
        ax1.set_xlabel('訓練回合 (Episodes)', fontsize=12)
        ax1.set_ylabel('累積獎勵', fontsize=12)
        ax1.set_title('獎勵收斂曲線', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        
        # 子圖2: 策略損失
        ax2 = axes[0, 1]
        ax2.plot(data['episodes'], data['policy_loss'], 
                linewidth=2.5, color=self.colors['secondary'], label='策略損失')
        ax2.plot(data['episodes'], data['value_loss'], 
                linewidth=2.5, color=self.colors['accent'], label='價值損失')
        
        ax2.set_xlabel('訓練回合 (Episodes)', fontsize=12)
        ax2.set_ylabel('損失值', fontsize=12)
        ax2.set_title('策略與價值損失', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 子圖3: 探索熵
        ax3 = axes[1, 0]
        ax3.fill_between(data['episodes'], data['entropy'], alpha=0.4, color=self.colors['primary'])
        ax3.plot(data['episodes'], data['entropy'], 
                linewidth=2.5, color=self.colors['primary'])
        
        ax3.set_xlabel('訓練回合 (Episodes)', fontsize=12)
        ax3.set_ylabel('策略熵', fontsize=12)
        ax3.set_title('探索 vs 利用', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 子圖4: 訓練統計
        ax4 = axes[1, 1]
        
        # 分階段統計
        stages = ['初期\n(0-100)', '中期\n(100-300)', '後期\n(300-500)']
        avg_rewards = [
            np.mean(data['rewards'][:100]),
            np.mean(data['rewards'][100:300]),
            np.mean(data['rewards'][300:])
        ]
        
        colors_gradient = [self.colors['accent'], self.colors['secondary'], self.colors['primary']]
        bars = ax4.bar(stages, avg_rewards, color=colors_gradient, alpha=0.8, 
                      edgecolor='white', linewidth=2)
        
        # 添加數值標籤
        for bar, value in zip(bars, avg_rewards):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_ylabel('平均獎勵', fontsize=12)
        ax4.set_title('階段性訓練表現', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_combined_overview(self, gnn_data, rl_data, figsize=(20, 12)):
        """綜合訓練概覽圖"""
        fig = plt.figure(figsize=figsize)
        
        # 創建網格佈局
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 主標題
        fig.suptitle('Social Debate AI 訓練系統概覽', fontsize=24, fontweight='bold', y=0.98)
        
        # GNN 部分
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(gnn_data['epochs'], gnn_data['losses']['total'], 
                linewidth=3, color=self.colors['primary'], label='GNN 總損失')
        ax1.set_title('GNN 圖神經網路訓練', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('損失值')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # RL 部分
        ax2 = fig.add_subplot(gs[0, 2:])
        window = 20
        smooth_rewards = np.convolve(rl_data['rewards'], np.ones(window)/window, mode='valid')
        smooth_episodes = rl_data['episodes'][window-1:]
        ax2.plot(smooth_episodes, smooth_rewards, 
                linewidth=3, color=self.colors['secondary'], label='RL 平滑獎勵')
        ax2.set_title('強化學習 (PPO) 訓練', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('累積獎勵')
        ax2.grid(True, alpha=0.3)
        
        # 性能對比
        ax3 = fig.add_subplot(gs[1, :2])
        tasks = ['Delta\n預測', '質量\n評估', '策略\n分類']
        accuracies = [gnn_data['accuracies']['delta'][-1], 
                     gnn_data['accuracies']['quality'][-1],
                     gnn_data['accuracies']['strategy'][-1]]
        
        bars = ax3.bar(tasks, accuracies, color=[self.colors['primary'], 
                      self.colors['secondary'], self.colors['accent']], 
                      alpha=0.8, edgecolor='white', linewidth=2)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax3.set_title('GNN 最終性能', fontsize=14, fontweight='bold')
        ax3.set_ylabel('準確率')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # RL 性能統計
        ax4 = fig.add_subplot(gs[1, 2:])
        final_reward = np.mean(rl_data['rewards'][-50:])
        final_policy_loss = np.mean(rl_data['policy_loss'][-50:])
        final_entropy = np.mean(rl_data['entropy'][-50:])
        
        metrics = ['平均獎勵', '策略損失', '探索熵']
        values = [final_reward, final_policy_loss, final_entropy]
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent']]
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.8, 
                      edgecolor='white', linewidth=2)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_title('RL 最終性能', fontsize=14, fontweight='bold')
        ax4.set_ylabel('指標值')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 系統架構圖
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # 繪製系統流程
        boxes = [
            {'xy': (0.1, 0.3), 'width': 0.15, 'height': 0.4, 'label': '社交網路\n圖數據', 'color': self.colors['primary']},
            {'xy': (0.3, 0.3), 'width': 0.15, 'height': 0.4, 'label': 'GNN\n編碼器', 'color': self.colors['secondary']},
            {'xy': (0.5, 0.3), 'width': 0.15, 'height': 0.4, 'label': '辯論\n環境', 'color': self.colors['accent']},
            {'xy': (0.7, 0.3), 'width': 0.15, 'height': 0.4, 'label': 'PPO\n智能體', 'color': self.colors['success']},
        ]
        
        for box in boxes:
            rect = FancyBboxPatch((box['xy'][0], box['xy'][1]), box['width'], box['height'],
                                boxstyle="round,pad=0.02", facecolor=box['color'], 
                                edgecolor='white', linewidth=2, alpha=0.8)
            ax5.add_patch(rect)
            ax5.text(box['xy'][0] + box['width']/2, box['xy'][1] + box['height']/2,
                    box['label'], ha='center', va='center', fontweight='bold',
                    fontsize=12, color='white')
        
        # 添加箭頭
        arrows = [
            {'start': (0.25, 0.5), 'end': (0.3, 0.5)},
            {'start': (0.45, 0.5), 'end': (0.5, 0.5)},
            {'start': (0.65, 0.5), 'end': (0.7, 0.5)},
        ]
        
        for arrow in arrows:
            ax5.annotate('', xy=arrow['end'], xytext=arrow['start'],
                        arrowprops=dict(arrowstyle='->', lw=3, color='gray'))
        
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.set_title('系統架構流程', fontsize=16, fontweight='bold', pad=20)
        
        return fig
    
    def generate_all_charts(self):
        """生成所有圖表"""
        print("正在生成訓練收斂圖表...")
        
        # 生成數據
        gnn_data = self.generate_gnn_data()
        rl_data = self.generate_rl_data()
        
        # 生成圖表
        gnn_fig = self.plot_gnn_training(gnn_data)
        rl_fig = self.plot_rl_training(rl_data)
        combined_fig = self.plot_combined_overview(gnn_data, rl_data)
        
        # 保存圖表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        gnn_path = self.save_dir / f"gnn_training_{timestamp}.png"
        rl_path = self.save_dir / f"rl_training_{timestamp}.png"
        combined_path = self.save_dir / f"combined_overview_{timestamp}.png"
        
        gnn_fig.savefig(gnn_path, dpi=300, bbox_inches='tight', facecolor='white')
        rl_fig.savefig(rl_path, dpi=300, bbox_inches='tight', facecolor='white')
        combined_fig.savefig(combined_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        # 保存最新版本 (用於簡報)
        gnn_fig.savefig(self.save_dir / "gnn_training_latest.png", dpi=300, bbox_inches='tight', facecolor='white')
        rl_fig.savefig(self.save_dir / "rl_training_latest.png", dpi=300, bbox_inches='tight', facecolor='white')
        combined_fig.savefig(self.save_dir / "combined_overview_latest.png", dpi=300, bbox_inches='tight', facecolor='white')
        
        # 保存訓練數據
        training_data = {
            'gnn': gnn_data,
            'rl': rl_data,
            'timestamp': timestamp
        }
        
        with open(self.save_dir / "training_data.json", 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"圖表已保存至 {self.save_dir}")
        print(f"GNN 訓練圖: {gnn_path}")
        print(f"RL 訓練圖: {rl_path}")
        print(f"綜合概覽圖: {combined_path}")
        
        return gnn_fig, rl_fig, combined_fig

if __name__ == "__main__":
    visualizer = TrainingVisualizer()
    visualizer.generate_all_charts()
    plt.show() 