# 📚 Social Debate AI Documentation

*English | [中文](#中文版本)*

Welcome to the Social Debate AI documentation! This comprehensive guide covers system architecture, module references, and practical guides.

---

## 📑 Documentation Structure

The documentation is organized into four main sections:

1.  **📖 Core Docs**: Deep learning study notes and theoretical background.
2.  **🏗️ Architecture**: System design, LangGraph workflow, and data flow.
3.  **📦 Modules**: Detailed references for GNN, RL, RAG, and Scoring modules.
4.  **📋 Guides**: Quick start, configuration, training, and deployment guides.

---

## 🚀 Quick Navigation

### 📖 Core Learning Resource

| Document | Description |
|----------|-------------|
| **[LEARNING_NOTE.md](LEARNING_NOTE.md)** | 🇬🇧 **English** - Comprehensive deep learning notes (GNN, PPO, RAG, LangGraph) |
| **[LEARNING_NOTE_zh-TW.md](LEARNING_NOTE_zh-TW.md)** | 🇹🇼 **Traditional Chinese** - 深度學習完整筆記（含原理與實作細節） |

> **Start here!** These notes are the most complete resource for understanding the technical details of the project.

---

### 🏗️ Architecture

| Document | Description |
|----------|-------------|
| [System Overview](architecture/OVERVIEW.md) | High-level architecture and component overview |
| [LangGraph Orchestration](architecture/LANGGRAPH.md) | StateGraph-based workflow engine |
| [Data Flow](architecture/DATA_FLOW.md) | State management and data flow |

---

### 📦 Module Quick Reference

| Module | Description | Deep Dive |
|--------|-------------|-----------|
| [GNN Module](modules/GNN.md) | Graph Neural Network for social analysis | [LEARNING_NOTE §2](LEARNING_NOTE.md#part-2-gnn-deep-dive) |
| [RL Module](modules/RL.md) | PPO reinforcement learning for strategy | [LEARNING_NOTE §3](LEARNING_NOTE.md#part-3-rl-deep-dive) |
| [RAG Module](modules/RAG.md) | Evidence retrieval system | [LEARNING_NOTE §4](LEARNING_NOTE.md#part-4-rag-system) |
| [Scoring System](modules/SCORING.md) | Debate evaluation and victory determination | - |

---

### 📋 Practical Guides

| Guide | Description |
|-------|-------------|
| [Quick Start](guides/QUICKSTART.md) | Get up and running in 5 minutes |
| [Configuration](guides/CONFIGURATION.md) | System configuration reference |
| [Training](guides/TRAINING.md) | Model training instructions |
| [Deployment](guides/DEPLOYMENT.md) | Production deployment guide |

---

### 🔌 API Reference

| Document | Description |
|----------|-------------|
| [REST API](api/REST_API.md) | Flask API endpoints reference |

---

## 📊 Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.2.1 | 2025-12 | Restructured docs with separate English/Chinese learning notes |
| 0.2.0 | 2025-12 | Added LangGraph orchestration, restructured documentation |
| 0.1.0 | 2025-07 | Initial release with manual orchestration |

---

<a name="中文版本"></a>

# 📚 Social Debate AI 文檔

*[English](#-social-debate-ai-documentation) | 中文*

歡迎使用 Social Debate AI 文檔！本指南涵蓋系統架構、模組參考和實用指南。

---

## 📑 文檔結構

文檔主要分為四個部分：

1.  **📖 核心文檔**：深度學習筆記與理論背景。
2.  **🏗️ 架構**：系統設計、LangGraph 工作流與資料流。
3.  **📦 模組**：GNN、RL、RAG 與評分模組的詳細參考。
4.  **📋 指南**：快速開始、配置、訓練與部署指南。

---

## 🚀 快速導航

### 📖 核心學習資源

| 文檔 | 說明 |
|------|------|
| **[LEARNING_NOTE.md](LEARNING_NOTE.md)** | 🇬🇧 **英文版** - 完整深度學習筆記（GNN、PPO、RAG、LangGraph） |
| **[LEARNING_NOTE_zh-TW.md](LEARNING_NOTE_zh-TW.md)** | 🇹🇼 **繁體中文版** - 深度學習完整筆記（含原理與實作細節） |

> **從這裡開始！** 這是了解專案技術細節最完整的資源。

---

### 🏗️ 架構

| 文檔 | 說明 |
|------|------|
| [系統概覽](architecture/OVERVIEW.md) | 高層架構和組件概述 |
| [LangGraph 編排](architecture/LANGGRAPH.md) | 基於 StateGraph 的工作流引擎 |
| [資料流](architecture/DATA_FLOW.md) | 狀態管理和資料流 |

---

### 📦 模組快速參考

| 模組 | 說明 | 深入了解 |
|------|------|----------|
| [GNN 模組](modules/GNN.md) | 社交分析圖神經網路 | [中文筆記 §2](LEARNING_NOTE_zh-TW.md#第二部分-gnn-深度解析) |
| [RL 模組](modules/RL.md) | PPO 強化學習策略 | [中文筆記 §3](LEARNING_NOTE_zh-TW.md#第三部分-rl-深度解析) |
| [RAG 模組](modules/RAG.md) | 證據檢索系統 | [中文筆記 §4](LEARNING_NOTE_zh-TW.md#第四部分-rag-系統) |
| [評分系統](modules/SCORING.md) | 辯論評估和勝負判定 | - |

---

### 📋 實用指南

| 指南 | 說明 |
|------|------|
| [快速開始](guides/QUICKSTART.md) | 5 分鐘內啟動系統 |
| [配置指南](guides/CONFIGURATION.md) | 系統配置參考 |
| [訓練指南](guides/TRAINING.md) | 模型訓練說明 |
| [部署指南](guides/DEPLOYMENT.md) | 生產環境部署 |

---

### 🔌 API 參考

| 文檔 | 說明 |
|------|------|
| [REST API](api/REST_API.md) | Flask API 端點參考 |

---

## 📊 版本歷史

| 版本 | 日期 | 變更 |
|------|------|------|
| 0.2.1 | 2025-12 | 重構文檔，拆分中英文學習筆記 |
| 0.2.0 | 2025-12 | 新增 LangGraph 編排，重組文檔結構 |
| 0.1.0 | 2025-07 | 初始版本，手動編排 |
