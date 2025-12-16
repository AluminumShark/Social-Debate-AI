# Social Debate AI Documentation

This directory contains comprehensive documentation for the Social Debate AI system.

## Documentation Structure

```
docs/
├── README.md                    # This file - documentation index
├── architecture/                # System architecture documentation
│   ├── OVERVIEW.md             # High-level system overview
│   ├── LANGGRAPH.md            # LangGraph orchestration architecture
│   └── DATA_FLOW.md            # Data flow and state management
├── guides/                      # User and developer guides
│   ├── QUICKSTART.md           # 5-minute getting started guide
│   ├── CONFIGURATION.md        # Configuration reference
│   ├── TRAINING.md             # Model training guide
│   └── DEPLOYMENT.md           # Production deployment guide
├── api/                         # API documentation
│   └── REST_API.md             # Flask REST API reference
└── modules/                     # Module-specific documentation
    ├── RAG.md                   # RAG retrieval system
    ├── GNN.md                   # Graph Neural Network module
    ├── RL.md                    # Reinforcement Learning module
    └── SCORING.md               # Debate scoring system
```

## Quick Links

### Getting Started
- [Quick Start Guide](guides/QUICKSTART.md) - Get up and running in 5 minutes
- [Configuration Guide](guides/CONFIGURATION.md) - System configuration options

### Architecture
- [System Overview](architecture/OVERVIEW.md) - High-level architecture
- [LangGraph Orchestration](architecture/LANGGRAPH.md) - **NEW** LangGraph-based workflow
- [Data Flow](architecture/DATA_FLOW.md) - State management and data flow

### API Reference
- [REST API](api/REST_API.md) - Flask API endpoints

### Module Documentation
- [RAG Module](modules/RAG.md) - Retrieval Augmented Generation
- [GNN Module](modules/GNN.md) - Graph Neural Network
- [RL Module](modules/RL.md) - Reinforcement Learning
- [Scoring System](modules/SCORING.md) - Debate evaluation and scoring

### Deployment
- [Training Guide](guides/TRAINING.md) - Model training instructions
- [Deployment Guide](guides/DEPLOYMENT.md) - Production deployment

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.2.0 | 2024-12 | Added LangGraph orchestration, restructured documentation |
| 0.1.0 | 2024-11 | Initial release with manual orchestration |

