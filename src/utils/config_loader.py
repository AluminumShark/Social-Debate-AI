"""
Configuration loader for YAML files
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigLoader:
    _configs: Dict[str, Dict] = {}
    _config_dir: Path = Path("configs")
    
    @classmethod
    def load(cls, config_name: str, config_dir: Optional[str] = None) -> Dict[str, Any]:
        if config_dir:
            cls._config_dir = Path(config_dir)
            
        if config_name in cls._configs:
            return cls._configs[config_name]
        
        config_path = cls._config_dir / f"{config_name}.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
                cls._configs[config_name] = config
                print(f"Loaded config: {config_path}")
                return config
        except FileNotFoundError:
            print(f"Config not found: {config_path}")
            default_config = cls._get_default_config(config_name)
            cls._configs[config_name] = default_config
            return default_config
        except Exception as e:
            print(f"Config load failed: {e}")
            default_config = cls._get_default_config(config_name)
            cls._configs[config_name] = default_config
            return default_config
    
    @classmethod
    def get(cls, config_name: str, key_path: str, default: Any = None) -> Any:
        config = cls.load(config_name)
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
    
    @classmethod
    def _get_default_config(cls, config_name: str) -> Dict[str, Any]:
        defaults = {
            'debate': {
                'debate': {
                    'max_rounds': 5,
                    'agents': ['Agent_A', 'Agent_B', 'Agent_C'],
                    'default_topic': 'Should artificial intelligence be regulated by government?'
                },
                'gpt': {
                    'model': 'gpt-3.5-turbo',
                    'temperature': 0.7,
                    'max_tokens': 500
                }
            },
            'rag': {
                'retriever': {
                    'index_path': 'data/chroma/social_debate',
                    'embedding_model': 'text-embedding-3-small',
                    'top_k': 5,
                    'score_threshold': 0.0
                },
                'simple_retriever': {
                    'index_path': 'src/rag/data/rag/simple_index.json',
                    'top_k': 3
                }
            },
            'gnn': {
                'model': {
                    'input_dim': 768,
                    'hidden_dim': 256,
                    'output_dim': 128,
                    'num_layers': 2,
                    'dropout': 0.1
                },
                'training': {
                    'epochs': 200,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'patience': 20
                }
            },
            'rl': {
                'model': {
                    'hidden_size': 256,
                    'social_dim': 128,
                    'num_strategies': 4
                },
                'training': {
                    'epochs': 3,
                    'learning_rate': 5e-5,
                    'batch_size': 16,
                    'max_length': 512
                },
                'strategies': {
                    'aggressive': 0,
                    'defensive': 1,
                    'analytical': 2,
                    'empathetic': 3
                }
            }
        }
        
        return defaults.get(config_name, {})
    
    @classmethod
    def reset(cls):
        cls._configs.clear()
    
    @classmethod
    def save_config(cls, config_name: str, config: Dict[str, Any], config_dir: Optional[str] = None):
        if config_dir:
            save_dir = Path(config_dir)
        else:
            save_dir = cls._config_dir
            
        save_dir.mkdir(parents=True, exist_ok=True)
        config_path = save_dir / f"{config_name}.yaml"
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"Saved config: {config_path}")


# Compatibility layer
class Config:
    @classmethod
    def load(cls, path: str = "configs/debate.yaml"):
        config_name = Path(path).stem
        return ConfigLoader.load(config_name)
    
    @classmethod
    def reset(cls):
        ConfigLoader.reset()