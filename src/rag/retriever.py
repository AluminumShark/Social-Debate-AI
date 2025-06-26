"""
Enhanced retriever for debate arguments
"""

from pathlib import Path
from typing import List, Dict, Optional
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import numpy as np

class EnhancedRetriever:
    def __init__(self):
        self.base_dir = Path('data/index/enhanced')
        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
        self.stores = {}
        self._load_stores()
    
    def _load_stores(self):
        configs = {
            'high_quality': {
                'path': self.base_dir / 'high_quality',
                'collection': 'hq_pairs'
            },
            'comprehensive': {
                'path': self.base_dir / 'comprehensive', 
                'collection': 'all_discussions'
            }
        }
        
        for name, config in configs.items():
            if config['path'].exists():
                try:
                    self.stores[name] = Chroma(
                        persist_directory=str(config['path']),
                        embedding_function=self.embeddings,
                        collection_name=config['collection']
                    )
                    print(f"Loaded {name} index")
                except Exception as e:
                    print(f"Failed to load {name} index: {e}")
        
        if not self.stores:
            raise RuntimeError("No available indexes found")
    
    def retrieve(self, 
                 query: str, 
                 k: int = 5,
                 index_type: str = 'high_quality',
                 topics: Optional[List[str]] = None,
                 complexity: Optional[str] = None,
                 min_score: int = 0,
                 persuasion_only: bool = False) -> List[Dict]:
        
        if index_type not in self.stores:
            print(f"Index {index_type} not found, using default")
            index_type = list(self.stores.keys())[0]
        
        store = self.stores[index_type]
        
        try:
            docs = store.similarity_search_with_score(query, k=k*2)
            
            results = []
            seen_content = set()
            
            for doc, score in docs:
                content = doc.page_content.strip()
                if content in seen_content:
                    continue
                seen_content.add(content)
                
                metadata = doc.metadata
                
                if self._should_filter(metadata, topics, complexity, min_score, persuasion_only):
                    continue
                
                topics_str = metadata.get('topics', '')
                topics_list = topics_str.split(',') if topics_str else []
                
                results.append({
                    'content': content,
                    'metadata': metadata,
                    'similarity_score': float(1 - score),
                    'type': metadata.get('type', 'unknown'),
                    'topics': topics_list,
                    'complexity': metadata.get('complexity', 'unknown'),
                    'score': metadata.get('score', 0),
                    'title': metadata.get('title', '')
                })
                
                if len(results) >= k:
                    break
            
            results.sort(key=lambda x: (x['similarity_score'], x['score']), reverse=True)
            
            print(f"Retrieved {len(results)} results from {index_type}")
            return results[:k]
            
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []
    
    def _should_filter(self, metadata, topics, complexity, min_score, persuasion_only):
        if topics:
            doc_topics_str = metadata.get('topics', '')
            doc_topics = doc_topics_str.split(',') if doc_topics_str else []
            if not any(topic in doc_topics for topic in topics):
                return True
        
        if complexity and metadata.get('complexity') != complexity:
            return True
        
        if metadata.get('score', 0) < min_score:
            return True
        
        if persuasion_only and not metadata.get('persuasion_success', False):
            return True
        
        return False
    
    def get_topic_distribution(self, query: str, index_type: str = 'high_quality') -> Dict[str, int]:
        if index_type not in self.stores:
            return {}
        
        store = self.stores[index_type]
        docs = store.similarity_search(query, k=50)
        
        topic_count = {}
        for doc in docs:
            topics_str = doc.metadata.get('topics', '')
            topics = topics_str.split(',') if topics_str else []
            for topic in topics:
                topic = topic.strip()
                if topic:
                    topic_count[topic] = topic_count.get(topic, 0) + 1
        
        return dict(sorted(topic_count.items(), key=lambda x: x[1], reverse=True))
    
    def retrieve_by_topic(self, topic: str, k: int = 10, index_type: str = 'high_quality') -> List[Dict]:
        return self.retrieve(
            query=f"discussion about {topic}",
            k=k,
            index_type=index_type,
            topics=[topic]
        )
    
    def retrieve_successful_arguments(self, query: str, k: int = 5) -> List[Dict]:
        return self.retrieve(
            query=query,
            k=k,
            index_type='high_quality',
            persuasion_only=True
        )
    
    def retrieve_diverse_perspectives(self, query: str, k: int = 10) -> List[Dict]:
        results = []
        
        if 'high_quality' in self.stores:
            hq_results = self.retrieve(query, k=k//2, index_type='high_quality')
            results.extend(hq_results)
        
        if 'comprehensive' in self.stores:
            comp_results = self.retrieve(query, k=k//2, index_type='comprehensive')
            results.extend(comp_results)
        
        seen = set()
        unique_results = []
        for result in results:
            content_hash = hash(result['content'][:100])
            if content_hash not in seen:
                seen.add(content_hash)
                unique_results.append(result)
        
        unique_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return unique_results[:k]
    
    def get_stats(self) -> Dict:
        return {
            'available_indexes': list(self.stores.keys()),
            'total_indexes': len(self.stores)
        }

def create_enhanced_retriever():
    try:
        return EnhancedRetriever()
    except Exception as e:
        print(f"Failed to create enhanced retriever: {e}")
        return SimpleRetrieverAdapter()

class SimpleRetrieverAdapter:
    def __init__(self):
        self.retriever = self._load_simple_retriever()
    
    def _load_simple_retriever(self):
        from .simple_retriever import SimpleRetriever
        return SimpleRetriever()
    
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict]:
        return self.retriever.retrieve(query, top_k)
        
    def retrieve_by_topic(self, topic: str, k: int = 10, **kwargs) -> List[Dict]:
        return self.retrieve(f"discussion about {topic}", k)
    
    def retrieve_successful_arguments(self, query: str, k: int = 5) -> List[Dict]:
        return self.retrieve(query, k)
    
    def retrieve_diverse_perspectives(self, query: str, k: int = 10) -> List[Dict]:
        return self.retrieve(query, k)
    
    def get_topic_distribution(self, query: str, **kwargs) -> Dict[str, int]:
        return {}
    
    def get_stats(self) -> Dict:
        return {'available_indexes': ['simple'], 'total_indexes': 1}
