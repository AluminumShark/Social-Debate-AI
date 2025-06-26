"""
Simple retriever for RAG system
"""

import json
import os
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    text: str
    score: float
    metadata: Dict = None

class SimpleRetriever:
    def __init__(self, index_path: str = None):
        self.index_path = index_path or "src/rag/data/rag/simple_index.json"
        self.documents = []
        self._load_index()
    
    def _load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            print(f"Loaded simple index: {len(self.documents)} documents")
        else:
            print(f"Index file not found: {self.index_path}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if not self.documents:
            return []
        
        query_words = set(query.lower().split())
        
        scored_docs = []
        for doc in self.documents:
            doc_words = set(doc['text'].lower().split())
            
            # Simple overlap scoring
            overlap = len(query_words & doc_words)
            score = overlap / max(len(query_words), 1)
            
            if score > 0:
                scored_docs.append((doc, score))
        
        # Sort by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc, score in scored_docs[:top_k]:
            results.append(RetrievalResult(
                text=doc['text'],
                score=score,
                metadata=doc.get('metadata', {})
            ))
        
        return results
    
    def retrieve_by_topic(self, topic: str, top_k: int = 5) -> List[RetrievalResult]:
        # Filter by topic if available
        topic_docs = []
        for doc in self.documents:
            if 'topic' in doc.get('metadata', {}):
                if topic.lower() in doc['metadata']['topic'].lower():
                    topic_docs.append(RetrievalResult(
                        text=doc['text'],
                        score=1.0,
                        metadata=doc.get('metadata', {})
                    ))
        
        return topic_docs[:top_k]
    
    def get_stats(self) -> Dict:
        return {
            'total_documents': len(self.documents),
            'index_path': self.index_path
        } 