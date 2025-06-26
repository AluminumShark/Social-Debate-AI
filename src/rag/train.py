"""
RAG index builder
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.rag.build_index import build_chroma_index, build_simple_index
from src.utils.config_loader import ConfigLoader
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Build RAG retrieval index")
    parser.add_argument("--type", type=str, choices=["chroma", "simple", "both"], 
                       default="both", help="Index type")
    parser.add_argument("--data_path", type=str, default="data/raw/pairs.jsonl", 
                       help="Raw data path")
    parser.add_argument("--output_dir", type=str, default="data/chroma/social_debate", 
                       help="Chroma index output directory")
    parser.add_argument("--simple_output", type=str, default="src/rag/data/rag/simple_index.json",
                       help="Simple index output path")
    parser.add_argument("--max_docs", type=int, default=None, help="Maximum documents")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    args = parser.parse_args()
    
    print("Building RAG index...")
    
    config = ConfigLoader.load("rag")
    
    chroma_config = config.get("chroma", {})
    indexing_config = config.get("indexing", {})
    embedding_config = chroma_config.get("embedding", {})
    
    data_path = args.data_path or indexing_config.get("data_source", "data/raw/pairs.jsonl")
    batch_size = args.batch_size or embedding_config.get("batch_size", 500)
    
    print(f"Index type: {args.type}")
    print(f"Data path: {data_path}")
    print(f"Batch size: {batch_size}")
    if args.max_docs:
        print(f"Max documents: {args.max_docs}")
    
    try:
        if args.type in ["chroma", "both"]:
            print("Building Chroma vector index...")
            
            output_dir = Path(args.output_dir)
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            
            stats = build_chroma_index(
                data_path=data_path,
                output_dir=str(output_dir),
                max_docs=args.max_docs,
                batch_size=batch_size
            )
            
            print("Chroma index complete")
            print(f"Total documents: {stats.get('total_docs', 0)}")
            print(f"Index location: {output_dir}")
        
        if args.type in ["simple", "both"]:
            print("Building simple JSON index...")
            
            simple_output = Path(args.simple_output)
            simple_output.parent.mkdir(parents=True, exist_ok=True)
            
            docs = build_simple_index(
                data_path=data_path,
                output_path=str(simple_output),
                max_docs=args.max_docs
            )
            
            print("Simple index complete")
            print(f"Documents: {len(docs)}")
            print(f"Index location: {simple_output}")
        
        print("All index building complete")
        
    except Exception as e:
        print(f"Build failed: {e}")
        raise

def build_demo_index():
    demo_docs = [
        {
            "id": "doc_001",
            "content": "AI regulation requires careful balance between innovation and safety. Proponents argue regulation prevents misuse while critics worry about stifling progress.",
            "metadata": {
                "type": "expert_opinion",
                "topic": "AI Regulation",
                "stance": "balanced",
                "quality_score": 0.85
            }
        },
        {
            "id": "doc_002",
            "content": "Tech industry opposes heavy AI regulation citing innovation concerns. Companies argue strict rules hurt global competitiveness.",
            "metadata": {
                "type": "industry_perspective",
                "topic": "AI Regulation",
                "stance": "oppose",
                "quality_score": 0.82
            }
        },
        {
            "id": "doc_003",
            "content": "MIT study suggests risk-based AI regulation framework. High-risk applications need stricter oversight while allowing innovation.",
            "metadata": {
                "type": "research",
                "topic": "AI Regulation",
                "stance": "nuanced",
                "quality_score": 0.90
            }
        }
    ]
    
    demo_path = Path("src/rag/data/rag/simple_index.json")
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(demo_path, 'w', encoding='utf-8') as f:
        json.dump({
            "documents": demo_docs,
            "metadata": {
                "version": "1.0",
                "created_at": "2024-01-01",
                "total_documents": len(demo_docs)
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Demo index saved: {demo_path}")

if __name__ == "__main__":
    if not Path("data/raw/pairs.jsonl").exists():
        print("Raw data not found, building demo index...")
        build_demo_index()
    else:
        main() 