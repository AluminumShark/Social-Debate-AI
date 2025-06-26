"""
RAG index builder
"""

import json, os, time, re
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import tiktoken
from collections import Counter, defaultdict

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    OPENAI_KEY = ""
if not OPENAI_KEY:
    raise SystemExit("Please set OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

PAIRS_PATH = Path("data/raw/pairs.jsonl")
DB_BASE_DIR = Path("data/index/enhanced")
THREADS_PATH = Path("data/raw/threads.jsonl")

INDEX_CONFIGS = {
    'high_quality': {
        'path': DB_BASE_DIR / 'high_quality',
        'collection': 'hq_pairs',
    },
    'by_topic': {
        'path': DB_BASE_DIR / 'by_topic',
        'collection': 'topic_sorted',
    },
    'comprehensive': {
        'path': DB_BASE_DIR / 'comprehensive',
        'collection': 'all_discussions',
    }
}

CHUNK_SIZE, CHUNK_OVERLAP = 1024, 256
EMB_MODEL = "text-embedding-3-small"
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

TOPIC_KEYWORDS = {
    'politics': ['government', 'politics', 'democracy', 'voting', 'election'],
    'economics': ['economy', 'money', 'income', 'tax', 'wealth', 'market'],
    'technology': ['technology', 'AI', 'automation', 'digital'],
    'social_justice': ['equality', 'rights', 'discrimination', 'justice'],
    'education': ['education', 'school', 'university', 'learning'],
    'healthcare': ['health', 'medical', 'healthcare', 'medicine'],
    'environment': ['environment', 'climate', 'pollution', 'sustainability'],
    'ethics': ['ethics', 'moral', 'right', 'wrong'],
    'work': ['work', 'job', 'employment', 'career'],
    'relationships': ['relationship', 'marriage', 'family', 'dating'],
    'law': ['law', 'legal', 'court', 'crime'],
    'religion': ['religion', 'religious', 'god', 'faith']
}

def classify_topics(text):
    text_lower = text.lower()
    topics = []
    
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            topics.append(topic)
    
    return topics if topics else ['general']

def extract_metadata(submission, comment=None):
    title = submission.get('title', '')
    selftext = submission.get('selftext', '')
    full_text = f"{title} {selftext}"
    
    topics_list = classify_topics(full_text)
    metadata = {
        'submission_id': submission.get('id', ''),
        'score': submission.get('score', 0),
        'num_comments': submission.get('num_comments', 0),
        'created_utc': submission.get('created_utc', 0),
        'topics': ','.join(topics_list),
        'primary_topic': topics_list[0] if topics_list else 'general',
        'title': title[:100],
    }
    
    sentences = len(re.findall(r'[.!?]+', full_text))
    words = len(full_text.split())
    complexity = 'simple' if words < 100 else 'intermediate' if words < 300 else 'complex'
    metadata['complexity'] = complexity
    
    if comment:
        metadata.update({
            'type': 'delta_comment',
            'comment_id': comment.get('id', ''),
            'comment_score': comment.get('score', 0),
            'persuasion_success': True,
            'argument_strength': min(comment.get('score', 0) / 10.0, 1.0)
        })
    else:
        metadata['type'] = 'submission'
    
    return metadata

class SimpleEmbeddings(OpenAIEmbeddings):
    def __init__(self, *args, batch_size=500, **kwargs):
        super().__init__(*args, **kwargs)
        self._total_tokens = 0
        self._batch_size = batch_size
    
    @property
    def total_tokens(self): 
        return self._total_tokens
    
    def embed_documents(self, texts):
        print(f"Embedding {len(texts):,} documents...")
        total_tokens = sum(len(tokenizer.encode(text)) for text in texts)
        print(f"Estimated tokens: {total_tokens:,}")
        
        batch_size = self._batch_size
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing"):
            batch = texts[i:i + batch_size]
            batch_embeddings = super().embed_documents(batch)
            embeddings.extend(batch_embeddings)
            
            batch_tokens = sum(len(tokenizer.encode(text)) for text in batch)
            self._total_tokens += batch_tokens
            time.sleep(0.1)
        
        print("Embedding complete")
        return embeddings

def build_high_quality_index():
    print("Building high-quality index...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    
    docs = []
    with PAIRS_PATH.open(encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading pairs"):
            pair = json.loads(line)
            submission = pair["submission"]
            
            sub_meta = extract_metadata(submission)
            body = submission.get("selftext") or submission.get("title", "")
            if body:
                for chunk in splitter.split_text(body):
                    docs.append(Document(page_content=chunk, metadata=sub_meta))
            
            delta_comment = pair.get("delta_comment", {})
            if delta_comment and delta_comment.get("body"):
                comment_meta = extract_metadata(submission, delta_comment)
                for chunk in splitter.split_text(delta_comment["body"]):
                    docs.append(Document(page_content=chunk, metadata=comment_meta))
    
    print(f"Collected {len(docs):,} document chunks")
    
    config = INDEX_CONFIGS['high_quality']
    config['path'].mkdir(parents=True, exist_ok=True)
    
    embeddings = SimpleEmbeddings(model=EMB_MODEL)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(config['path']),
        collection_name=config['collection']
    )
    vectorstore.persist()
    
    print(f"Index saved: {config['path']}")
    return len(docs)

def build_comprehensive_index():
    print("Building comprehensive index...")
    
    if not THREADS_PATH.exists():
        print("Threads data not found, skipping comprehensive index")
        return 0
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    docs = []
    with THREADS_PATH.open(encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Loading threads")):
            if i >= 5000:  # Limit for demo
                break
                
            try:
                thread = json.loads(line)
                if thread.get('selftext') or thread.get('title'):
                    meta = extract_metadata(thread)
                    text = thread.get('selftext') or thread.get('title', '')
                    
                    for chunk in splitter.split_text(text):
                        docs.append(Document(page_content=chunk, metadata=meta))
            except:
                continue
    
    if not docs:
        print("No valid documents found")
        return 0
    
    config = INDEX_CONFIGS['comprehensive']
    config['path'].mkdir(parents=True, exist_ok=True)
    
    embeddings = SimpleEmbeddings(model=EMB_MODEL)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(config['path']),
        collection_name=config['collection']
    )
    vectorstore.persist()
    
    print(f"Comprehensive index saved: {config['path']}")
    return len(docs)

def build_all_indexes():
    print("Building all indexes...")
    
    results = {}
    
    if PAIRS_PATH.exists():
        results['high_quality'] = build_high_quality_index()
    else:
        print("Pairs data not found")
        results['high_quality'] = 0
    
    results['comprehensive'] = build_comprehensive_index()
    
    print("All indexes complete:")
    for name, count in results.items():
        print(f"  {name}: {count} documents")
    
    return results

def build_chroma_index(data_path: str, output_dir: str, max_docs: int = None, batch_size: int = 500):
    print(f"Building Chroma index from {data_path}...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=256
    )
    
    docs = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Loading data")):
            if max_docs and i >= max_docs:
                break
                
            try:
                pair = json.loads(line)
                submission = pair['submission']
                
                meta = extract_metadata(submission)
                text = submission.get('selftext') or submission.get('title', '')
                
                if text:
                    for chunk in splitter.split_text(text):
                        docs.append(Document(page_content=chunk, metadata=meta))
                
                if 'delta_comment' in pair and pair['delta_comment'].get('body'):
                    comment_meta = extract_metadata(submission, pair['delta_comment'])
                    for chunk in splitter.split_text(pair['delta_comment']['body']):
                        docs.append(Document(page_content=chunk, metadata=comment_meta))
                        
            except Exception as e:
                continue
    
    if not docs:
        print("No documents to index")
        return {'total_docs': 0}
    
    print(f"Creating vector store with {len(docs)} documents...")
    
    embeddings = SimpleEmbeddings(model=EMB_MODEL, batch_size=batch_size)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=output_dir,
        collection_name='social_debate'
    )
    vectorstore.persist()
    
    return {'total_docs': len(docs)}

def build_simple_index(data_path: str, output_path: str, max_docs: int = None):
    print(f"Building simple index from {data_path}...")
    
    docs = []
    
    if not Path(data_path).exists():
        print("Data file not found, creating demo index...")
        demo_docs = [
            {
                "id": "demo_001",
                "content": "AI regulation discussion with various perspectives on government oversight and innovation balance.",
                "metadata": {
                    "type": "discussion",
                    "topic": "AI Regulation",
                    "quality_score": 0.85
                }
            },
            {
                "id": "demo_002", 
                "content": "Economic policy debate focusing on wealth distribution and market mechanisms.",
                "metadata": {
                    "type": "debate",
                    "topic": "Economics", 
                    "quality_score": 0.82
                }
            }
        ]
        
        index_data = {
            "documents": demo_docs,
            "metadata": {
                "version": "1.0",
                "total_documents": len(demo_docs)
            }
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
        
        print(f"Demo index saved: {output_path}")
        return demo_docs
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Processing")):
            if max_docs and i >= max_docs:
                break
                
            try:
                pair = json.loads(line)
                submission = pair['submission']
                
                doc = {
                    "id": f"doc_{i:06d}",
                    "content": submission.get('selftext') or submission.get('title', ''),
                    "metadata": extract_metadata(submission)
                }
                docs.append(doc)
                
                if 'delta_comment' in pair and pair['delta_comment'].get('body'):
                    comment_doc = {
                        "id": f"comment_{i:06d}",
                        "content": pair['delta_comment']['body'],
                        "metadata": extract_metadata(submission, pair['delta_comment'])
                    }
                    docs.append(comment_doc)
                    
            except:
                continue
    
    index_data = {
        "documents": docs,
        "metadata": {
            "version": "1.0",
            "total_documents": len(docs)
        }
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)
    
    print(f"Simple index saved: {output_path}")
    return docs

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["all", "hq", "comprehensive"], default="hq")
    args = parser.parse_args()
    
    if args.type == "all":
        build_all_indexes()
    elif args.type == "hq":
        build_high_quality_index()
    elif args.type == "comprehensive":
        build_comprehensive_index() 