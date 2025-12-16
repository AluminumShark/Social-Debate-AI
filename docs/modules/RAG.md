# RAG (Retrieval-Augmented Generation) 模組詳解

## 目錄

1. [什麼是 RAG？](#什麼是-rag)
2. [為什麼需要 RAG？](#為什麼需要-rag)
3. [RAG 的工作原理](#rag-的工作原理)
4. [訓練資料詳解](#訓練資料詳解)
5. [檢索系統架構](#檢索系統架構)
6. [向量資料庫](#向量資料庫)
7. [混合檢索策略](#混合檢索策略)
8. [重排序機制](#重排序機制)
9. [實際應用](#實際應用)
10. [程式碼範例](#程式碼範例)
11. [常見問題](#常見問題)

## 什麼是 RAG？

### 基本概念

想像你在寫論文：
1. 你有一個觀點要論證
2. 你去圖書館找相關資料（**檢索**）
3. 你引用這些資料來支持論點（**增強**）
4. 你寫出完整的論述（**生成**）

**RAG（檢索增強生成）**就是讓 AI 也能這樣做：先找到相關資料，再基於資料生成回應。

### RAG vs 純生成模型

```python
# 純生成模型
問題: "碳稅的經濟影響是什麼？"
回答: [僅基於模型訓練時的知識]

# RAG 模型
問題: "碳稅的經濟影響是什麼？"
步驟1: 檢索相關文獻、報告、研究
步驟2: 基於檢索結果生成回答
回答: [引用具體數據和研究的準確回答]
```

### RAG 的組成部分

```
查詢 → [檢索器] → 相關文檔 → [生成器] → 增強的回答
         ↑                      ↑
    向量資料庫              語言模型
```

## 為什麼需要 RAG？

### 傳統語言模型的限制

1. **知識過時**：訓練數據有截止日期
2. **幻覺問題**：可能編造不存在的事實
3. **缺乏引用**：無法提供信息來源
4. **領域限制**：專業知識可能不足

### RAG 的優勢

1. **實時更新**：可以檢索最新資料
2. **可驗證性**：提供具體引用來源
3. **領域適應**：輕鬆添加專業資料庫
4. **減少幻覺**：基於真實文檔生成

### 在辯論系統中的價值

```python
# 沒有 RAG
辯論: "核能是否安全？"
回應: "我認為核能相對安全..." [缺乏具體數據]

# 有 RAG
辯論: "核能是否安全？"
檢索: [IAEA 2023報告, 核能安全統計, 事故分析]
回應: "根據 IAEA 2023 年報告，現代核電站的事故率為..." [有據可查]
```

## RAG 的工作原理

### 1. 文檔預處理

```python
def preprocess_documents(documents):
    chunks = []
    for doc in documents:
        # 1. 分割文檔
        doc_chunks = split_into_chunks(doc, chunk_size=512)
        
        # 2. 添加元數據
        for chunk in doc_chunks:
            chunk.metadata = {
                'source': doc.source,
                'date': doc.date,
                'author': doc.author,
                'topic': doc.topic
            }
        
        # 3. 清理文本
        cleaned_chunks = [clean_text(chunk) for chunk in doc_chunks]
        chunks.extend(cleaned_chunks)
    
    return chunks
```

### 2. 向量化（Embedding）

將文本轉換為向量，使電腦能理解語義相似度：

```python
def create_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        # 使用 BERT 或其他模型編碼
        embedding = embedding_model.encode(chunk.text)
        embeddings.append({
            'id': chunk.id,
            'embedding': embedding,  # 768維向量
            'text': chunk.text,
            'metadata': chunk.metadata
        })
    return embeddings
```

### 3. 檢索過程

```python
def retrieve(query, k=5):
    # 1. 查詢向量化
    query_embedding = embedding_model.encode(query)
    
    # 2. 相似度搜索
    similarities = []
    for doc_embedding in document_embeddings:
        similarity = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((doc_embedding, similarity))
    
    # 3. 排序並返回 top-k
    top_k = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
    return [doc for doc, _ in top_k]
```

### 4. 生成增強回應

```python
def generate_augmented_response(query, retrieved_docs):
    # 構建上下文
    context = "\n".join([doc.text for doc in retrieved_docs])
    
    # 構建提示
    prompt = f"""
    基於以下資料回答問題：
    
    資料：
    {context}
    
    問題：{query}
    
    請引用具體資料來源回答。
    """
    
    # 生成回應
    response = language_model.generate(prompt)
    return response
```

## 訓練資料詳解

### 資料來源

我們的 RAG 系統使用多種資料來源：

1. **學術論文**
   - ArXiv 論文
   - Google Scholar
   - 期刊文章

2. **新聞資料**
   - 主流媒體報導
   - 專業媒體分析
   - 時事評論

3. **專業報告**
   - 政府報告
   - 智庫研究
   - 行業分析

4. **百科資料**
   - Wikipedia
   - 專業百科
   - 教科書

### 資料格式

```json
{
  "id": "doc_001",
  "title": "碳稅對經濟發展的影響研究",
  "content": "根據 2023 年的研究顯示...",
  "source": "經濟學期刊",
  "date": "2023-06-15",
  "author": "張博士",
  "topics": ["碳稅", "經濟", "環保"],
  "credibility_score": 0.95,
  "citations": 45
}
```

### 資料處理流程

```python
# 1. 資料清理
def clean_document(doc):
    # 移除 HTML 標籤
    doc = remove_html_tags(doc)
    # 修正編碼問題
    doc = fix_encoding(doc)
    # 標準化格式
    doc = normalize_whitespace(doc)
    return doc

# 2. 分塊策略
def chunk_document(doc, method='sliding_window'):
    if method == 'sliding_window':
        # 滑動窗口，有重疊
        chunks = []
        window_size = 512
        stride = 256
        for i in range(0, len(doc), stride):
            chunk = doc[i:i+window_size]
            chunks.append(chunk)
    
    elif method == 'semantic':
        # 基於語義的分塊
        chunks = split_by_sentences(doc)
        chunks = group_semantic_chunks(chunks)
    
    return chunks

# 3. 品質過濾
def filter_quality(chunks):
    filtered = []
    for chunk in chunks:
        # 檢查長度
        if len(chunk.split()) < 50:
            continue
        # 檢查資訊密度
        if information_density(chunk) < 0.3:
            continue
        # 檢查語言品質
        if language_quality(chunk) < 0.7:
            continue
        filtered.append(chunk)
    return filtered
```

## 檢索系統架構

### 1. 雙編碼器架構

```python
class DualEncoder:
    def __init__(self):
        # 查詢編碼器
        self.query_encoder = AutoModel.from_pretrained('bert-base')
        # 文檔編碼器（可以相同或不同）
        self.doc_encoder = AutoModel.from_pretrained('bert-base')
    
    def encode_query(self, query):
        # 針對查詢優化的編碼
        query_tokens = tokenize(query, max_length=128)
        query_embedding = self.query_encoder(query_tokens)
        return query_embedding
    
    def encode_document(self, document):
        # 針對文檔優化的編碼
        doc_tokens = tokenize(document, max_length=512)
        doc_embedding = self.doc_encoder(doc_tokens)
        return doc_embedding
```

### 2. 相似度計算

```python
def calculate_similarity(query_emb, doc_emb, method='cosine'):
    if method == 'cosine':
        # 餘弦相似度
        similarity = cosine_similarity(query_emb, doc_emb)
    
    elif method == 'dot_product':
        # 點積
        similarity = np.dot(query_emb, doc_emb)
    
    elif method == 'euclidean':
        # 歐氏距離（轉為相似度）
        distance = np.linalg.norm(query_emb - doc_emb)
        similarity = 1 / (1 + distance)
    
    return similarity
```

### 3. 索引結構

```python
class VectorIndex:
    def __init__(self, dimension=768):
        # 使用 FAISS 建立索引
        self.index = faiss.IndexFlatIP(dimension)  # 內積索引
        self.documents = []
    
    def add_documents(self, embeddings, documents):
        # 正規化向量（用於餘弦相似度）
        embeddings = normalize_vectors(embeddings)
        # 添加到索引
        self.index.add(embeddings)
        self.documents.extend(documents)
    
    def search(self, query_embedding, k=5):
        # 正規化查詢向量
        query_embedding = normalize_vector(query_embedding)
        # 搜索
        distances, indices = self.index.search(query_embedding, k)
        # 返回文檔
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append({
                'document': self.documents[idx],
                'score': dist
            })
        return results
```

## 向量資料庫

### 什麼是向量資料庫？

向量資料庫是專門用於存儲和檢索高維向量的資料庫系統。

```
傳統資料庫：
SELECT * FROM documents WHERE content LIKE '%碳稅%'

向量資料庫：
SEARCH vectors WHERE similarity(vector, query_vector) > 0.8
```

### 常用向量資料庫

| 資料庫 | 特點 | 適用場景 |
|--------|------|----------|
| FAISS | 快速、內存型 | 中小規模、高性能需求 |
| Pinecone | 雲端、托管式 | 大規模、無需維護 |
| Weaviate | 全功能、GraphQL | 複雜查詢、混合搜索 |
| Milvus | 分散式、可擴展 | 超大規模、高並發 |

### 實現範例

```python
class VectorDatabase:
    def __init__(self, db_type='faiss'):
        if db_type == 'faiss':
            self.engine = FAISSEngine()
        elif db_type == 'pinecone':
            self.engine = PineconeEngine()
    
    def insert(self, vectors, metadata):
        """插入向量和元數據"""
        ids = generate_ids(len(vectors))
        self.engine.upsert(
            ids=ids,
            vectors=vectors,
            metadata=metadata
        )
        return ids
    
    def query(self, vector, filter=None, top_k=5):
        """查詢相似向量"""
        results = self.engine.query(
            vector=vector,
            filter=filter,  # 元數據過濾
            top_k=top_k
        )
        return results
    
    def update(self, id, vector=None, metadata=None):
        """更新向量或元數據"""
        if vector is not None:
            self.engine.update_vector(id, vector)
        if metadata is not None:
            self.engine.update_metadata(id, metadata)
    
    def delete(self, ids):
        """刪除向量"""
        self.engine.delete(ids)
```

## 混合檢索策略

### 為什麼需要混合檢索？

單一檢索方法各有優缺點：
- **向量檢索**：理解語義，但可能錯過關鍵詞
- **關鍵詞檢索**：精確匹配，但缺乏語義理解
- **混合檢索**：結合兩者優勢

### 混合檢索實現

```python
class HybridRetriever:
    def __init__(self):
        self.vector_retriever = VectorRetriever()
        self.keyword_retriever = BM25Retriever()
        self.weight_vector = 0.7  # 向量檢索權重
        self.weight_keyword = 0.3  # 關鍵詞檢索權重
    
    def retrieve(self, query, k=10):
        # 1. 向量檢索
        vector_results = self.vector_retriever.search(query, k=k*2)
        
        # 2. 關鍵詞檢索
        keyword_results = self.keyword_retriever.search(query, k=k*2)
        
        # 3. 分數融合
        combined_scores = {}
        
        # 處理向量檢索結果
        for doc, score in vector_results:
            combined_scores[doc.id] = {
                'doc': doc,
                'score': score * self.weight_vector
            }
        
        # 處理關鍵詞檢索結果
        for doc, score in keyword_results:
            if doc.id in combined_scores:
                combined_scores[doc.id]['score'] += score * self.weight_keyword
            else:
                combined_scores[doc.id] = {
                    'doc': doc,
                    'score': score * self.weight_keyword
                }
        
        # 4. 排序返回
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:k]
        
        return [(r['doc'], r['score']) for r in sorted_results]
```

### 進階混合策略

```python
class AdvancedHybridRetriever:
    def __init__(self):
        self.retrievers = {
            'vector': VectorRetriever(),
            'bm25': BM25Retriever(),
            'tfidf': TFIDFRetriever(),
            'semantic': SemanticRetriever()
        }
        self.ensemble_model = load_ensemble_model()
    
    def retrieve(self, query, context=None, k=10):
        # 1. 多路檢索
        all_results = {}
        for name, retriever in self.retrievers.items():
            results = retriever.search(query, k=k*2)
            all_results[name] = results
        
        # 2. 特徵提取
        candidates = self.merge_candidates(all_results)
        features = []
        
        for doc in candidates:
            feature = {
                'vector_score': self.get_score(doc, all_results['vector']),
                'bm25_score': self.get_score(doc, all_results['bm25']),
                'tfidf_score': self.get_score(doc, all_results['tfidf']),
                'semantic_score': self.get_score(doc, all_results['semantic']),
                'doc_length': len(doc.text.split()),
                'query_coverage': self.calculate_coverage(query, doc),
                'freshness': self.calculate_freshness(doc),
                'credibility': doc.metadata.get('credibility', 0.5)
            }
            features.append(feature)
        
        # 3. 學習權重（使用訓練好的模型）
        scores = self.ensemble_model.predict(features)
        
        # 4. 排序返回
        ranked_docs = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        return ranked_docs
```

## 重排序機制

### 為什麼需要重排序？

初始檢索可能不夠精確，重排序可以：
1. 考慮更多上下文信息
2. 使用更複雜的模型
3. 個性化排序結果

### 交叉編碼器重排序

```python
class CrossEncoderReranker:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'cross-encoder/ms-marco-MiniLM-L-12-v2'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            'cross-encoder/ms-marco-MiniLM-L-12-v2'
        )
    
    def rerank(self, query, documents, top_k=5):
        # 1. 準備輸入對
        pairs = []
        for doc in documents:
            pairs.append([query, doc.text])
        
        # 2. 批次編碼
        encoded = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # 3. 預測相關性分數
        with torch.no_grad():
            outputs = self.model(**encoded)
            scores = outputs.logits.squeeze()
            scores = torch.sigmoid(scores).numpy()
        
        # 4. 重新排序
        reranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return reranked
```

### 多階段重排序

```python
class MultiStageReranker:
    def __init__(self):
        self.light_reranker = LightweightReranker()  # 快速
        self.heavy_reranker = CrossEncoderReranker()  # 精確
        self.personalization = PersonalizationReranker()  # 個性化
    
    def rerank(self, query, documents, user_profile=None):
        # 階段1：輕量級重排序（處理所有文檔）
        stage1_results = self.light_reranker.rerank(
            query, documents, top_k=20
        )
        
        # 階段2：重量級重排序（處理 top 20）
        stage2_results = self.heavy_reranker.rerank(
            query, [doc for doc, _ in stage1_results], top_k=10
        )
        
        # 階段3：個性化（如果有用戶資料）
        if user_profile:
            final_results = self.personalization.rerank(
                query, 
                [doc for doc, _ in stage2_results],
                user_profile,
                top_k=5
            )
        else:
            final_results = stage2_results[:5]
        
        return final_results
```

## 實際應用

### 1. 辯論證據檢索

```python
class DebateEvidenceRetriever:
    def __init__(self):
        self.rag_system = RAGSystem()
        self.fact_checker = FactChecker()
    
    def find_supporting_evidence(self, claim, stance):
        # 1. 構建查詢
        query = f"證據支持：{claim}"
        if stance == 'support':
            query += " 正面論據"
        else:
            query += " 反面論據"
        
        # 2. 檢索相關文檔
        documents = self.rag_system.retrieve(query, k=10)
        
        # 3. 事實檢查
        verified_docs = []
        for doc in documents:
            if self.fact_checker.verify(doc):
                verified_docs.append(doc)
        
        # 4. 提取關鍵論據
        arguments = []
        for doc in verified_docs:
            arg = {
                'claim': extract_main_claim(doc),
                'evidence': extract_evidence(doc),
                'source': doc.metadata['source'],
                'credibility': doc.metadata['credibility']
            }
            arguments.append(arg)
        
        return arguments
```

### 2. 實時知識更新

```python
class RealTimeKnowledgeUpdater:
    def __init__(self):
        self.vector_db = VectorDatabase()
        self.news_crawler = NewsCrawler()
        self.update_interval = 3600  # 每小時更新
    
    def update_knowledge_base(self):
        # 1. 爬取最新資訊
        new_articles = self.news_crawler.get_latest(
            topics=['technology', 'politics', 'economy'],
            hours=1
        )
        
        # 2. 處理新文章
        processed_docs = []
        for article in new_articles:
            # 品質檢查
            if not self.quality_check(article):
                continue
            
            # 分塊
            chunks = self.chunk_article(article)
            
            # 向量化
            embeddings = self.encode_chunks(chunks)
            
            processed_docs.extend(zip(chunks, embeddings))
        
        # 3. 更新向量資料庫
        for chunk, embedding in processed_docs:
            self.vector_db.insert(
                vector=embedding,
                metadata={
                    'text': chunk.text,
                    'source': chunk.source,
                    'timestamp': chunk.timestamp,
                    'topic': chunk.topic
                }
            )
        
        print(f"Updated {len(processed_docs)} documents")
```

### 3. 個性化檢索

```python
class PersonalizedRAG:
    def __init__(self):
        self.rag_system = RAGSystem()
        self.user_profiler = UserProfiler()
    
    def retrieve_personalized(self, query, user_id):
        # 1. 獲取用戶檔案
        user_profile = self.user_profiler.get_profile(user_id)
        
        # 2. 調整查詢
        enhanced_query = self.enhance_query(query, user_profile)
        
        # 3. 個性化過濾
        filters = {
            'difficulty': user_profile.knowledge_level,
            'language': user_profile.preferred_language,
            'topics': user_profile.interests
        }
        
        # 4. 檢索
        results = self.rag_system.retrieve(
            query=enhanced_query,
            filters=filters,
            k=10
        )
        
        # 5. 個性化重排序
        reranked = self.personalize_ranking(
            results, 
            user_profile
        )
        
        return reranked
    
    def personalize_ranking(self, results, profile):
        scored_results = []
        
        for doc in results:
            score = doc.base_score
            
            # 興趣匹配加分
            topic_match = self.calculate_topic_match(
                doc.topics, 
                profile.interests
            )
            score += topic_match * 0.2
            
            # 難度匹配加分
            if abs(doc.difficulty - profile.knowledge_level) < 0.2:
                score += 0.1
            
            # 來源偏好加分
            if doc.source in profile.trusted_sources:
                score += 0.15
            
            scored_results.append((doc, score))
        
        return sorted(scored_results, key=lambda x: x[1], reverse=True)
```

## 程式碼範例

### 完整的 RAG 系統實現

```python
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
import faiss
from typing import List, Dict, Tuple

class CompleteRAGSystem:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        # 初始化編碼器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # 初始化向量索引
        self.dimension = 384  # MiniLM 輸出維度
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # 文檔存儲
        self.documents = []
        self.metadata = []
        
        # 重排序器
        self.reranker = CrossEncoderReranker()
    
    def add_documents(self, documents: List[Dict]):
        """添加文檔到系統"""
        embeddings = []
        
        for doc in documents:
            # 分塊
            chunks = self.chunk_document(doc['content'])
            
            for chunk in chunks:
                # 編碼
                embedding = self.encode_text(chunk)
                embeddings.append(embedding)
                
                # 存儲
                self.documents.append(chunk)
                self.metadata.append({
                    'source': doc.get('source', 'unknown'),
                    'date': doc.get('date', None),
                    'title': doc.get('title', ''),
                    'chunk_id': len(self.documents) - 1
                })
        
        # 添加到索引
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)  # 正規化用於餘弦相似度
        self.index.add(embeddings)
    
    def chunk_document(self, text: str, chunk_size: int = 512, overlap: int = 128):
        """文檔分塊"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.split()) > 50:  # 最小塊大小
                chunks.append(chunk)
        
        return chunks
    
    def encode_text(self, text: str) -> np.ndarray:
        """文本編碼"""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # 使用 CLS token 或平均池化
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.numpy()[0]
    
    def retrieve(self, query: str, k: int = 5, use_reranking: bool = True):
        """檢索相關文檔"""
        # 1. 編碼查詢
        query_embedding = self.encode_text(query)
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # 2. 向量檢索
        distances, indices = self.index.search(query_embedding, k * 2 if use_reranking else k)
        
        # 3. 收集結果
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append({
                    'text': self.documents[idx],
                    'score': float(dist),
                    'metadata': self.metadata[idx]
                })
        
        # 4. 重排序（可選）
        if use_reranking and len(results) > 0:
            texts = [r['text'] for r in results]
            reranked = self.reranker.rerank(query, texts, top_k=k)
            
            # 更新分數
            reranked_results = []
            for text, score in reranked:
                for r in results:
                    if r['text'] == text:
                        r['score'] = score
                        reranked_results.append(r)
                        break
            
            results = reranked_results
        
        return results[:k]
    
    def generate_answer(self, query: str, retrieved_docs: List[Dict], 
                       language_model=None) -> str:
        """基於檢索結果生成答案"""
        # 構建上下文
        context = "\n\n".join([
            f"來源 {i+1} [{doc['metadata']['source']}]:\n{doc['text']}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        # 構建提示
        prompt = f"""基於以下資料回答問題。請引用具體來源。

資料：
{context}

問題：{query}

回答："""
        
        # 如果有語言模型，生成回答
        if language_model:
            answer = language_model.generate(prompt)
        else:
            # 簡單的摘要方法
            answer = self.simple_summarize(query, retrieved_docs)
        
        return answer
    
    def simple_summarize(self, query: str, docs: List[Dict]) -> str:
        """簡單的摘要生成"""
        relevant_sentences = []
        
        for doc in docs:
            sentences = doc['text'].split('.')
            for sent in sentences:
                if any(word in sent.lower() for word in query.lower().split()):
                    relevant_sentences.append(sent.strip())
        
        # 返回最相關的句子
        summary = '. '.join(relevant_sentences[:3]) + '.'
        sources = list(set([doc['metadata']['source'] for doc in docs]))
        
        return f"{summary}\n\n資料來源：{', '.join(sources)}"

# 使用範例
if __name__ == "__main__":
    # 初始化系統
    rag = CompleteRAGSystem()
    
    # 添加文檔
    documents = [
        {
            'content': '人工智慧正在改變醫療診斷...',
            'source': 'AI醫療期刊',
            'date': '2024-01-15',
            'title': 'AI在醫療領域的應用'
        },
        {
            'content': '機器學習算法可以預測疾病...',
            'source': '科技日報',
            'date': '2024-01-20',
            'title': '機器學習的醫療應用'
        }
    ]
    
    rag.add_documents(documents)
    
    # 檢索
    query = "AI如何幫助醫療診斷？"
    results = rag.retrieve(query, k=3)
    
    # 生成答案
    answer = rag.generate_answer(query, results)
    print(f"問題：{query}")
    print(f"答案：{answer}")
    
    # 顯示檢索結果
    print("\n檢索到的文檔：")
    for i, result in enumerate(results):
        print(f"{i+1}. 來源：{result['metadata']['source']}")
        print(f"   分數：{result['score']:.3f}")
        print(f"   內容：{result['text'][:100]}...")
```

### 訓練檢索器

```python
class RetrieverTrainer:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
    def prepare_training_data(self, qa_pairs):
        """準備訓練數據"""
        training_examples = []
        
        for qa in qa_pairs:
            query = qa['question']
            positive = qa['positive_passage']
            negatives = qa['negative_passages']
            
            # 創建訓練樣本
            example = InputExample(
                texts=[query, positive, negatives[0]],
                label=1.0  # 正樣本標籤
            )
            training_examples.append(example)
        
        return training_examples
    
    def train(self, training_data, epochs=10, batch_size=16):
        """訓練檢索器"""
        # 準備數據載入器
        train_dataloader = DataLoader(
            training_data, 
            shuffle=True, 
            batch_size=batch_size
        )
        
        # 定義損失函數
        train_loss = losses.TripletLoss(
            model=self.model,
            distance_metric=TripletDistanceMetric.EUCLIDEAN,
            triplet_margin=0.5
        )
        
        # 訓練
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=int(len(train_dataloader) * 0.1),
            show_progress_bar=True
        )
        
        # 保存模型
        self.model.save('fine_tuned_retriever')
        
    def evaluate(self, test_queries, test_corpus, relevant_docs):
        """評估檢索器"""
        # 編碼所有文檔
        corpus_embeddings = self.model.encode(test_corpus)
        
        # 評估指標
        metrics = {
            'mrr': [],  # Mean Reciprocal Rank
            'recall@5': [],
            'recall@10': []
        }
        
        for query, relevant in zip(test_queries, relevant_docs):
            # 編碼查詢
            query_embedding = self.model.encode(query)
            
            # 計算相似度
            similarities = cosine_similarity(
                [query_embedding], 
                corpus_embeddings
            )[0]
            
            # 排序
            top_indices = np.argsort(similarities)[::-1]
            
            # 計算指標
            for i, idx in enumerate(top_indices):
                if idx in relevant:
                    metrics['mrr'].append(1 / (i + 1))
                    break
            
            metrics['recall@5'].append(
                len(set(top_indices[:5]) & set(relevant)) / len(relevant)
            )
            metrics['recall@10'].append(
                len(set(top_indices[:10]) & set(relevant)) / len(relevant)
            )
        
        # 平均指標
        for key in metrics:
            metrics[key] = np.mean(metrics[key])
        
        return metrics
```

## 常見問題

### Q1: 如何選擇適合的編碼模型？

**A**: 考慮以下因素：
- **多語言需求**：使用 multilingual-e5-base
- **速度優先**：使用 all-MiniLM-L6-v2
- **精度優先**：使用 all-mpnet-base-v2
- **領域特定**：微調專門的模型

### Q2: 文檔分塊的最佳策略？

**A**: 
1. **固定大小**：簡單快速，適合一般文本
2. **句子邊界**：保持語義完整性
3. **段落分割**：適合結構化文檔
4. **滑動窗口**：避免邊界信息丟失

建議：512 tokens 塊大小，128 tokens 重疊

### Q3: 如何處理多模態資料？

**A**: 
```python
class MultiModalRAG:
    def __init__(self):
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()
        self.audio_encoder = AudioEncoder()
    
    def encode_multimodal(self, item):
        embeddings = []
        
        if item.text:
            embeddings.append(self.text_encoder(item.text))
        if item.image:
            embeddings.append(self.image_encoder(item.image))
        if item.audio:
            embeddings.append(self.audio_encoder(item.audio))
        
        # 融合策略
        return np.concatenate(embeddings)
```

### Q4: 如何優化檢索速度？

**A**: 
1. **使用 GPU**：批次編碼
2. **索引優化**：使用 IVF、HNSW 等近似算法
3. **預計算**：離線編碼文檔
4. **緩存**：儲存常見查詢結果
5. **分片**：大規模資料分散處理

### Q5: 如何評估 RAG 系統？

**A**: 評估指標：
- **檢索質量**：Recall@K、MRR、NDCG
- **生成質量**：BLEU、ROUGE、BERTScore
- **端到端**：人工評估、任務完成率
- **效率**：延遲、吞吐量

### Q6: RAG 的限制和挑戰？

**A**: 
1. **檢索噪音**：可能檢索到無關文檔
2. **上下文長度**：模型輸入長度限制
3. **實時性**：索引更新延遲
4. **一致性**：檢索結果可能相互矛盾
5. **成本**：存儲和計算成本較高

## 總結

RAG 在辯論系統中的價值：
1. **事實支撐**：提供可驗證的證據
2. **知識更新**：獲取最新信息
3. **深度論證**：引用專業資料
4. **可信度**：增強論點說服力

通過 RAG，我們讓 AI 不僅能「思考」，還能「查證」，真正實現有理有據的智能辯論。 