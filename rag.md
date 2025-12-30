# Chapter 7: Retrieval Augmented Generation

## 7.1 The Knowledge Problem

Imagine asking a brilliant friend about yesterday's news. No matter how smart they are, if they've been offline for a month, they can't help you. Large language models face a similar challenge—they're frozen in time, trained on data from months or years ago. They also can't access your private documents, your company's database, or specialized knowledge bases.

This is where **Retrieval Augmented Generation (RAG)** comes in. Instead of relying solely on a model's parametric memory (knowledge baked into its weights), we augment it with *retrieval*: fetching relevant information from external sources and providing it as context.

Think of RAG as giving your AI a library card and teaching it to look things up.

## 7.2 The Basic RAG Pipeline

A RAG system has three core components:

1. **Indexing**: Convert documents into searchable embeddings
2. **Retrieval**: Find relevant documents for a query
3. **Generation**: Use retrieved context to answer the query

Let's build this step by step.

### 7.2.1 Setting Up Our Environment

```python
import ollama
import numpy as np
from typing import List, Dict

def get_embedding(text: str, model: str = "nomic-embed-text") -> List[float]:
    """Get embedding vector for text."""
    response = ollama.embeddings(model=model, prompt=text)
    return response['embedding']

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

### 7.2.2 Building a Simple Document Store

```python
class SimpleDocumentStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []
    
    def add_document(self, text: str, metadata: Dict = None):
        """Add a document and compute its embedding."""
        self.documents.append({"text": text, "metadata": metadata or {}})
        embedding = get_embedding(text)
        self.embeddings.append(embedding)
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Find most relevant documents."""
        query_emb = get_embedding(query)
        scores = [cosine_similarity(query_emb, doc_emb) 
                  for doc_emb in self.embeddings]
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]
```

## 7.3 Your First RAG Application

Let's build a simple RAG system for course syllabi—something every student wishes existed!

```python
# Initialize document store
store = SimpleDocumentStore()

# Add some course information
store.add_document(
    "CS 101 office hours are Monday and Wednesday 2-4pm in Room 305.",
    {"course": "CS101", "type": "logistics"}
)
store.add_document(
    "The final project for CS 101 is due December 15th and worth 40% of your grade.",
    {"course": "CS101", "type": "assessment"}
)
store.add_document(
    "CS 101 covers Python basics, data structures, and algorithm fundamentals.",
    {"course": "CS101", "type": "content"}
)

def rag_query(question: str, store: SimpleDocumentStore) -> str:
    """Answer question using RAG."""
    # Retrieve relevant documents
    relevant_docs = store.search(question, top_k=2)
    
    # Build context from retrieved documents
    context = "\n\n".join([doc["text"] for doc in relevant_docs])
    
    # Create prompt with context
    prompt = f"""Based on the following information:

{context}

Question: {question}

Answer the question using only the information provided above."""

    # Generate response
    response = ollama.generate(model="qwen2.5:latest", prompt=prompt)
    return response['response']

# Test it
print(rag_query("When is the final project due?", store))
```

**Output**: "The final project is due December 15th."

Notice how the model only uses retrieved information—it doesn't hallucinate dates or make up policies.

## 7.4 Chunking: Breaking Down Documents

Real documents are too long to embed as single units. We need to break them into meaningful *chunks*. This is trickier than it sounds!

### 7.4.1 Naive Chunking

```python
def chunk_by_sentences(text: str, sentences_per_chunk: int = 3) -> List[str]:
    """Simple sentence-based chunking."""
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = '. '.join(sentences[i:i+sentences_per_chunk]) + '.'
        chunks.append(chunk)
    return chunks
```

### 7.4.2 Overlapping Chunks

Better results often come from overlapping chunks—this preserves context across boundaries.

```python
def chunk_with_overlap(text: str, chunk_size: int = 200, 
                       overlap: int = 50) -> List[str]:
    """Create overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks
```

### 7.4.3 Semantic Chunking

The most sophisticated approach: chunk at natural semantic boundaries.

```python
def semantic_chunk(text: str, similarity_threshold: float = 0.7) -> List[str]:
    """Chunk text at semantic boundaries."""
    sentences = text.split('. ')
    if len(sentences) < 2:
        return [text]
    
    chunks = [sentences[0]]
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        # Compare similarity between current and next sentence
        prev_emb = get_embedding(sentences[i-1])
        curr_emb = get_embedding(sentences[i])
        similarity = cosine_similarity(prev_emb, curr_emb)
        
        if similarity < similarity_threshold:
            # Start new chunk at semantic boundary
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return chunks
```

## 7.5 Advanced Retrieval Strategies

Simple cosine similarity is just the beginning. Let's explore more sophisticated retrieval methods.

### 7.5.1 Hybrid Search: Combining Dense and Sparse Retrieval

Dense retrieval (embeddings) captures semantic meaning. Sparse retrieval (keyword matching) captures exact terms. Combining them gives us the best of both worlds.

```python
class HybridSearchStore(SimpleDocumentStore):
    def keyword_score(self, query: str, document: str) -> float:
        """Simple keyword matching score (BM25-like)."""
        query_terms = set(query.lower().split())
        doc_terms = document.lower().split()
        
        # Term frequency
        matches = sum(1 for term in doc_terms if term in query_terms)
        return matches / len(doc_terms) if doc_terms else 0
    
    def hybrid_search(self, query: str, top_k: int = 3, 
                      alpha: float = 0.5) -> List[Dict]:
        """Combine semantic and keyword search."""
        # Semantic scores
        query_emb = get_embedding(query)
        semantic_scores = [cosine_similarity(query_emb, emb) 
                          for emb in self.embeddings]
        
        # Keyword scores
        keyword_scores = [self.keyword_score(query, doc["text"]) 
                         for doc in self.documents]
        
        # Combine with alpha weighting
        combined_scores = [
            alpha * sem + (1 - alpha) * key
            for sem, key in zip(semantic_scores, keyword_scores)
        ]
        
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]
```

### 7.5.2 Metadata Filtering

Sometimes we want to constrain our search by metadata—like searching only within a specific course or time period.

```python
def filtered_search(store: SimpleDocumentStore, query: str, 
                   filters: Dict, top_k: int = 3) -> List[Dict]:
    """Search with metadata filtering."""
    # First, filter by metadata
    filtered_indices = []
    for i, doc in enumerate(store.documents):
        match = all(doc["metadata"].get(k) == v 
                   for k, v in filters.items())
        if match:
            filtered_indices.append(i)
    
    if not filtered_indices:
        return []
    
    # Then, semantic search within filtered set
    query_emb = get_embedding(query)
    scores = [(i, cosine_similarity(query_emb, store.embeddings[i])) 
              for i in filtered_indices]
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return [store.documents[i] for i, _ in scores[:top_k]]
```

## 7.6 Query Transformation Techniques

Users don't always ask questions in the optimal way for retrieval. Query transformation helps bridge this gap.

### 7.6.1 Query Expansion with LLM

```python
def expand_query(original_query: str) -> List[str]:
    """Generate alternative phrasings of a query."""
    prompt = f"""Generate 3 alternative ways to ask this question,
each focusing on different aspects or using different terminology:

Original: {original_query}

Alternatives (one per line):"""
    
    response = ollama.generate(model="qwen2.5:latest", prompt=prompt)
    alternatives = [original_query]  # Include original
    alternatives.extend(response['response'].strip().split('\n'))
    return [q.strip('- ').strip() for q in alternatives if q.strip()]

# Example usage
query = "What are the prerequisites for the AI course?"
expanded = expand_query(query)
# Search with all variations and combine results
```

### 7.6.2 Hypothetical Document Embeddings (HyDE)

Instead of embedding the query directly, generate a hypothetical answer and embed *that*. This often better matches document embeddings.

```python
def hyde_retrieval(query: str, store: SimpleDocumentStore, 
                   top_k: int = 3) -> List[Dict]:
    """Retrieve using hypothetical document embeddings."""
    # Generate hypothetical answer
    prompt = f"Write a brief, factual answer to: {query}"
    response = ollama.generate(model="qwen2.5:latest", prompt=prompt)
    hypothetical_doc = response['response']
    
    # Embed and search with hypothetical answer
    hyp_emb = get_embedding(hypothetical_doc)
    scores = [cosine_similarity(hyp_emb, doc_emb) 
              for doc_emb in store.embeddings]
    
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [store.documents[i] for i in top_indices]
```

## 7.7 Context Management and Prompt Engineering

Retrieving documents is only half the battle. We need to present them effectively to the LLM.

### 7.7.1 Reranking Retrieved Documents

Not all retrieved documents are equally relevant. Reranking refines our initial retrieval.

```python
def rerank_with_llm(query: str, documents: List[Dict]) -> List[Dict]:
    """Use LLM to rerank documents by relevance."""
    scores = []
    
    for doc in documents:
        prompt = f"""On a scale of 0-10, how relevant is this document to the query?

Query: {query}
Document: {doc['text']}

Respond with only a number 0-10:"""
        
        response = ollama.generate(model="qwen2.5:latest", prompt=prompt)
        try:
            score = float(response['response'].strip())
        except ValueError:
            score = 5.0  # Default if parsing fails
        scores.append(score)
    
    # Sort by reranked scores
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked]
```

### 7.7.2 Citation and Source Attribution

Good RAG systems cite their sources—crucial for trust and verification.

```python
def rag_with_citations(question: str, store: SimpleDocumentStore) -> str:
    """RAG with source citations."""
    relevant_docs = store.search(question, top_k=3)
    
    # Number each document
    context_parts = []
    for i, doc in enumerate(relevant_docs, 1):
        context_parts.append(f"[{i}] {doc['text']}")
    context = "\n\n".join(context_parts)
    
    prompt = f"""Answer the question using the provided sources.
Cite sources using [1], [2], etc.

Sources:
{context}

Question: {question}

Answer with citations:"""
    
    response = ollama.generate(model="qwen2.5:latest", prompt=prompt)
    
    # Return answer with source texts
    answer = response['response']
    sources = "\n\n".join([f"[{i}] {doc['text']}" 
                          for i, doc in enumerate(relevant_docs, 1)])
    return f"{answer}\n\n---\nSources:\n{sources}"
```

## 7.8 Handling Multi-Turn Conversations

RAG becomes more complex with conversation history. We need to maintain context across turns.

```python
class ConversationalRAG:
    def __init__(self, store: SimpleDocumentStore):
        self.store = store
        self.history = []
    
    def query(self, user_message: str) -> str:
        """Handle conversational query with history."""
        # Build conversation context
        history_context = "\n".join([
            f"User: {h['user']}\nAssistant: {h['assistant']}"
            for h in self.history[-3:]  # Last 3 turns
        ])
        
        # Retrieve relevant documents
        relevant_docs = self.store.search(user_message, top_k=2)
        doc_context = "\n\n".join([doc["text"] for doc in relevant_docs])
        
        prompt = f"""Previous conversation:
{history_context}

Relevant information:
{doc_context}

User: {user_message}
Assistant:"""
        
        response = ollama.generate(model="qwen2.5:latest", prompt=prompt)
        answer = response['response']
        
        # Update history
        self.history.append({"user": user_message, "assistant": answer})
        return answer

# Example usage
conv_rag = ConversationalRAG(store)
print(conv_rag.query("When are office hours?"))
print(conv_rag.query("Where exactly?"))  # Follows up on previous question
```

## 7.9 Evaluation Metrics for RAG Systems

How do we know if our RAG system is working well? We need metrics.

### 7.9.1 Retrieval Metrics

```python
def evaluate_retrieval(queries: List[str], 
                       ground_truth: List[List[int]], 
                       store: SimpleDocumentStore) -> Dict[str, float]:
    """Evaluate retrieval quality."""
    precisions = []
    recalls = []
    
    for query, relevant_ids in zip(queries, ground_truth):
        # Retrieve documents
        retrieved = store.search(query, top_k=5)
        retrieved_ids = [store.documents.index(doc) for doc in retrieved]
        
        # Calculate precision and recall
        relevant_set = set(relevant_ids)
        retrieved_set = set(retrieved_ids)
        
        true_positives = len(relevant_set & retrieved_set)
        precision = true_positives / len(retrieved_set) if retrieved_set else 0
        recall = true_positives / len(relevant_set) if relevant_set else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    return {
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1": 2 * np.mean(precisions) * np.mean(recalls) / 
              (np.mean(precisions) + np.mean(recalls))
    }
```

### 7.9.2 End-to-End RAG Quality

```python
def evaluate_answer_quality(question: str, answer: str, 
                           context: str) -> Dict[str, float]:
    """Evaluate answer quality with LLM-as-judge."""
    prompt = f"""Evaluate this answer on three criteria (0-10 scale):

1. Faithfulness: Does the answer only use information from the context?
2. Relevance: Does the answer address the question?
3. Completeness: Does the answer fully address the question?

Context: {context}
Question: {question}
Answer: {answer}

Respond in format:
Faithfulness: X
Relevance: Y
Completeness: Z"""
    
    response = ollama.generate(model="qwen2.5:latest", prompt=prompt)
    
    # Parse scores
    scores = {}
    for line in response['response'].split('\n'):
        if ':' in line:
            metric, value = line.split(':')
            try:
                scores[metric.strip().lower()] = float(value.strip())
            except ValueError:
                continue
    
    return scores
```

## 7.10 Advanced Patterns and Architectures

### 7.10.1 Agentic RAG

What if the LLM could decide *when* to retrieve information?

```python
def agentic_rag(question: str, store: SimpleDocumentStore, 
                max_iterations: int = 3) -> str:
    """LLM decides when to retrieve information."""
    
    for iteration in range(max_iterations):
        # Ask LLM if it needs more information
        prompt = f"""You are answering: {question}

Do you need to retrieve information? Respond with:
RETRIEVE: [search query] if you need information
ANSWER: [your answer] if you have enough information

Response:"""
        
        response = ollama.generate(model="qwen2.5:latest", prompt=prompt)
        response_text = response['response'].strip()
        
        if response_text.startswith("ANSWER:"):
            return response_text[7:].strip()
        
        elif response_text.startswith("RETRIEVE:"):
            query = response_text[9:].strip()
            docs = store.search(query, top_k=2)
            context = "\n".join([doc["text"] for doc in docs])
            
            # Update question with retrieved context
            question = f"{question}\n\nRetrieved information:\n{context}"
        
        else:
            # Default to answering
            return response_text
    
    return "Unable to answer after maximum iterations."
```

### 7.10.2 Corrective RAG (CRAG)

CRAG evaluates retrieved documents and corrects course if needed.

```python
def corrective_rag(question: str, store: SimpleDocumentStore) -> str:
    """RAG with self-correction."""
    # Initial retrieval
    docs = store.search(question, top_k=3)
    
    # Evaluate relevance of each document
    relevant_docs = []
    for doc in docs:
        prompt = f"""Is this document relevant to the question?

Question: {question}
Document: {doc['text']}

Respond with only YES or NO:"""
        
        response = ollama.generate(model="qwen2.5:latest", prompt=prompt)
        if "YES" in response['response'].upper():
            relevant_docs.append(doc)
    
    # If no relevant docs, try web search or query reformulation
    if not relevant_docs:
        # Reformulate query
        new_query = expand_query(question)[1]  # Use first alternative
        docs = store.search(new_query, top_k=3)
        relevant_docs = docs[:1]  # Take best match
    
    # Generate answer with relevant docs
    context = "\n\n".join([doc["text"] for doc in relevant_docs])
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    response = ollama.generate(model="qwen2.5:latest", prompt=prompt)
    return response['response']
```

## 7.11 Privacy and Security Considerations

RAG systems often handle sensitive information. Let's discuss privacy-preserving techniques.

### 7.11.1 Local-Only RAG

Running everything locally prevents data leakage to external services.

```python
# Using Ollama, we're already local!
# All embeddings and generation happen on-device

def private_rag_query(question: str, private_docs: List[str]) -> str:
    """RAG entirely on local machine."""
    local_store = SimpleDocumentStore()
    for doc in private_docs:
        local_store.add_document(doc)
    
    # Everything runs locally via Ollama
    relevant = local_store.search(question, top_k=2)
    context = "\n".join([doc["text"] for doc in relevant])
    
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    response = ollama.generate(model="qwen2.5:latest", prompt=prompt)
    return response['response']
```

### 7.11.2 Redaction and Access Control

```python
def rag_with_redaction(question: str, store: SimpleDocumentStore, 
                       user_clearance: str) -> str:
    """RAG with access control."""
    # Retrieve documents
    docs = store.search(question, top_k=3)
    
    # Filter by access level
    accessible_docs = [
        doc for doc in docs 
        if doc["metadata"].get("clearance", "public") == user_clearance
    ]
    
    if not accessible_docs:
        return "No accessible documents found for your clearance level."
    
    # Redact sensitive patterns (e.g., SSNs, credit cards)
    import re
    context_parts = []
    for doc in accessible_docs:
        text = doc["text"]
        text = re.sub(r'\d{3}-\d{2}-\d{4}', '[REDACTED-SSN]', text)
        text = re.sub(r'\d{4}-\d{4}-\d{4}-\d{4}', '[REDACTED-CC]', text)
        context_parts.append(text)
    
    context = "\n\n".join(context_parts)
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    response = ollama.generate(model="qwen2.5:latest", prompt=prompt)
    return response['response']
```

## 7.12 Building a Complete RAG Application

Let's bring everything together into a production-ready system.

```python
class ProductionRAGSystem:
    def __init__(self, model: str = "qwen2.5:latest"):
        self.store = HybridSearchStore()
        self.model = model
        self.query_cache = {}
    
    def ingest_documents(self, documents: List[str], 
                        metadata: List[Dict] = None):
        """Ingest and chunk documents."""
        for i, doc in enumerate(documents):
            chunks = chunk_with_overlap(doc, chunk_size=300, overlap=50)
            doc_metadata = metadata[i] if metadata else {}
            
            for j, chunk in enumerate(chunks):
                chunk_meta = {**doc_metadata, "chunk_id": j}
                self.store.add_document(chunk, chunk_meta)
    
    def query(self, question: str, use_cache: bool = True, 
              top_k: int = 3) -> Dict:
        """Main query interface with caching."""
        # Check cache
        if use_cache and question in self.query_cache:
            return self.query_cache[question]
        
        # Retrieve relevant documents
        relevant_docs = self.store.hybrid_search(question, top_k=top_k)
        
        # Rerank
        relevant_docs = rerank_with_llm(question, relevant_docs)
        
        # Build prompt with context
        context = "\n\n".join([
            f"[{i+1}] {doc['text']}" 
            for i, doc in enumerate(relevant_docs)
        ])
        
        prompt = f"""Answer using the provided context. Cite sources with [1], [2], etc.

Context:
{context}

Question: {question}

Answer:"""
        
        response = ollama.generate(model=self.model, prompt=prompt)
        
        result = {
            "answer": response['response'],
            "sources": relevant_docs,
            "context": context
        }
        
        # Cache result
        if use_cache:
            self.query_cache[question] = result
        
        return result

# Example usage
rag = ProductionRAGSystem()
rag.ingest_documents([
    "Python was created by Guido van Rossum in 1991. It emphasizes code readability.",
    "Python supports multiple programming paradigms including procedural, OOP, and functional.",
    "Python's standard library is extensive, covering file I/O, networking, and more."
])

result = rag.query("Who created Python?")
print(result["answer"])
print("\nSources:")
for i, source in enumerate(result["sources"], 1):
    print(f"[{i}] {source['text'][:100]}...")
```

## 7.13 Common Pitfalls and Debugging

### The "Lost in the Middle" Problem

LLMs often ignore information in the middle of long contexts. Solution: put most important retrieved docs at the beginning and end.

```python
def reorder_for_attention(docs: List[Dict]) -> List[Dict]:
    """Reorder docs to avoid 'lost in the middle' problem."""
    if len(docs) <= 2:
        return docs
    
    # Place highest-relevance docs at start and end
    reordered = [docs[0]]  # Best doc first
    reordered.extend(docs[2:-1])  # Middle docs
    reordered.append(docs[1])  # Second-best doc last
    return reordered
```

### The Hallucination Check

Always verify the LLM is using retrieved context, not making things up.

```python
def verify_grounding(answer: str, context: str) -> bool:
    """Check if answer is grounded in context."""
    prompt = f"""Does this answer contain information NOT present in the context?

Context: {context}
Answer: {answer}

Respond with YES if answer goes beyond context, NO otherwise:"""
    
    response = ollama.generate(model="qwen2.5:latest", prompt=prompt)
    return "NO" in response['response'].upper()
```

## 7.14 Future Directions and Research

RAG is rapidly evolving. Current research directions include:

- **Adaptive retrieval**: Dynamically deciding how many documents to retrieve
- **Multi-modal RAG**: Retrieving images, tables, and structured data alongside text
- **Graph RAG**: Using knowledge graphs to enhance retrieval with relational context
- **Federated RAG**: Retrieving from distributed sources while preserving privacy

## 7.15 Chapter Summary

We've journeyed from basic retrieval to sophisticated RAG architectures:

1. RAG augments LLMs with external knowledge through retrieval
2. The pipeline: index documents → retrieve relevant context → generate answer
3. Chunking strategies balance context preservation with retrieval precision
4. Advanced techniques: hybrid search, query transformation, reranking
5. Production systems need caching, access control, and evaluation metrics
6. Privacy-preserving RAG keeps sensitive data local

**Key Insight**: RAG transforms static language models into dynamic systems that can access, reason about, and synthesize information from vast knowledge bases—all while maintaining the benefits of local, privacy-preserving inference.

In the next chapter, we'll explore how to combine RAG with agents that can plan, use tools, and solve complex multi-step problems.

---

## Exercises

1. **Implement semantic chunking** that preserves paragraph boundaries while respecting max chunk size constraints.

2. **Build a multi-document RAG system** that retrieves from different specialized knowledge bases (e.g., one for academic papers, one for news articles) and merges results.

3. **Create an evaluation harness** for your RAG system using a dataset of questions and expected answers.

4. **Design a RAG system for code search** that retrieves relevant code snippets and generates programming solutions.

5. **Implement a "chain of retrieval"** where the LLM can make multiple retrieval calls, refining its search based on previous results.

**Challenge**: Build a RAG system that can explain *why* it chose specific documents, providing transparency into its retrieval decisions.