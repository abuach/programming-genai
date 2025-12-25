# Chapter 7: Code Embeddings – Teaching Machines to Understand Programs

What if we could teach computers to understand code the way they understand human language? What if a program could recognize that `for i in range(n)` and `while i < n` are semantically related, even though they use different syntax? Welcome to the fascinating world of **code embeddings**, where we transform the discrete, symbolic nature of programming languages into continuous mathematical spaces that machines can reason about.

## 7.1 The Vector Space of Code

Imagine every piece of code living in a high-dimensional space—a vast landscape where similar programs cluster together like neighborhoods in a city. A sorting algorithm lives near other sorting algorithms. Security vulnerabilities huddle together in their own corner. This isn't science fiction; it's the practical reality of code embeddings.

### What Are Embeddings?

An **embedding** is a learned mapping from discrete objects (like words, sentences, or code) into continuous vector spaces. Instead of treating code as raw text, we represent it as points in a mathematical space where:

- **Distance** captures semantic similarity
- **Direction** encodes meaningful relationships
- **Arithmetic** reveals hidden patterns

Let's start with a simple example using a pre-trained model:

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load UniXcoder - a unified model for code understanding
tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
model = AutoModel.from_pretrained("microsoft/unixcoder-base")

def embed_code(code_snippet):
    """Convert code into a vector embedding."""
    tokens = tokenizer(code_snippet, return_tensors="pt", 
                       truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**tokens)
        # Use [CLS] token embedding as code representation
        embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    
    return embedding

# Try it out!
code1 = "def bubble_sort(arr): return sorted(arr)"
code2 = "def quick_sort(lst): return sorted(lst)"
code3 = "print('Hello, World!')"

emb1 = embed_code(code1)
emb2 = embed_code(code2)
emb3 = embed_code(code3)

print(f"Embedding dimension: {emb1.shape[0]}")
```

This code transforms three Python snippets into 768-dimensional vectors. But what do these vectors actually *mean*?

## 7.2 Measuring Similarity in Code Space

The most immediate application of code embeddings is **similarity measurement**. Given two pieces of code, how semantically related are they?

```python
import torch.nn.functional as F

def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings."""
    return F.cosine_similarity(emb1.unsqueeze(0), 
                              emb2.unsqueeze(0)).item()

sim_sorts = cosine_similarity(emb1, emb2)
sim_sort_print = cosine_similarity(emb1, emb3)

print(f"Similarity (bubble_sort vs quick_sort): {sim_sorts:.4f}")
print(f"Similarity (bubble_sort vs print): {sim_sort_print:.4f}")
```

You'll notice the two sorting functions have much higher similarity than the sorting function and the print statement. The model has learned that `bubble_sort` and `quick_sort` are related concepts, even with different names!

### Why Cosine Similarity?

Cosine similarity measures the angle between vectors, ranging from -1 (opposite) to 1 (identical). It's perfect for embeddings because:

- It's **scale-invariant**: only direction matters, not magnitude
- It captures **semantic relatedness** better than Euclidean distance
- It's computationally efficient for high-dimensional spaces

## 7.3 Building a Code Search Engine

Now let's build something practical: a semantic code search engine that finds relevant functions even when they don't share keywords.

```python
import numpy as np

class CodeSearchEngine:
    def __init__(self):
        self.code_snippets = []
        self.embeddings = []
    
    def index(self, code_snippet):
        """Add code to the search index."""
        embedding = embed_code(code_snippet)
        self.code_snippets.append(code_snippet)
        self.embeddings.append(embedding)
    
    def search(self, query, top_k=3):
        """Find most similar code snippets."""
        query_emb = embed_code(query)
        
        similarities = [
            cosine_similarity(query_emb, emb) 
            for emb in self.embeddings
        ]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'code': self.code_snippets[idx],
                'similarity': similarities[idx]
            })
        
        return results

# Build a small code repository
search_engine = CodeSearchEngine()

search_engine.index("def add(a, b): return a + b")
search_engine.index("def multiply(x, y): return x * y")
search_engine.index("def read_file(path): return open(path).read()")
search_engine.index("def sum_list(items): return sum(items)")

# Search for "function that adds numbers"
results = search_engine.search("combine two numbers", top_k=2)

for i, result in enumerate(results, 1):
    print(f"\n{i}. Similarity: {result['similarity']:.4f}")
    print(f"   {result['code']}")
```

Notice how the search finds `add` and `sum_list` as the most relevant results, even though our query doesn't contain the word "add" or "sum"!

## 7.4 Code Vector Arithmetic: The Algebra of Programming

Here's where things get truly magical. Remember the famous word2vec example: `king - man + woman = queen`? Code embeddings exhibit similar algebraic properties!

### The Intuition

If embeddings capture semantic relationships, then vector arithmetic should reveal patterns:

- `sorting_function - recursion + iteration ≈ iterative_sort`
- `python_function - python + javascript ≈ javascript_function`
- `vulnerable_code - vulnerability ≈ safe_code`

Let's implement and test this:

```python
def code_analogy(a, b, c, candidates, top_k=1):
    """
    Solve: a is to b as c is to ?
    (i.e., find d such that a - b ≈ c - d)
    """
    emb_a = embed_code(a)
    emb_b = embed_code(b)
    emb_c = embed_code(c)
    
    # Compute target vector: c + (a - b)
    target = emb_c + (emb_a - emb_b)
    
    # Find closest candidate
    similarities = []
    for candidate in candidates:
        emb_d = embed_code(candidate)
        sim = cosine_similarity(target, emb_d)
        similarities.append((candidate, sim))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]

# Example: "recursive" is to "iterative" as "factorial_recursive" is to ?
recursive = "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
iterative = "def count(n):\n  i = 0\n  while i < n: i += 1"

fact_recursive = "def fact(n):\n  if n == 0: return 1\n  return n * fact(n-1)"

candidates = [
    "def fact_iter(n):\n  result = 1\n  for i in range(1, n+1): result *= i\n  return result",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "print('Hello')",
]

print("Analogy: recursive : iterative :: factorial_recursive : ?")
results = code_analogy(recursive, iterative, fact_recursive, candidates)

for code, score in results:
    print(f"\nScore: {score:.4f}")
    print(code)
```

The model should identify the iterative factorial as the best match! This demonstrates that embeddings capture **conceptual transformations** like "converting recursion to iteration."

## 7.5 Multi-Modal Code Understanding with Ollama

Modern code models can handle more than just code snippets—they understand **natural language descriptions**, **documentation**, and even **comments**. Let's use Ollama to explore multi-modal code understanding:

```python
import ollama

def explain_code_with_context(code, context):
    """Use Ollama to explain code in natural language."""
    prompt = f"""Given this context: {context}
    
Explain this code:
{code}

Provide a concise explanation of what the code does and how it works."""
    
    response = ollama.generate(
        model='codellama',
        prompt=prompt
    )
    
    return response['response']

# Example: Understanding code in context
code = """
def validate_email(email):
    import re
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))
"""

context = "This is part of a user registration system"
explanation = explain_code_with_context(code, context)
print(f"Explanation:\n{explanation}")
```

### Hybrid Retrieval: Best of Both Worlds

We can combine embedding-based semantic search with LLM generation:

```python
def hybrid_code_assistant(query, code_base):
    """
    1. Use embeddings to find relevant code
    2. Use LLM to generate answer with context
    """
    # Step 1: Semantic search
    search_engine = CodeSearchEngine()
    for code in code_base:
        search_engine.index(code)
    
    relevant = search_engine.search(query, top_k=3)
    
    # Step 2: Generate answer with context
    context = "\n\n".join([r['code'] for r in relevant])
    
    prompt = f"""Relevant code snippets:
{context}

Question: {query}

Answer based on the code above:"""
    
    response = ollama.generate(
        model='codellama',
        prompt=prompt
    )
    
    return {
        'relevant_code': relevant,
        'answer': response['response']
    }

# Test it
code_base = [
    "def authenticate(username, password): return check_credentials(username, password)",
    "def authorize(user, resource): return user.has_permission(resource)",
    "def hash_password(pwd): return bcrypt.hashpw(pwd.encode(), bcrypt.gensalt())",
]

result = hybrid_code_assistant(
    "How do I securely handle passwords?",
    code_base
)

print("Answer:", result['answer'])
```

## 7.6 Advanced Application: Clone Detection

One of the most powerful applications of code embeddings is detecting code clones—functionally similar code that may have different surface forms.

```python
def detect_clones(code_snippets, threshold=0.85):
    """Find pairs of similar code snippets (potential clones)."""
    embeddings = [embed_code(code) for code in code_snippets]
    clones = []
    
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            
            if sim > threshold:
                clones.append({
                    'code1': code_snippets[i],
                    'code2': code_snippets[j],
                    'similarity': sim
                })
    
    return clones

# Test with code clones
snippets = [
    "def max_val(a, b): return a if a > b else b",
    "def maximum(x, y): return x if x > y else y",  # Clone!
    "def add(a, b): return a + b",
    "def get_max(p, q):\n  if p > q: return p\n  return q",  # Clone!
]

clones = detect_clones(snippets, threshold=0.85)

print(f"Found {len(clones)} clone pairs:\n")
for clone in clones:
    print(f"Similarity: {clone['similarity']:.4f}")
    print(f"Code 1: {clone['code1']}")
    print(f"Code 2: {clone['code2']}\n")
```

This technique is invaluable for:
- **Code review**: Finding duplicated logic
- **Refactoring**: Identifying candidates for abstraction  
- **License compliance**: Detecting copied code
- **Bug propagation**: Finding related vulnerable code

## 7.7 Visualizing Code Embeddings

High-dimensional embeddings are hard to visualize, but we can use dimensionality reduction to peek into code space:

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_code_space(code_snippets, labels=None):
    """Reduce embeddings to 2D and plot."""
    # Generate embeddings
    embeddings = [embed_code(code).numpy() for code in code_snippets]
    embeddings = np.array(embeddings)
    
    # Reduce to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         alpha=0.6, s=100)
    
    # Add labels
    if labels:
        for i, label in enumerate(labels):
            plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                        fontsize=8, alpha=0.7)
    
    plt.title("Code Embedding Space (2D Projection)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.savefig("code_embeddings_viz.png", dpi=150)
    plt.close()
    
    return embeddings_2d

# Visualize different code categories
test_codes = [
    "def sort_list(arr): return sorted(arr)",
    "def bubble_sort(a): pass",
    "def read_file(path): return open(path).read()",
    "def write_file(path, data): open(path, 'w').write(data)",
    "print('Hello, world!')",
    "def greet(): print('Hi!')",
]

labels = ["sort1", "sort2", "read", "write", "print1", "print2"]
coords = visualize_code_space(test_codes, labels)

print("Visualization saved! Notice how similar functions cluster together.")
```

## 7.8 The Theory Behind Code Embeddings

Why do code embeddings work so well? The secret lies in how these models are trained.

### Pre-training Objectives

UniXcoder and similar models use several training objectives:

1. **Masked Language Modeling (MLM)**: Randomly mask tokens and predict them
   ```python
   # Input:  "def sort([MASK]): return sorted(arr)"
   # Predict: "arr"
   ```

2. **Contrastive Learning**: Similar code should have similar embeddings
   - Positive pairs: (code, docstring), (original, paraphrased)
   - Negative pairs: Random unrelated code

3. **Next Token Prediction**: Predict the next token in a sequence

Here's a simplified example of the training objective:

```python
def contrastive_loss(code_emb, doc_emb, temperature=0.07):
    """
    InfoNCE loss: Pull together (code, doc) pairs,
    push apart unrelated pairs.
    """
    # Normalize embeddings
    code_emb = F.normalize(code_emb, dim=-1)
    doc_emb = F.normalize(doc_emb, dim=-1)
    
    # Compute similarity matrix
    similarity = torch.matmul(code_emb, doc_emb.T) / temperature
    
    # Labels: diagonal elements are positive pairs
    batch_size = code_emb.size(0)
    labels = torch.arange(batch_size)
    
    # Cross-entropy loss
    loss = F.cross_entropy(similarity, labels)
    
    return loss

# This is conceptual - actual training requires large datasets!
```

### What Makes Good Code Embeddings?

The ideal code embedding model should:

- **Capture semantics**, not just syntax
- **Be robust** to variable naming and style
- **Understand context** across multiple lines
- **Generalize** to unseen code patterns

## 7.9 Practical Applications in Software Engineering

Let's build a complete application that uses embeddings for **intelligent code completion**:

```python
class SemanticCodeCompleter:
    def __init__(self, codebase):
        self.search_engine = CodeSearchEngine()
        
        # Index entire codebase
        for snippet in codebase:
            self.search_engine.index(snippet)
    
    def suggest_completion(self, partial_code, num_suggestions=3):
        """Suggest completions based on semantic similarity."""
        # Find similar code
        similar = self.search_engine.search(partial_code, top_k=10)
        
        # Use LLM to generate contextual completion
        context = "\n".join([s['code'] for s in similar[:3]])
        
        prompt = f"""Similar code patterns:
{context}

Complete this code:
{partial_code}

Provide only the completion (no explanation):"""
        
        response = ollama.generate(
            model='codellama',
            prompt=prompt,
            options={'temperature': 0.7, 'num_predict': 100}
        )
        
        return {
            'completion': response['response'],
            'similar_patterns': similar[:num_suggestions]
        }

# Example usage
codebase = [
    "def validate_email(email):\n  import re\n  return bool(re.match(r'^[\\w.-]+@[\\w.-]+\\.\\w+$', email))",
    "def validate_phone(phone):\n  import re\n  return bool(re.match(r'^\\+?1?\\d{9,15}$', phone))",
    "def sanitize_input(text):\n  return text.strip().lower()",
]

completer = SemanticCodeCompleter(codebase)

partial = "def validate_username(username):"
suggestion = completer.suggest_completion(partial)

print(f"Suggested completion:\n{suggestion['completion']}")
```

## 7.10 Limitations and Future Directions

While code embeddings are powerful, they have limitations:

**Current Challenges:**
- **Computational cost**: Large models require significant resources
- **Context length**: Limited to 512-2048 tokens typically
- **Execution semantics**: Embeddings don't capture runtime behavior
- **Adversarial fragility**: Small changes can cause large embedding shifts

**Future Directions:**
- **Program synthesis**: Generating code from embeddings
- **Formal verification**: Connecting embeddings to correctness proofs
- **Multi-language models**: Unified embeddings across programming languages
- **Execution-aware embeddings**: Incorporating runtime behavior

## 7.11 Hands-On Exercise: Build Your Own Code Recommender

Let's wrap up with a complete project combining everything we've learned:

```python
class IntelligentCodeRecommender:
    def __init__(self):
        self.index = CodeSearchEngine()
        
    def add_snippet(self, code, metadata=None):
        """Add code with optional metadata (tags, description)."""
        self.index.index(code)
    
    def recommend_by_task(self, task_description):
        """Recommend code snippets for a given task."""
        results = self.index.search(task_description, top_k=5)
        return results
    
    def find_refactoring_opportunities(self, code):
        """Find similar code that might be refactored together."""
        embedding = embed_code(code)
        
        similar = []
        for i, existing_code in enumerate(self.index.code_snippets):
            existing_emb = self.index.embeddings[i]
            sim = cosine_similarity(embedding, existing_emb)
            
            if 0.7 < sim < 0.95:  # Similar but not identical
                similar.append({
                    'code': existing_code,
                    'similarity': sim
                })
        
        return sorted(similar, key=lambda x: x['similarity'], reverse=True)

# Build a recommender
recommender = IntelligentCodeRecommender()

# Add some code snippets
recommender.add_snippet("def read_json(path): import json; return json.load(open(path))")
recommender.add_snippet("def write_json(path, data): import json; json.dump(data, open(path, 'w'))")
recommender.add_snippet("def parse_csv(path): import csv; return list(csv.reader(open(path)))")

# Get recommendations
recs = recommender.recommend_by_task("load data from a file")
print("Recommendations:")
for r in recs[:3]:
    print(f"- {r['code']} (score: {r['similarity']:.3f})")
```

## Summary

Code embeddings transform how we work with programs, enabling:

✓ **Semantic code search** beyond keyword matching  
✓ **Clone detection** for maintainability  
✓ **Intelligent completion** aware of coding patterns  
✓ **Vector arithmetic** revealing program structure  
✓ **Multi-modal understanding** bridging code and language  

The journey from discrete symbols to continuous representations mirrors a profound shift in how machines understand computation itself. As you continue exploring AI and programming, remember: every line of code now lives in a vast mathematical space, waiting to reveal its secrets.

---

**Further Reading:**
- *CodeBERT: A Pre-Trained Model for Programming and Natural Languages* (Feng et al., 2020)
- *UniXcoder: Unified Cross-Modal Pre-training for Code Representation* (Guo et al., 2022)
- The Anthropic documentation on code understanding models

**Challenge Problems:**
1. Extend the clone detector to handle cross-language clones (Python → JavaScript)
2. Build a vulnerability detector using embeddings trained on CVE databases
3. Create a code style transfer system: convert Java → Pythonic style

*In the next chapter, we'll explore how these embeddings power even more sophisticated AI systems: from program synthesis to automated theorem proving. The best is yet to come!*