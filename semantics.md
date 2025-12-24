# Chapter 7: Semantic Similarity and the Geometry of Meaning

## Introduction: When Words Become Numbers

Here's a joke that perfectly captures what we're about to explore: *Why did the word embedding go to therapy? Because it had too many dimensions to its personality!*

But seriously—imagine if we could take the meaning of a word, a sentence, or even an entire program, and represent it as a point in space. Not metaphorical space, but actual mathematical space where we can measure distances, find neighbors, and perform calculations. This is the fundamental idea behind **embeddings**, and it's one of the most powerful concepts in modern AI.

In this chapter, we'll explore how machines learned to understand similarity—not through rules programmed by humans, but by discovering patterns in vast amounts of text and code. We'll see how "king" minus "man" plus "woman" genuinely equals "queen" in embedding space, and how the same principles that help us find similar words can help us find similar functions in a codebase.

## 7.1: The Problem of Meaning

Let's start with a simple question: How similar are these two sentences?

1. "The cat sat on the mat."
2. "A feline rested on the rug."

To a human, these are clearly similar—nearly identical in meaning. But to a computer working with raw text, they share only one word: "the." By a naive word-matching metric, they're about 14% similar (1 word in common out of 7-8 total). This is obviously wrong.

The challenge is that **meaning doesn't live in the surface form of text**. It lives in the relationships between concepts, in the contexts where words appear, and in the ways ideas connect to each other.

### The Traditional Approach: Bag of Words

Before embeddings, the standard approach was to represent text as a "bag of words"—literally just counting which words appear:

```python
from collections import Counter

def bag_of_words(text):
    """Convert text to word frequency dictionary."""
    return Counter(text.lower().split())

sentence1 = "The cat sat on the mat"
sentence2 = "A feline rested on the rug"

bow1 = bag_of_words(sentence1)
bow2 = bag_of_words(sentence2)

print(f"Sentence 1: {bow1}")
print(f"Sentence 2: {bow2}")
```

This approach loses all word order and, more importantly, treats every word as completely unrelated to every other word. "Cat" and "feline" are no more similar than "cat" and "economics."

## 7.2: The Distributional Hypothesis

The breakthrough came from a deceptively simple idea, articulated by linguist John Firth in 1957: **"You shall know a word by the company it keeps."**

This is called the **distributional hypothesis**: words that appear in similar contexts tend to have similar meanings. Think about it:

- "The ___ chased the mouse" 
- "The ___ climbed the tree"
- "The ___ meowed loudly"

What word fits in all these blanks? Probably "cat." And if you saw a new word, say "feline," appearing in the same contexts, you'd reasonably conclude it means something similar.

This insight is profound because it gives us a way to learn meaning from raw text, without anyone having to explicitly define what words mean.

## 7.3: Word Embeddings: Words as Vectors

A **word embedding** represents each word as a vector—a point in high-dimensional space. Typically, these vectors have 50, 100, 300, or even more dimensions. Words with similar meanings end up close together in this space.

Let's build intuition with a toy example in 2D:

```python
import numpy as np
import matplotlib.pyplot as plt

# Toy embeddings for animal words (2D for visualization)
embeddings = {
    'cat': np.array([0.8, 0.9]),
    'dog': np.array([0.9, 0.8]),
    'kitten': np.array([0.75, 0.95]),
    'puppy': np.array([0.95, 0.75]),
    'car': np.array([0.1, 0.1]),
    'vehicle': np.array([0.15, 0.05])
}

# Visualize
plt.figure(figsize=(8, 8))
for word, vec in embeddings.items():
    plt.scatter(vec[0], vec[1], s=100)
    plt.annotate(word, vec, fontsize=12)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Toy Word Embeddings in 2D Space')
plt.grid(True, alpha=0.3)
plt.show()
```

In this space, `cat` and `kitten` are close together, as are `dog` and `puppy`. But `car` is far from the animals. The geometry encodes semantic relationships!

## 7.4: Measuring Similarity: Cosine Distance

How do we quantify similarity between vectors? The most common metric is **cosine similarity**, which measures the angle between vectors:

$$\text{cosine similarity}(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{|\vec{a}| \cdot |\vec{b}|}$$

This gives us a value between -1 (opposite) and 1 (identical), with 0 meaning perpendicular (unrelated).

```python
def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Compare similarities
cat_dog = cosine_similarity(embeddings['cat'], embeddings['dog'])
cat_kitten = cosine_similarity(embeddings['cat'], embeddings['kitten'])
cat_car = cosine_similarity(embeddings['cat'], embeddings['car'])

print(f"cat-dog similarity: {cat_dog:.3f}")
print(f"cat-kitten similarity: {cat_kitten:.3f}")
print(f"cat-car similarity: {cat_car:.3f}")
```

Why cosine instead of Euclidean distance? Because we care about direction (meaning) more than magnitude. Two vectors pointing in the same direction are similar, even if one is longer.

## 7.5: Word2Vec: Learning Embeddings from Context

The Word2Vec algorithm, introduced by Mikolov et al. in 2013, was a watershed moment. It learns embeddings by training a neural network to predict words from their context (or vice versa).

### The Skip-Gram Model

The **skip-gram** variant works like this: given a center word, predict the surrounding context words. For example, in "the quick brown fox jumps," if "brown" is the center word, we try to predict "quick" and "fox."

```python
# Simplified skip-gram training concept (not production code)
def generate_training_pairs(text, window_size=2):
    """Generate (center_word, context_word) pairs."""
    words = text.split()
    pairs = []
    
    for i, center in enumerate(words):
        # Get context words within window
        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)
        
        for j in range(start, end):
            if i != j:  # Don't pair word with itself
                pairs.append((center, words[j]))
    
    return pairs

text = "the quick brown fox jumps over the lazy dog"
pairs = generate_training_pairs(text, window_size=2)
print(f"Training pairs: {pairs[:10]}")
```

The network learns by adjusting embeddings so that words appearing in similar contexts have similar vectors. The magic is that these vectors capture semantic relationships we never explicitly programmed!

## 7.6: Vector Arithmetic and Analogies

One of the most fascinating properties of word embeddings is that they support arithmetic. The famous example:

$$\text{king} - \text{man} + \text{woman} \approx \text{queen}$$

Let's see this with real embeddings:

```python
# Using pretrained embeddings (conceptual example)
def analogy(embeddings, a, b, c):
    """Solve analogy: a is to b as c is to ?"""
    # king - man + woman = ?
    result_vec = embeddings[a] - embeddings[b] + embeddings[c]
    
    # Find closest word to result
    best_word = None
    best_sim = -1
    
    for word, vec in embeddings.items():
        if word in [a, b, c]:  # Exclude input words
            continue
        sim = cosine_similarity(result_vec, vec)
        if sim > best_sim:
            best_sim = sim
            best_word = word
    
    return best_word

# Would return 'queen' with real embeddings!
```

This works because the embedding space captures relationships. The "royalty" dimension and the "gender" dimension exist implicitly in the geometry, learned entirely from data.

## 7.7: Sentence Embeddings: Beyond Individual Words

Words are useful, but we often need to understand entire sentences or documents. How do we embed a sentence?

### Simple Averaging (Bag of Words Embeddings)

The simplest approach: average the word vectors.

```python
def sentence_embedding_average(sentence, word_embeddings):
    """Create sentence embedding by averaging word vectors."""
    words = sentence.lower().split()
    vectors = [word_embeddings[w] for w in words if w in word_embeddings]
    
    if not vectors:
        return np.zeros(len(next(iter(word_embeddings.values()))))
    
    return np.mean(vectors, axis=0)

# Example usage
sent1 = "The cat sat on the mat"
sent2 = "A feline rested on the rug"

emb1 = sentence_embedding_average(sent1, embeddings)
emb2 = sentence_embedding_average(sent2, embeddings)

similarity = cosine_similarity(emb1, emb2)
print(f"Sentence similarity: {similarity:.3f}")
```

This is better than bag-of-words counting, but it still loses word order and grammar.

### Modern Approaches: Transformers

Today's state-of-the-art sentence embeddings come from **transformer models** like BERT and sentence-transformers. These models use attention mechanisms to understand how words relate within a sentence.

```python
# Using sentence-transformers (modern approach)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "The cat sat on the mat",
    "A feline rested on the rug",
    "Python is a programming language"
]

# Get embeddings (each is a 384-dimensional vector)
embeddings = model.encode(sentences)

# Calculate similarities
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(embeddings)
print(f"Similarity matrix:\n{similarities}")
```

The first two sentences will have high similarity (~0.7-0.8), while the third is dissimilar (~0.1-0.2).

## 7.8: Applications of Text Embeddings

Embeddings power countless modern applications:

### Semantic Search

Traditional search matches keywords. Semantic search finds meaning:

```python
def semantic_search(query, documents, model):
    """Find most relevant documents using embeddings."""
    query_emb = model.encode([query])
    doc_embs = model.encode(documents)
    
    similarities = cosine_similarity(query_emb, doc_embs)[0]
    
    # Return documents sorted by relevance
    ranked = sorted(zip(documents, similarities), 
                   key=lambda x: x[1], reverse=True)
    return ranked

docs = [
    "Machine learning is a subset of AI",
    "Python is great for data science",
    "The weather today is sunny",
    "Neural networks learn from data"
]

query = "artificial intelligence and algorithms"
results = semantic_search(query, docs, model)

for doc, score in results:
    print(f"{score:.3f}: {doc}")
```

This finds relevant documents even without exact keyword matches!

### Document Clustering

Group similar documents together:

```python
from sklearn.cluster import KMeans

# Embed a collection of documents
docs_emb = model.encode(documents)

# Cluster into k groups
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(docs_emb)

print(f"Document clusters: {clusters}")
```

### Duplicate Detection

Find near-duplicate content by setting a similarity threshold:

```python
def find_duplicates(documents, model, threshold=0.9):
    """Find near-duplicate documents."""
    embeddings = model.encode(documents)
    similarities = cosine_similarity(embeddings)
    
    duplicates = []
    for i in range(len(documents)):
        for j in range(i+1, len(documents)):
            if similarities[i][j] > threshold:
                duplicates.append((i, j, similarities[i][j]))
    
    return duplicates
```

## 7.9: Code Embeddings: Programs as Vectors

Here's where things get really interesting: the same principles apply to source code! We can embed functions, methods, or entire programs as vectors.

### Why Code Embeddings?

Code has semantic meaning just like natural language:

- Two functions might solve the same problem differently
- Comments and variable names carry intent
- Code structure reveals algorithmic approach

```python
# These functions do the same thing:
def sum_list_v1(numbers):
    total = 0
    for n in numbers:
        total += n
    return total

def sum_list_v2(lst):
    return sum(lst)

# A good embedding should recognize their similarity!
```

### Code Embedding Models

Modern code embeddings come from models trained on millions of repositories:

- **CodeBERT**: BERT trained on code and comments
- **GraphCodeBERT**: Incorporates code structure
- **UniXcoder**: Unified model for multiple languages

```python
# Using a code embedding model
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

def embed_code(code_snippet):
    """Generate embedding for code snippet."""
    inputs = tokenizer(code_snippet, return_tensors="pt", 
                      truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use [CLS] token embedding as code representation
    return outputs.last_hidden_state[:, 0, :].numpy()

# Compare our two sum functions
emb1 = embed_code("""
def sum_list(numbers):
    total = 0
    for n in numbers:
        total += n
    return total
""")

emb2 = embed_code("""
def sum_list(lst):
    return sum(lst)
""")

similarity = cosine_similarity(emb1, emb2)[0][0]
print(f"Code similarity: {similarity:.3f}")
```

The embeddings recognize these as similar despite different implementations!

## 7.10: Applications of Code Embeddings

### Code Search

Find relevant code examples from natural language queries:

```python
def code_search(query, code_snippets, model, tokenizer):
    """Search code using natural language."""
    # Embed the query as text
    query_emb = embed_code(query)
    
    # Embed all code snippets
    code_embs = [embed_code(snippet) for snippet in code_snippets]
    
    # Find most similar
    similarities = [cosine_similarity(query_emb, emb)[0][0] 
                   for emb in code_embs]
    
    ranked = sorted(zip(code_snippets, similarities),
                   key=lambda x: x[1], reverse=True)
    return ranked

# Example: search for "function that sorts a list"
query = "sort a list of numbers"
results = code_search(query, my_code_snippets, model, tokenizer)
```

### Clone Detection

Find duplicate or near-duplicate code:

```python
def find_code_clones(functions, threshold=0.85):
    """Detect code clones using embeddings."""
    embeddings = [embed_code(func) for func in functions]
    
    clones = []
    for i in range(len(functions)):
        for j in range(i+1, len(functions)):
            sim = cosine_similarity(embeddings[i], embeddings[j])[0][0]
            if sim > threshold:
                clones.append((i, j, sim))
    
    return clones
```

This is incredibly useful for refactoring and understanding codebases!

### Bug Detection

Similar code might have similar bugs. If we find a bug in one function, we can search for similar functions that might have the same issue:

```python
def find_similar_functions(buggy_func, codebase):
    """Find functions similar to a buggy one."""
    buggy_emb = embed_code(buggy_func)
    
    similar = []
    for func in codebase:
        func_emb = embed_code(func)
        sim = cosine_similarity(buggy_emb, func_emb)[0][0]
        if sim > 0.7:  # High similarity threshold
            similar.append((func, sim))
    
    return sorted(similar, key=lambda x: x[1], reverse=True)
```

## 7.11: Cross-Modal Embeddings: Code and Text Together

The most powerful models learn a **shared embedding space** where both code and natural language descriptions live. This enables:

- Generating code from natural language
- Generating documentation from code
- Matching functions to descriptions

```python
# Conceptual example of cross-modal search
def cross_modal_search(description, code_functions):
    """Find code matching natural language description."""
    # Embed the description as text
    desc_emb = embed_text(description)
    
    # Embed code as code
    code_embs = [embed_code(func) for func in code_functions]
    
    # They live in the same space, so we can compare!
    similarities = [cosine_similarity(desc_emb, emb)[0][0] 
                   for emb in code_embs]
    
    best_idx = np.argmax(similarities)
    return code_functions[best_idx]

# "find function that reverses a string"
description = "reverse the order of characters in a string"
matching_code = cross_modal_search(description, my_functions)
```

This is the technology behind tools like GitHub Copilot!

## 7.12: Vector Databases and Efficient Search

When working with millions of embeddings, naive similarity search is too slow. We need **approximate nearest neighbor** (ANN) search.

### Vector Databases

Specialized databases store and search embeddings efficiently:

```python
import chromadb

# Create a vector database
client = chromadb.Client()
collection = client.create_collection("code_snippets")

# Add code with embeddings
collection.add(
    documents=code_snippets,
    embeddings=code_embeddings,
    ids=[f"code_{i}" for i in range(len(code_snippets))]
)

# Fast similarity search
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

print(f"Top 5 similar code snippets: {results['documents']}")
```

Popular vector databases include:
- **Pinecone**: Cloud-native, managed service
- **Weaviate**: Open-source with GraphQL API
- **Chroma**: Simple, embedded database
- **FAISS**: Facebook's library for efficient similarity search

## 7.13: The Geometry of Meaning: Going Deeper

Let's explore what's actually happening in embedding space with some geometric intuition.

### Subspaces and Concepts

Different regions and directions in embedding space correspond to semantic concepts. There's roughly a "gender" subspace, an "animal vs. object" subspace, etc.

```python
def explore_dimensions(embeddings_dict, word_pairs):
    """Explore semantic dimensions."""
    for w1, w2 in word_pairs:
        diff = embeddings_dict[w2] - embeddings_dict[w1]
        print(f"\n{w1} → {w2} direction:")
        
        # Apply this direction to other words
        for word, vec in embeddings_dict.items():
            if word not in [w1, w2]:
                projection = np.dot(diff, vec) / np.linalg.norm(diff)
                print(f"  {word}: {projection:.3f}")

# Explore gender dimension
pairs = [('king', 'queen'), ('man', 'woman')]
explore_dimensions(embeddings, pairs)
```

### Clustering Structure

Embeddings naturally form clusters of related concepts:

```python
from sklearn.decomposition import PCA

def visualize_embeddings(embeddings_dict):
    """Reduce to 2D and visualize clusters."""
    words = list(embeddings_dict.keys())
    vectors = np.array(list(embeddings_dict.values()))
    
    # Reduce dimensions for visualization
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.5)
    
    for i, word in enumerate(words):
        plt.annotate(word, coords[i], fontsize=8)
    
    plt.title('Word Embeddings Reduced to 2D')
    plt.show()

visualize_embeddings(large_embeddings_dict)
```

Related words cluster together naturally!

## 7.14: Limitations and Biases

Embeddings are powerful but not perfect. They inherit biases from their training data:

### Social Biases

```python
# Embeddings can encode problematic associations
def measure_bias(embeddings, target_word, group1, group2):
    """Measure association bias."""
    target = embeddings[target_word]
    
    g1_sims = [cosine_similarity(target, embeddings[w]) for w in group1]
    g2_sims = [cosine_similarity(target, embeddings[w]) for w in group2]
    
    return np.mean(g1_sims) - np.mean(g2_sims)

# Occupational bias example
professions = ['doctor', 'engineer', 'nurse', 'teacher']
male_words = ['man', 'he', 'male']
female_words = ['woman', 'she', 'female']

for job in professions:
    bias = measure_bias(embeddings, job, male_words, female_words)
    print(f"{job}: {bias:.3f} {'(male-associated)' if bias > 0 else '(female-associated)'}")
```

These biases reflect societal stereotypes in the training data. Researchers are actively working on debiasing techniques.

### Context Limitations

Static embeddings give each word one vector, ignoring context:

```python
# "bank" means different things here:
sentences = [
    "I deposited money at the bank",  # financial institution
    "We sat on the river bank"        # land beside water
]

# Static embeddings give "bank" the same vector in both!
# Contextual embeddings (like BERT) solve this.
```

## 7.15: Contextual Embeddings: The Transformer Revolution

Modern embeddings are **contextual**—the same word gets different vectors depending on context.

### BERT-style Models

```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_contextual_embedding(sentence, target_word):
    """Get context-aware embedding for a word."""
    # Tokenize with target word marked
    inputs = tokenizer(sentence, return_tensors='pt')
    
    # Get all token embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Find target word's position and return its embedding
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    target_idx = tokens.index(target_word)
    
    return outputs.last_hidden_state[0, target_idx, :].numpy()

# "bank" gets different embeddings based on context
emb1 = get_contextual_embedding("I went to the bank", "bank")
emb2 = get_contextual_embedding("The river bank was muddy", "bank")

similarity = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))
print(f"Same word, different contexts: {similarity[0][0]:.3f}")
```

The embeddings are quite different, capturing the different meanings!

## 7.16: Building Your Own Embedding Model

Let's build a simple embedding model from scratch to understand the mechanics:

```python
import torch.nn as nn
import torch.optim as optim

class SimpleEmbedding(nn.Module):
    """Simple word embedding model."""
    
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, word_ids):
        return self.embeddings(word_ids)

# Training loop (simplified skip-gram)
def train_embeddings(training_pairs, vocab_size, embedding_dim=50, epochs=10):
    model = SimpleEmbedding(vocab_size, embedding_dim)
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        total_loss = 0
        
        for center, context in training_pairs:
            # Get embeddings
            center_emb = model(torch.tensor([center]))
            context_emb = model(torch.tensor([context]))
            
            # Simple objective: maximize similarity
            loss = -torch.cosine_similarity(center_emb, context_emb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    
    return model

# Train on your corpus!
```

Real models use negative sampling and other tricks, but this captures the core idea.

## 7.17: Advanced Topics: Multimodal Embeddings

The frontier of embedding research combines multiple modalities—text, code, images, and more—into a single unified space.

### CLIP: Images and Text Together

```python
# CLIP can embed both images and text
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Find image matching text description
images = [image1, image2, image3]  # PIL images
text = ["a cat", "a dog", "a car"]

inputs = processor(text=text, images=images, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    
# Similarity between each text and each image
logits = outputs.logits_per_text
print(f"Text-image similarities:\n{logits}")
```

This enables powerful cross-modal search and understanding!

## 7.18: Practical Considerations

### Choosing Embedding Dimensionality

Higher dimensions capture more information but require more computation:

```python
dimensions = [50, 100, 300, 768]
for dim in dimensions:
    storage_per_word = dim * 4  # 4 bytes per float32
    storage_1m_words = storage_per_word * 1_000_000 / (1024**2)
    print(f"{dim}D: {storage_1m_words:.1f} MB for 1M words")
```

Typical choices:
- **50-100D**: Fast, lightweight applications
- **300-384D**: Good balance for most tasks
- **768-1024D**: Maximum quality for critical applications

### Normalization

Always normalize embeddings for cosine similarity:

```python
def normalize_embedding(vec):
    """L2 normalize vector to unit length."""
    return vec / np.linalg.norm(vec)

# After normalization, cosine similarity = dot product!
norm_emb1 = normalize_embedding(emb1)
norm_emb2 = normalize_embedding(emb2)

similarity = np.dot(norm_emb1, norm_emb2)  # Simpler!
```

## 7.19: The Future of Embeddings

Embeddings continue to evolve rapidly:

1. **Sparse Embeddings**: Combining dense vectors with sparse keyword signals
2. **Learned Retrievals**: End-to-end learning of retrieval systems
3. **Multilingual Embeddings**: Single space for all languages
4. **Dynamic Embeddings**: Embeddings that update based on recent context

```python
# Hypothetical future: embeddings with temporal context
def temporal_embedding(word, timestamp, context_window):
    """Embed word considering when it's used."""
    base_emb = get_embedding(word)
    temporal_shift = learn_temporal_shift(timestamp, context_window)
    return base_emb + temporal_shift
```

## Conclusion: The Geometry of Intelligence

Embeddings transform the abstract concept of "meaning" into concrete mathematics. By representing words, sentences, and code as vectors, we can compute with semantics—measuring similarity, finding patterns, and enabling machines to understand the relationships between ideas.

The journey from bag-of-words to modern contextual embeddings mirrors the evolution of AI itself: from brittle, hand-crafted rules to flexible, learned representations that capture the richness of human language and thought.

As you continue your studies, remember: every time you use a search engine, a recommendation system, or an AI assistant, embeddings are working behind the scenes, measuring distances in a high-dimensional space that encodes human knowledge. Pretty amazing when you think about it!

### Key Takeaways

- **Embeddings map discrete symbols to continuous vectors** that encode semantic meaning
- **Cosine similarity** measures semantic similarity between embeddings
- **Word2Vec** learns embeddings by predicting words from context
- **Sentence embeddings** extend these ideas to longer text
- **Code embeddings** apply the same principles to programs
- **Vector databases** enable efficient similarity search at scale
- **Contextual embeddings** capture meaning that depends on context
- **Biases in training data** can be reflected in embeddings

### Further Exploration

1. Try different embedding models on your own text or code
2. Build a semantic search system for your favorite domain
3. Visualize embeddings to understand their structure
4. Explore bias in embeddings and mitigation techniques
5. Experiment with cross-modal embeddings (text + images)

