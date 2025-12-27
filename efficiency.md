# Chapter 7: Performance, Efficiency, and Context Management in Ollama

> "In computing, efficiency isn't just about speed—it's about making intelligent trade-offs between resources, accuracy, and capability." — Your enthusiastic professor who once optimized a Python script only to discover it was already fast enough 🐌→🚀

## 7.1 Introduction: Why Performance Matters

Welcome to one of my favorite topics! When I first started working with large language models, I ran a 7B parameter model on my laptop and watched it churn through tokens at a glacial pace. That experience taught me something crucial: **performance isn't just a nice-to-have—it fundamentally shapes what's possible**.

In this chapter, we'll explore how different Ollama models perform across three critical dimensions:
- **Time complexity**: How fast do models generate responses?
- **Memory efficiency**: What are the hardware requirements?
- **Context management**: How do we handle the finite attention budget?

But first, a nerdy joke: Why did the programmer quit his job? Because he didn't get arrays! 😄 (Okay, I promise the rest will be better.)

## 7.2 Understanding Model Performance Fundamentals

### 7.2.1 The Iron Triangle of LLM Performance

Every model exists within a three-way trade-off:

```
        Capability
           /\
          /  \
         /    \
        /      \
       /________\
    Speed      Memory
```

Let's quantify this with actual Ollama models:

```python
import ollama
import time
import psutil
import matplotlib.pyplot as plt
import numpy as np

def measure_model_performance(model_name, prompt, num_runs=3):
    """
    Measure time and memory for a given model.
    """
    times = []
    
    for _ in range(num_runs):
        # Get initial memory
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start = time.time()
        response = ollama.generate(model=model_name, prompt=prompt)
        end = time.time()
        
        mem_after = process.memory_info().rss / 1024 / 1024
        times.append(end - start)
    
    avg_time = np.mean(times)
    return {
        'model': model_name,
        'avg_time': avg_time,
        'tokens_generated': len(response['response'].split()),
        'tokens_per_second': len(response['response'].split()) / avg_time
    }

# Test different models
models = ['llama3.2:1b', 'llama3.2:3b', 'llama3.2:7b']
prompt = "Explain recursion in one paragraph."

results = [measure_model_performance(m, prompt) for m in models]
```

**Key Insight**: Smaller models are faster but less capable. There's no free lunch! 🍔

### 7.2.2 Memory Scaling Laws

Model memory requirements follow a predictable pattern. Here's a rough formula:

```
Memory (GB) ≈ Parameters × Bits_per_Parameter / 8 billion
```

For quantized models (the default in Ollama):

```python
def estimate_memory_gb(params_billions, quantization_bits=4):
    """Estimate model memory requirements."""
    return params_billions * quantization_bits / 8

# Examples
models_memory = {
    '1B (4-bit)': estimate_memory_gb(1, 4),
    '3B (4-bit)': estimate_memory_gb(3, 4),
    '7B (4-bit)': estimate_memory_gb(7, 4),
    '13B (4-bit)': estimate_memory_gb(13, 4),
}

for model, mem in models_memory.items():
    print(f"{model}: ~{mem:.1f} GB")
```

**Output:**
```
1B (4-bit): ~0.5 GB
3B (4-bit): ~1.5 GB
7B (4-bit): ~3.5 GB
13B (4-bit): ~6.5 GB
```

## 7.3 Benchmarking Ollama Models

Let's conduct a systematic performance comparison:

```python
import pandas as pd

def comprehensive_benchmark(models, prompts):
    """Run comprehensive benchmarks across models."""
    results = []
    
    for model in models:
        print(f"Testing {model}...")
        for prompt_type, prompt in prompts.items():
            try:
                start = time.time()
                response = ollama.generate(
                    model=model,
                    prompt=prompt,
                    options={'num_predict': 100}  # Limit output
                )
                elapsed = time.time() - start
                
                results.append({
                    'model': model,
                    'task': prompt_type,
                    'time_seconds': elapsed,
                    'output_tokens': len(response['response'].split()),
                    'tokens_per_sec': len(response['response'].split()) / elapsed
                })
            except Exception as e:
                print(f"Error with {model}: {e}")
    
    return pd.DataFrame(results)

# Test prompts of varying complexity
test_prompts = {
    'simple': "What is 2+2?",
    'reasoning': "If a train leaves Chicago at 60mph and another leaves NYC at 80mph, 800 miles apart, when do they meet?",
    'creative': "Write a haiku about artificial intelligence.",
    'code': "Write a Python function to compute Fibonacci numbers."
}

models = ['llama3.2:1b', 'llama3.2:3b', 'mistral:7b']
df = comprehensive_benchmark(models, test_prompts)

# Visualize
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='task', y='tokens_per_sec', hue='model')
plt.title('Token Generation Speed by Model and Task')
plt.ylabel('Tokens/Second')
plt.xticks(rotation=45)
plt.legend(title='Model')
plt.tight_layout()
plt.show()
```

### 7.3.1 Interpreting the Results

What you'll typically observe:

1. **Smaller models → faster tokens/sec** (but potentially lower quality)
2. **Complex reasoning → slower generation** (more computation per token)
3. **Quantized models → slight quality trade-off for 4x memory savings**

Here's a pro tip from my own experiments: For production systems, I usually run 3B models for routine tasks and only invoke 7B+ models when complexity demands it. Think of it like having both a bicycle and a car—use the right tool for the journey! 🚲🚗

## 7.4 Context Management: The Attention Budget Problem

Now we arrive at one of the most crucial concepts in modern AI engineering. Let me share something that surprised me when I first learned about it:

> **Context isn't unlimited, and more isn't always better.**

### 7.4.1 Understanding Context Windows

Every LLM has a maximum context window (e.g., 8K, 32K, or 128K tokens). But here's the catch: performance degrades as you approach that limit. This phenomenon is called **context rot**.

```python
def demonstrate_context_window(model='llama3.2:3b'):
    """Show how context size affects performance."""
    
    # Create contexts of varying lengths
    base_context = "You are a helpful assistant. "
    test_contexts = {
        'small': base_context + "Answer briefly.",
        'medium': base_context + ("Background info. " * 100),
        'large': base_context + ("Background info. " * 500),
    }
    
    prompt = "What is machine learning?"
    
    for size, context in test_contexts.items():
        start = time.time()
        response = ollama.generate(
            model=model,
            prompt=context + prompt,
            options={'num_predict': 50}
        )
        elapsed = time.time() - start
        
        print(f"{size.upper()} context ({len(context.split())} words):")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  First 100 chars: {response['response'][:100]}...\n")

demonstrate_context_window()
```

### 7.4.2 The Science Behind Context Rot

Why does performance degrade? Two main factors:

1. **Quadratic Attention Complexity**: For *n* tokens, the model must compute *n²* attention relationships
2. **Diluted Attention Budget**: Each token gets less "attention capacity"

Think of it like trying to remember every detail of a 500-page book versus a 50-page article. You'll recall the article better! 📚

```python
def visualize_attention_complexity():
    """Visualize quadratic scaling of attention."""
    context_lengths = np.array([1000, 2000, 4000, 8000, 16000])
    attention_ops = context_lengths ** 2
    
    plt.figure(figsize=(10, 6))
    plt.plot(context_lengths, attention_ops / 1e6, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Context Length (tokens)')
    plt.ylabel('Attention Operations (millions)')
    plt.title('Quadratic Scaling of Attention Mechanism')
    plt.grid(True, alpha=0.3)
    
    # Annotate key points
    for x, y in zip(context_lengths, attention_ops / 1e6):
        plt.annotate(f'{y:.0f}M', 
                    xy=(x, y), 
                    xytext=(10, 10),
                    textcoords='offset points')
    
    plt.tight_layout()
    plt.show()

visualize_attention_complexity()
```

## 7.5 Effective Context Engineering Strategies

Based on the techniques described in Anthropic's research, here are practical strategies for Ollama:

### 7.5.1 Strategy 1: Context Compaction

**The Idea**: Periodically summarize and compress conversation history.

```python
class ContextManager:
    def __init__(self, model='llama3.2:3b', max_tokens=4000):
        self.model = model
        self.max_tokens = max_tokens
        self.conversation = []
    
    def add_message(self, role, content):
        """Add message to conversation."""
        self.conversation.append({'role': role, 'content': content})
        
        # Check if compaction needed
        if self._estimate_tokens() > self.max_tokens:
            self._compact()
    
    def _estimate_tokens(self):
        """Rough token estimate (1 token ≈ 0.75 words)."""
        total_words = sum(len(msg['content'].split()) 
                         for msg in self.conversation)
        return int(total_words / 0.75)
    
    def _compact(self):
        """Summarize older messages."""
        print("🗜️ Compacting context...")
        
        # Keep system message and last 2 exchanges
        system_msg = self.conversation[0] if self.conversation[0]['role'] == 'system' else None
        recent = self.conversation[-4:]
        to_summarize = self.conversation[1:-4] if system_msg else self.conversation[:-4]
        
        if not to_summarize:
            return
        
        # Generate summary
        summary_prompt = f"Summarize this conversation concisely:\n\n"
        for msg in to_summarize:
            summary_prompt += f"{msg['role']}: {msg['content']}\n"
        
        response = ollama.generate(model=self.model, prompt=summary_prompt)
        summary = response['response']
        
        # Rebuild conversation
        new_conv = []
        if system_msg:
            new_conv.append(system_msg)
        new_conv.append({'role': 'assistant', 'content': f"[Summary]: {summary}"})
        new_conv.extend(recent)
        
        self.conversation = new_conv
        print(f"✅ Compacted to {len(self.conversation)} messages")
    
    def generate_response(self, user_message):
        """Generate response with context management."""
        self.add_message('user', user_message)
        
        # Build prompt from conversation
        prompt = "\n\n".join(f"{m['role']}: {m['content']}" 
                            for m in self.conversation)
        
        response = ollama.generate(model=self.model, prompt=prompt)
        assistant_msg = response['response']
        
        self.add_message('assistant', assistant_msg)
        return assistant_msg

# Example usage
manager = ContextManager()
manager.add_message('system', 'You are a helpful coding assistant.')

for i in range(10):
    response = manager.generate_response(f"Tell me about Python feature #{i+1}")
    print(f"\nQ{i+1}: {response[:100]}...")
```

### 7.5.2 Strategy 2: Structured Note-Taking

**The Idea**: Maintain a separate "memory" file that persists key information.

```python
import json
from pathlib import Path

class MemoryManager:
    def __init__(self, model='llama3.2:3b', memory_file='memory.json'):
        self.model = model
        self.memory_file = Path(memory_file)
        self.memory = self._load_memory()
    
    def _load_memory(self):
        """Load memory from disk."""
        if self.memory_file.exists():
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        return {'facts': [], 'preferences': {}, 'history': []}
    
    def _save_memory(self):
        """Persist memory to disk."""
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def extract_and_store(self, conversation):
        """Extract key facts from conversation."""
        prompt = f"""From this conversation, extract:
1. Important facts (as a list)
2. User preferences
3. Key decisions

Conversation:
{conversation}

Respond in JSON format:
{{"facts": [...], "preferences": {{...}}, "decisions": [...]}}"""
        
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            format='json'  # Request JSON output
        )
        
        try:
            extracted = json.loads(response['response'])
            self.memory['facts'].extend(extracted.get('facts', []))
            self.memory['preferences'].update(extracted.get('preferences', {}))
            self.memory['history'].append(extracted.get('decisions', []))
            self._save_memory()
            print(f"💾 Stored {len(extracted.get('facts', []))} new facts")
        except json.JSONDecodeError:
            print("⚠️ Could not parse extracted information")
    
    def recall(self, query):
        """Retrieve relevant memories."""
        # Simple keyword matching (could use embeddings for production)
        query_words = set(query.lower().split())
        relevant = []
        
        for fact in self.memory['facts']:
            fact_words = set(fact.lower().split())
            if query_words & fact_words:  # Intersection
                relevant.append(fact)
        
        return relevant[:3]  # Top 3

# Usage example
memory = MemoryManager()

# After a conversation...
conversation = """
User: I'm building a web scraper for research.
Assistant: Great! I recommend using BeautifulSoup and requests.
User: I prefer async programming.
Assistant: Then use aiohttp instead!
"""

memory.extract_and_store(conversation)

# Later...
recalled = memory.recall("web scraping recommendations")
print("📝 Recalled:", recalled)
```

### 7.5.3 Strategy 3: Just-in-Time Context Loading

**The Idea**: Load information only when needed, not upfront.

```python
class JITContextManager:
    """Just-In-Time context loading."""
    
    def __init__(self, model='llama3.2:3b', docs_dir='./documents'):
        self.model = model
        self.docs_dir = Path(docs_dir)
        self.doc_index = self._build_index()
    
    def _build_index(self):
        """Build lightweight index of available documents."""
        index = {}
        if not self.docs_dir.exists():
            return index
        
        for doc_path in self.docs_dir.glob('*.txt'):
            # Store metadata, not content
            index[doc_path.stem] = {
                'path': doc_path,
                'size': doc_path.stat().st_size,
                'modified': doc_path.stat().st_mtime
            }
        return index
    
    def search_docs(self, query):
        """Search for relevant documents."""
        # Simplified search (could use embeddings)
        query_terms = query.lower().split()
        matches = []
        
        for doc_name, meta in self.doc_index.items():
            if any(term in doc_name.lower() for term in query_terms):
                matches.append(doc_name)
        
        return matches
    
    def load_doc(self, doc_name):
        """Load document content only when needed."""
        if doc_name not in self.doc_index:
            return None
        
        path = self.doc_index[doc_name]['path']
        with open(path, 'r') as f:
            return f.read()
    
    def answer_with_jit(self, question):
        """Answer question, loading docs only as needed."""
        # First, see if we need documents
        initial_prompt = f"""Question: {question}
        
Available documents: {', '.join(self.doc_index.keys())}

Do you need any documents to answer? Reply with document names or 'none'."""
        
        response = ollama.generate(model=self.model, prompt=initial_prompt)
        needed_docs = response['response'].strip().lower()
        
        # Load only necessary documents
        context = ""
        if needed_docs != 'none':
            matches = self.search_docs(needed_docs)
            for doc in matches[:2]:  # Limit to 2 docs
                content = self.load_doc(doc)
                if content:
                    context += f"\n\n=== {doc} ===\n{content[:1000]}"  # First 1K chars
        
        # Generate final answer
        final_prompt = f"""{context}

Question: {question}

Answer:"""
        
        return ollama.generate(model=self.model, prompt=final_prompt)['response']

# Usage
jit = JITContextManager()
answer = jit.answer_with_jit("What are the best practices for Python error handling?")
print(answer)
```

## 7.6 Advanced: Multi-Model Orchestration

For complex tasks, use different models for different subtasks:

```python
class MultiModelOrchestrator:
    """Coordinate multiple models for optimal efficiency."""
    
    def __init__(self):
        self.fast_model = 'llama3.2:1b'  # Quick classification/routing
        self.balanced_model = 'llama3.2:3b'  # General purpose
        self.power_model = 'llama3.1:8b'  # Complex reasoning
    
    def route_query(self, query):
        """Determine which model to use."""
        classification_prompt = f"""Classify this query's complexity:
Query: {query}

Categories:
- simple: factual questions, greetings
- moderate: explanations, how-to questions  
- complex: multi-step reasoning, analysis

Respond with ONLY the category:"""
        
        response = ollama.generate(
            model=self.fast_model,
            prompt=classification_prompt
        )
        
        complexity = response['response'].strip().lower()
        
        model_map = {
            'simple': self.fast_model,
            'moderate': self.balanced_model,
            'complex': self.power_model
        }
        
        return model_map.get(complexity, self.balanced_model)
    
    def process(self, query):
        """Process query with optimal model."""
        selected_model = self.route_query(query)
        print(f"🎯 Routing to: {selected_model}")
        
        start = time.time()
        response = ollama.generate(model=selected_model, prompt=query)
        elapsed = time.time() - start
        
        return {
            'response': response['response'],
            'model': selected_model,
            'time': elapsed
        }

# Test it
orchestrator = MultiModelOrchestrator()

queries = [
    "What's 5+3?",
    "Explain how binary search works.",
    "Analyze the time complexity of this recursive algorithm and suggest optimizations."
]

for q in queries:
    result = orchestrator.process(q)
    print(f"\nQ: {q}")
    print(f"A: {result['response'][:100]}...")
    print(f"⏱️ {result['time']:.2f}s with {result['model']}\n")
```

## 7.7 Measuring and Monitoring Performance

Production systems need monitoring:

```python
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque

@dataclass
class PerfMetrics:
    timestamp: float
    latency_ms: float
    tokens_generated: int
    model: str

class PerformanceMonitor:
    def __init__(self, window_size=100):
        self.metrics: Deque[PerfMetrics] = deque(maxlen=window_size)
    
    def record(self, model, start_time, token_count):
        """Record a request's performance."""
        latency = (time.time() - start_time) * 1000  # Convert to ms
        metric = PerfMetrics(
            timestamp=time.time(),
            latency_ms=latency,
            tokens_generated=token_count,
            model=model
        )
        self.metrics.append(metric)
    
    def get_stats(self):
        """Calculate performance statistics."""
        if not self.metrics:
            return {}
        
        latencies = [m.latency_ms for m in self.metrics]
        tokens = [m.tokens_generated for m in self.metrics]
        
        return {
            'avg_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'avg_tokens': np.mean(tokens),
            'total_requests': len(self.metrics)
        }
    
    def plot_performance(self):
        """Visualize performance over time."""
        if not self.metrics:
            return
        
        timestamps = [m.timestamp - self.metrics[0].timestamp for m in self.metrics]
        latencies = [m.latency_ms for m in self.metrics]
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(timestamps, latencies, 'b-', alpha=0.6)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Latency (ms)')
        plt.title('Response Latency Over Time')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(latencies, bins=30, edgecolor='black')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')
        plt.title('Latency Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Usage
monitor = PerformanceMonitor()

# Simulate requests
for i in range(50):
    start = time.time()
    resp = ollama.generate(
        model='llama3.2:3b',
        prompt=f"Quick fact #{i+1}",
        options={'num_predict': 30}
    )
    monitor.record('llama3.2:3b', start, len(resp['response'].split()))

stats = monitor.get_stats()
print("\n📊 Performance Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value:.2f}")

monitor.plot_performance()
```

## 7.8 Best Practices and Production Tips

After running Ollama in production for several projects, here are my hard-won lessons:

### 7.8.1 The Dos

1. **Profile before optimizing**: Measure first, optimize second
2. **Use model quantization**: 4-bit quantized models are usually fine
3. **Implement caching**: Cache common queries
4. **Monitor context size**: Set alerts at 75% of max context
5. **Batch when possible**: Process multiple requests together

```python
def batch_generate(model, prompts, batch_size=5):
    """Process prompts in batches for efficiency."""
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        # Ollama doesn't support true batching yet,
        # but this pattern prepares for future optimization
        for prompt in batch:
            resp = ollama.generate(model=model, prompt=prompt)
            results.append(resp['response'])
    
    return results
```

### 7.8.2 The Don'ts 

1. **Don't exceed context limits**: You'll get errors or poor results
2. **Don't use large models for simple tasks**: Overkill is real
3. **Don't forget to handle errors**: Networks fail, models crash
4. **Don't ignore user experience**: Speed matters!

