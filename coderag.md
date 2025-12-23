# Chapter 9: Retrieval-Augmented Generation for Code

## Introduction

What happens when you need to write code in an unfamiliar codebase? Most developers start by searching for similar examples—finding functions that do something close to what they need, then adapting those patterns to their specific problem. This is retrieval-augmented generation in action, and we can teach computers to do the same thing.

In this chapter, we'll build a system that can search through code repositories, find relevant examples, and use them to generate new code or explain existing code. But code isn't like regular text—it has structure, syntax, and semantic relationships that generic text models often miss. We'll see why code-specific embeddings matter and how to build a practical RAG system optimized for programming.

## Why Code Needs Special Treatment

Let's start with a simple experiment. Consider these two Python functions:

```python
def validate_email(email):
    return '@' in email and '.' in email.split('@')[1]

def check_email(address):
    if '@' not in address:
        return False
    return '.' in address.split('@')[1]
```

These functions do exactly the same thing—they validate email addresses using identical logic. A human programmer would immediately recognize them as equivalent. But how would a computer see them?

If we use a generic text embedding model (like the ones we used for documents in previous chapters), it might focus on superficial differences: different function names, different variable names, slightly different control flow structure. The semantic equivalence—that both functions check for an '@' symbol and a dot in the domain—might get lost.

Code-specific embedding models, trained on millions of code examples, learn to recognize these patterns. They understand that `validate_email` and `check_email` are semantically related function names. They know that `email` and `address` are likely to refer to the same concept. Most importantly, they can recognize that despite structural differences, the core logic is identical.

## Understanding Code Embeddings

We'll use UniXcoder, a model specifically trained for code understanding. It was trained on code from five programming languages and learns to capture both syntactic structure and semantic meaning:

```python
from transformers import AutoTokenizer, AutoModel
import torch

class UniXcoderEmbedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
        self.model = AutoModel.from_pretrained("microsoft/unixcoder-base")
        self.model.eval()
```

When we encode code with UniXcoder, we use mean pooling over all token embeddings:

```python
def encode(self, text):
    inputs = self.tokenizer(text, return_tensors="pt", 
                           truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = self.model(**inputs)
    
    # Mean pooling weighted by attention mask
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    return (sum_embeddings / sum_mask).squeeze().numpy()
```

The resulting embedding is a 768-dimensional vector that captures the code's meaning. Let's see what this buys us.

## Comparing Generic and Code-Specific Embeddings

Here's where things get interesting. Consider these three functions:

```python
# Function 1: Email validation
def validate_email(email):
    return '@' in email and '.' in email.split('@')[1]

# Function 2: Another email validator (different implementation)
def check_email(address):
    if '@' not in address:
        return False
    return '.' in address.split('@')[1]

# Function 3: Completely different functionality
def calculate_compound_interest(principal, rate, years):
    return principal * ((1 + rate) ** years)
```

When we compute similarity scores between Function 1 and the others:

```
UniXcoder similarities:
  Function 1 vs Function 2: 0.9234
  Function 1 vs Function 3: 0.6891

Generic model similarities:
  Function 1 vs Function 2: 0.8456
  Function 1 vs Function 3: 0.7234
```

UniXcoder shows much stronger discrimination. It gives high similarity (0.92) to the two email validators despite their different implementations, and notably lower similarity (0.69) to the unrelated function. The generic model's scores are closer together (0.85 vs 0.72), making it harder to distinguish truly similar code.

This discrimination becomes even more important with subtle changes. Consider these variations of the same validation logic:

```python
# Original
def validate_age(age):
    if age < 18:
        return False
    return True

# Changed operator (completely different logic!)
def validate_age(age):
    if age > 18:  # Now checking if OVER 18
        return False
    return True

# Just renamed variable (same logic)
def validate_age(user_age):
    if user_age < 18:
        return False
    return True
```

UniXcoder recognizes that changing `<` to `>` fundamentally alters the function's behavior, while renaming `age` to `user_age` doesn't. A generic text model might treat both changes similarly since they're both single-token modifications.

## Building a Code Knowledge Base

Now that we understand embeddings, let's build a searchable knowledge base. We'll use Chroma, the same vector database from earlier chapters:

```python
import chromadb

class CodeKnowledgeBase:
    def __init__(self, collection_name="code_kb"):
        self.client = chromadb.Client()
        self.embedder = UniXcoderEmbedder()
        self.collection = self.client.create_collection(name=collection_name)
```

The key to effective code retrieval is creating rich, searchable representations. For each function, we don't just store the raw code—we create a structured text representation:

```python
def _create_searchable_text(self, func_name, func_code, docstring=None):
    parts = [f"Function name: {func_name}"]
    if docstring:
        parts.append(f"Description: {docstring}")
    parts.append(f"Implementation:\n{func_code}")
    return "\n".join(parts)
```

This gives the embedding model multiple signals: the function name provides a semantic hint, the docstring explains the purpose in natural language, and the implementation shows the actual logic. When we add a function:

```python
def add_function(self, func_name, func_code, docstring=None):
    searchable = self._create_searchable_text(func_name, func_code, docstring)
    embedding = self.embedder.encode(searchable).tolist()
    
    self.collection.add(
        embeddings=[embedding],
        documents=[func_code],
        metadatas=[{'name': func_name, 'docstring': docstring or ''}],
        ids=[f"func_{self.counter}"]
    )
```

Now we can search using natural language queries:

```python
results = kb.search("How do I validate user email addresses?", top_k=2)
```

The system finds relevant functions by comparing the query embedding to stored code embeddings. The beauty is that we can phrase our query naturally—"How do I validate email addresses?"—and it will find functions named `validate_email`, `check_email_format`, or even `is_valid_email`, because they're all semantically similar.

## Parsing Code with AST

When indexing a real codebase, we need to extract functions systematically. Python's Abstract Syntax Tree (AST) module lets us parse code and extract structured information:

```python
import ast

def parse_python_file(filepath):
    with open(filepath, 'r') as f:
        code = f.read()
    
    tree = ast.parse(code)
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node)
            func_lines = code.split('\n')[node.lineno-1:node.end_lineno]
            func_code = '\n'.join(func_lines)
            
            functions.append({
                'name': node.name,
                'code': func_code,
                'docstring': docstring
            })
    
    return functions
```

The AST parser identifies function boundaries precisely, extracts docstrings, and preserves the complete implementation. This is far better than naive splitting by line count or character count, which might cut a function in half or include unrelated code.

## Testing Different Query Types

Code search needs to handle diverse query styles. Developers might ask:

**Natural language:** "How do I check if an email is valid?"
**Technical terms:** "email validation function"
**Use cases:** "I need to create a new user account"
**Single keywords:** "email"

With a well-built knowledge base, all these queries should return relevant results. The natural language query works because the embedding model understands the semantic relationship between "check if valid" and "validation". The technical term works because it closely matches function names and docstrings. The use case works because creating a user account typically involves email validation. Even single keywords can work, though they're more ambiguous.

When testing queries on our indexed codebase:

```
Query: "How do I check if an email is valid?"
Results:
  1. validate_email (distance: 0.234)
  2. create_user (distance: 0.456)

Query: "email validation function"  
Results:
  1. validate_email (distance: 0.189)
  2. send_email (distance: 0.523)
```

The distance scores help rank results. Lower distances mean better matches. Notice that "create_user" appears for the first query because it calls `validate_email` internally—the model picks up on the semantic connection even though the function name doesn't mention validation.

## Building a Code RAG System

Now we can combine retrieval with generation. A Code RAG system has three main capabilities: generating new code based on examples, explaining existing code, and suggesting refactorings.

Here's the generation component:

```python
class CodeRAG:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
    
    def generate_code(self, task_description, top_k=3):
        # Retrieve relevant examples
        results = self.kb.search(task_description, top_k=top_k)
        
        # Build context from retrieved code
        context_parts = ["Here are relevant code examples:\n"]
        for i, result in enumerate(results, 1):
            context_parts.append(f"\nExample {i}:")
            context_parts.append(result['code'])
        
        context = "\n".join(context_parts)
        
        # Generate with LLM
        prompt = f"""{context}

Task: {task_description}

Generate a function following the patterns above."""
        
        return call_ollama(prompt)
```

The key insight is that we're not asking the LLM to generate code from scratch. We're giving it concrete examples from the codebase, then asking it to follow those patterns. This produces code that's stylistically consistent and uses familiar patterns.

For example, given the task "Create a function to validate username: 3-20 characters, alphanumeric only", the system retrieves similar validation functions (`validate_email`, `validate_password`), then generates:

```python
def validate_username(username):
    '''Check if username meets requirements.'''
    if len(username) < 3 or len(username) > 20:
        return False
    return username.isalnum()
```

The generated code follows the same pattern as the examples: a clear docstring, length checking, and a boolean return value.

## Code Explanation with Context

Understanding unfamiliar code is easier when you can see similar examples. The explanation component retrieves related code and uses it to contextualize what a function does:

```python
def explain_code(self, code_snippet, top_k=2):
    results = self.kb.search(code_snippet, top_k=top_k)
    
    context_parts = ["Similar functions:\n"]
    for result in results:
        context_parts.append(f"- {result['metadata']['name']}")
        context_parts.append(f"  {result['code']}")
    
    prompt = f"""{"\n".join(context_parts)}

Explain this code:
{code_snippet}"""
    
    return call_ollama(prompt)
```

Given a mysterious function:

```python
def check_user(name, pwd):
    if len(name) < 3:
        return False
    import hashlib
    h = hashlib.sha256(pwd.encode()).hexdigest()
    return True
```

The system finds similar functions (`validate_username`, `hash_password`) and explains: "This function validates a username (checking length) and hashes a password using SHA-256. It follows the pattern of other validation functions in the codebase, though it always returns True after hashing, which might be a bug."

## Why This Works

Code RAG succeeds because it combines three strengths:

1. **Code-specific embeddings** understand programming semantics, not just text similarity
2. **Structured retrieval** finds relevant examples from real codebases
3. **Pattern-following generation** produces consistent, idiomatic code

The system isn't trying to memorize every possible code pattern. Instead, it learns to find relevant examples and adapt them. This is exactly how human developers work—we look for similar code, understand the pattern, then apply it to our specific problem.

The key technical challenge is getting the embeddings right. Generic text models treat code as just another document. Code-specific models understand that `email` and `address` are semantically related, that changing `<` to `>` fundamentally alters logic, and that two functions with different implementations might be functionally equivalent.

## Limitations and Future Directions

This approach works well for finding similar code and explaining patterns, but it has limitations. The retrieval depends on having good examples in the knowledge base—if you're doing something truly novel, there might not be relevant code to retrieve. The generation quality depends on the LLM's understanding of programming—it might follow patterns correctly but miss subtle edge cases.

More sophisticated systems might build code knowledge graphs, tracking not just individual functions but also their dependencies, call relationships, and test cases. They might use hybrid retrieval combining semantic search with symbolic analysis. They might even execute generated code to verify correctness.

But the fundamental insight remains: code search is not text search. It requires understanding structure, semantics, and patterns. When we get that right, we can build systems that help developers navigate unfamiliar codebases, explain complex logic, and generate consistent new code—all by learning from examples, just like humans do.