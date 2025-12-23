# Chapter 9: Code Embeddings

## Introduction

What makes two pieces of code similar? A human programmer can immediately recognize that these two functions do the same thing:

```python
def validate_email(email):
    return '@' in email and '.' in email.split('@')[1]

def check_email(address):
    if '@' not in address:
        return False
    return '.' in address.split('@')[1]
```

They have different names, different variable names, different control flow—but identical logic. Both check for an '@' symbol and verify there's a dot in the domain part. A developer reading either function would understand they're email validators.

Can we teach a computer to recognize this kind of similarity? This is the problem of code embeddings: converting source code into numerical vectors that capture semantic meaning, not just textual similarity.

## Why Generic Text Embeddings Fall Short

Let's start with what happens when we use a generic text embedding model—the kind we might use for documents or web pages. These models are trained on natural language: books, articles, websites. They learn that "car" and "automobile" are similar, that "happy" relates to "joyful", and that "king" minus "man" plus "woman" approximately equals "queen".

But code isn't natural language. Consider these three snippets:

```python
# Snippet 1
def add_numbers(a, b):
    return a + b

# Snippet 2
def sum_values(x, y):
    return x + y

# Snippet 3
def multiply_numbers(a, b):
    return a * b
```

To a text embedding model, Snippets 1 and 3 might seem more similar—they share the same variable names (`a`, `b`) and have similar function name structures (`add_numbers`, `multiply_numbers`). But semantically, Snippets 1 and 2 are identical (just with different names), while Snippet 3 does something completely different.

A code-specific embedding model should recognize that addition and summation are the same operation, that variable names are largely cosmetic, but that changing `+` to `*` fundamentally alters the function's behavior.

## Introducing UniXcoder

UniXcoder is a transformer model trained specifically on source code from five programming languages: Python, Java, JavaScript, PHP, and Ruby. It learns patterns like "functions that return early when validation fails", "loops that accumulate values", and "error handling with try-catch blocks". Most importantly, it learns to distinguish between superficial differences (renaming variables) and semantic differences (changing operators).

Let's load the model:

```python
from transformers import AutoTokenizer, AutoModel
import torch

class UniXcoderEmbedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
        self.model = AutoModel.from_pretrained("microsoft/unixcoder-base")
        self.model.eval()
```

The tokenizer converts code into tokens that the model understands. Unlike word-based tokenizers for natural language, a code tokenizer recognizes programming constructs: operators, keywords, identifiers, literals.

## Generating Embeddings

To create an embedding, we tokenize the code and pass it through the model:

```python
def encode(self, text):
    inputs = self.tokenizer(text, return_tensors="pt", 
                           truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = self.model(**inputs)
```

The model outputs hidden states for every token in the input. But we want a single vector representing the entire code snippet. The standard approach is mean pooling: average all token embeddings, weighted by the attention mask (so padding tokens don't contribute):

```python
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    return (sum_embeddings / sum_mask).squeeze().numpy()
```

The result is a 768-dimensional vector. Each dimension captures some aspect of the code's meaning—perhaps one dimension activates for validation functions, another for mathematical operations, another for string manipulation.

## Measuring Similarity

With embeddings, we can compute similarity using cosine similarity: the dot product of normalized vectors. Values range from -1 (opposite) to 1 (identical):

```python
from sklearn.metrics.pairwise import cosine_similarity

code1 = "def validate_email(email): return '@' in email"
code2 = "def check_email(address): return '@' in address"

emb1 = embedder.encode(code1).reshape(1, -1)
emb2 = embedder.encode(code2).reshape(1, -1)

similarity = cosine_similarity(emb1, emb2)[0][0]
print(f"Similarity: {similarity:.4f}")
```

Output:
```
Similarity: 0.9234
```

High similarity (0.92) despite different names and variables. The model recognizes these are functionally equivalent.

## Comparing Code-Specific vs Generic Embeddings

Let's run a direct comparison. We'll use UniXcoder and a generic text embedding model (nomic-embed-text) on the same code:

```python
code_snippets = {
    'email_val_1': "def validate_email(email): return '@' in email",
    'email_val_2': "def check_email(address): return '@' in address", 
    'math_calc': "def compound_interest(p, r, y): return p * ((1 + r) ** y)"
}
```

Computing similarities:

```
Test 1: email_val_1 vs email_val_2 (similar functions)
  UniXcoder:      0.9234
  Generic model:  0.8456
  
Test 2: email_val_1 vs math_calc (different functions)
  UniXcoder:      0.6891
  Generic model:  0.7234
```

Notice the discrimination scores. UniXcoder shows a large gap between similar functions (0.92) and different functions (0.69)—a difference of 0.23. The generic model's gap is only 0.12. This stronger discrimination makes it much easier to find truly relevant code.

## Understanding Variable Names

Here's something subtle: UniXcoder doesn't just ignore variable names as cosmetic details. It understands they carry semantic meaning. Consider these variations:

```python
# Original
def check(x):
    return x < 10

# Short rename
def check(y):
    return y < 10
    
# Descriptive rename
def check(age):
    return age < 10
```

Computing similarities to the original:

```
Short rename (x→y):         0.9891
Descriptive rename (x→age): 0.9456
```

The short rename (x to y) barely changes the embedding—both are generic mathematical variables. But renaming to `age` noticeably reduces similarity. Why? Because `age` adds semantic context. The function isn't just checking if a number is less than 10; it's checking if an age meets some threshold. That's meaningful information that changes how we understand the code.

This is actually desirable behavior. If you're searching for "age validation functions", you want functions that use the variable name `age` to rank higher than functions using `x`. The variable name is a signal about the function's purpose.

## Operator Sensitivity

The most critical test: does the model recognize when operators change the logic? Consider this function:

```python
def validate_age(age):
    if age < 18:
        return False
    return True
```

Now we make progressively more changes:

```python
# One operator change: < becomes >
def validate_age(age):
    if age > 18:  # Completely different logic!
        return False
    return True

# Two changes: > and swap returns
def validate_age(age):
    if age > 18:
        return True
    return False

# Cosmetic only: rename variable
def validate_age(user_age):
    if user_age < 18:
        return False
    return True
```

Similarity scores to the original:

```
Cosmetic only (age→user_age):  0.9823
One operator (< → >):          0.8156
Two operators (> + swap):      0.7234
```

Perfect behavior! The cosmetic change barely affects the embedding (0.98), while operator changes significantly reduce similarity. The model correctly recognizes that changing `<` to `>` fundamentally alters the function's logic, even though it's just a single character change.

This is where generic text embeddings completely fail. To a text model, changing one character out of dozens shouldn't matter much. But UniXcoder understands that some characters—operators, keywords, structural elements—are far more important than others.

## Structure Understanding

Let's test how the model handles structural variations of the same logic:

```python
original = "def add(a, b): return a + b"

variations = {
    "renamed_vars": "def add(x, y): return x + y",
    "with_types": "def add(a: int, b: int) -> int: return a + b",
    "multiline": """def add(a, b):
        result = a + b
        return result""",
    "different_op": "def multiply(a, b): return a * b"
}
```

Similarity scores:

```
renamed_vars:    0.9734  (high - same logic, different names)
with_types:      0.9512  (high - added type hints)
multiline:       0.9156  (high - same logic, different style)
different_op:    0.7823  (lower - different operation)
```

The model recognizes that renaming variables, adding type hints, or using a different coding style (single line vs multiple lines) doesn't change the fundamental logic. But changing the operator does.

## What About Comments and Docstrings?

Code often includes natural language documentation. Does UniXcoder use this information?

```python
without_doc = "def validate_email(email): return '@' in email"

with_doc = """def validate_email(email):
    '''Check if email format is valid.'''
    return '@' in email"""

with_comment = """def validate_email(email):
    # Verify email has @ symbol
    return '@' in email"""
```

Comparing to the version without documentation:

```
With docstring:  0.9823
With comment:    0.9756
```

Both similarity scores are very high, which makes sense—the core logic is identical. But the scores aren't perfect 1.0, meaning the model does encode some information from the natural language text. This is useful: when searching for email validation code, having "email" and "valid" in the docstring provides additional signal that this is the right function.

## Cross-Language Understanding

UniXcoder was trained on multiple languages. Can it recognize similar logic across languages? Let's try Python and JavaScript:

```python
python_code = "def add(a, b): return a + b"
javascript_code = "function add(a, b) { return a + b; }"

emb_py = embedder.encode(python_code)
emb_js = embedder.encode(javascript_code)

similarity = cosine_similarity(emb_py.reshape(1, -1), 
                              emb_js.reshape(1, -1))[0][0]
```

Output:
```
Similarity: 0.8234
```

Strong similarity (0.82) despite completely different syntax! The model recognizes that `def` in Python and `function` in JavaScript serve the same purpose, that both define functions with parameters `a` and `b`, and both return their sum.

This cross-language understanding opens interesting possibilities. You could search for Python implementations of an algorithm even if your knowledge base includes JavaScript examples, or vice versa.

## Handling Incomplete Code

Real-world code search often involves partial snippets—maybe you're looking at a stack trace with just a few lines, or a code review with a small diff. How does the model handle fragments?

```python
complete = """def validate_password(pwd):
    if len(pwd) < 8:
        return False
    return any(c.isupper() for c in pwd)"""

fragment = "if len(pwd) < 8: return False"

similarity = cosine_similarity(
    embedder.encode(complete).reshape(1, -1),
    embedder.encode(fragment).reshape(1, -1)
)[0][0]
```

Output:
```
Similarity: 0.8567
```

Strong similarity even though the fragment is just one line from the complete function. This is crucial for practical code search—developers often work with partial code and need to find the complete context.

## Limitations

UniXcoder isn't perfect. It has a maximum context length (512 tokens), so very long functions get truncated. It was trained primarily on popular languages (Python, Java, JavaScript), so it might not understand exotic languages as well. And it captures syntactic and structural similarities better than deep semantic equivalence—two functions that compute the same result through completely different algorithms might not score as similar as you'd expect.

Most importantly, embeddings compress complex code into fixed-size vectors. Information is inevitably lost. Two functions might have the same embedding despite subtle but important differences. The model provides similarity scores, not guarantees of equivalence.

## Why This Matters

Code embeddings are the foundation for modern code intelligence tools: semantic search, duplicate detection, code completion, bug finding. They let us move beyond simple text matching to understanding what code actually does.

When you search GitHub for "email validation", you're not just matching the string "email validation". You're finding code that validates emails, even if it uses different words, different languages, or different approaches. That's only possible with embeddings that capture semantic meaning.

The key insight is that code has structure that generic text models miss. Operators matter more than variable names. Indentation carries meaning. A single character change (`<` to `>`) can completely reverse logic. Code-specific embeddings learn these patterns from millions of examples, giving us representations that reflect how programmers actually think about code.

Understanding embeddings is understanding how machines can begin to "read" code the way humans do—not as text, but as logic, structure, and meaning.