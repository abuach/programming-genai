# Chapter 3: The Art and Science of Prompt Engineering

> "The difference between the almost right word and the right word is really a large matter—'tis the difference between the lightning bug and the lightning." — Mark Twain

If Mark Twain had lived to see large language models, he might have added: "And the difference between an almost-right prompt and the right prompt is the difference between a confused AI and a helpful collaborator."

## 3.1 Introduction: Conversations with Machines

When you first encounter a large language model, the experience can feel almost magical. You type a question, and back comes an answer that sounds remarkably human. But as you use these systems more, you begin to notice something curious: the *way* you ask the question matters just as much as *what* you're asking.

This is the domain of **prompt engineering**—the craft of designing inputs that guide language models to produce exactly the outputs you need. And make no mistake: it *is* a craft. While you don't need a PhD in machine learning to write effective prompts, you do need to understand how these models think (or rather, how they *don't* think in the way humans do).

Let's start our journey by understanding what we're actually talking to.

## 3.2 Understanding Your Interlocutor: How LLMs Really Work

Large language models are, at their core, sophisticated prediction engines. They don't "understand" language in the way you and I do. Instead, they've been trained on vast amounts of text to predict what word (or more precisely, what *token*) should come next in a sequence.

Here's a simple way to think about it: imagine you've read millions of books, articles, and conversations. Someone starts a sentence with "The weather today is..." and your brain immediately starts generating possibilities: "nice," "terrible," "unpredictable," "perfect for a walk." Your brain is doing something similar to what an LLM does—drawing on patterns it has seen before to predict what comes next.

But there's a crucial difference. When you predict what comes next, you're drawing on genuine understanding of concepts, context, and causality. When an LLM does it, it's performing an extraordinarily sophisticated pattern matching operation based on statistical regularities it learned during training.

This distinction matters because it shapes how we should interact with these systems. A prompt isn't just a question—it's a way of setting up the model's prediction machinery to generate a specific kind of continuation.

Let's see this in action with a simple example:

```python
import ollama

client = ollama.Client(host='http://localhost:11434')

def call_ollama(prompt, model="llama3.2", **options):
    """
    Send a prompt to Ollama and get a response.
    
    Args:
        prompt: The text prompt to send
        model: Which model to use
        **options: Additional parameters (temperature, top_k, etc.)
    
    Returns:
        The model's response as a string
    """
    response = client.generate(
        model=model,
        prompt=prompt,
        options=options
    )
    return response['response']

# Let's see how the model completes different prompts
prompts = [
    "The weather today is",
    "In my professional meteorological opinion, the weather today is",
    "WEATHER ALERT: Today's conditions are"
]

for prompt in prompts:
    response = call_ollama(prompt, temperature=0.7, num_predict=20)
    print(f"Prompt: {prompt}")
    print(f"Completion: {response}\n")
```

Notice how each prompt "sets up" the model differently. The first is neutral. The second implies we want a formal, expert opinion. The third suggests urgency and official information. The model responds to these cues because its training has taught it that certain language patterns typically follow others.

This is your first lesson in prompt engineering: **the prompt is context**. You're not just asking a question—you're creating a linguistic environment that shapes what the model predicts should come next.

## 3.3 The Control Panel: Model Parameters

Before we dive deeper into prompt design, we need to understand the knobs and dials we can turn to control the model's behavior. Think of these as the difference between asking someone to "suggest a restaurant" versus "list every restaurant in town alphabetically." The question is similar, but you want very different kinds of responses.

### 3.3.1 Temperature: Creativity vs. Consistency

**Temperature** controls how random or deterministic the model's outputs are. It's measured on a scale from 0.0 to 2.0 (though you'll rarely use values above 1.5).

- **Temperature = 0.0**: The model always picks the single most likely next token. Completely deterministic.
- **Temperature = 0.5**: Balanced between likely choices and occasional surprises.
- **Temperature = 1.0**: Full probability distribution—creative but coherent.
- **Temperature = 2.0**: Nearly random selection—very creative but often nonsensical.

Here's why this matters:

```python
def temperature_experiment():
    """
    Demonstrate how temperature affects output consistency and creativity.
    """
    prompt = "Write a creative opening line for a sci-fi story:"
    
    temperatures = [0.0, 0.7, 1.5]
    
    for temp in temperatures:
        print(f"\n{'='*60}")
        print(f"Temperature: {temp}")
        print(f"{'='*60}")
        
        # Generate 3 responses at this temperature
        for i in range(3):
            response = call_ollama(
                prompt, 
                temperature=temp,
                num_predict=30
            )
            print(f"Attempt {i+1}: {response}")

temperature_experiment()
```

When you run this, you'll notice something fascinating:

- At temperature 0.0, all three attempts produce *identical* output. The model is completely deterministic.
- At 0.7, you get variety, but the responses feel coherent and reasonable.
- At 1.5, you might get wild creativity—or occasionally, nonsense.

**When to use different temperatures:**

- **0.0 - 0.3**: Code generation, factual answers, anything where consistency matters
- **0.5 - 0.8**: General conversation, balanced creativity
- **0.9 - 1.5**: Creative writing, brainstorming, exploring possibilities

### 3.3.2 Top-K and Top-P: Narrowing the Field

Temperature alone doesn't give us complete control. We also need ways to limit *which* tokens the model considers at each step.

**Top-K sampling** limits the model to choosing from only the K most likely tokens:

```python
def demonstrate_topk():
    """
    Show how top-k constrains token selection.
    """
    prompt = "The secret to great coffee is"
    
    # Very restrictive: only top 5 tokens considered
    response_narrow = call_ollama(
        prompt,
        temperature=0.8,
        top_k=5,
        num_predict=30
    )
    
    # More exploratory: top 50 tokens
    response_wide = call_ollama(
        prompt,
        temperature=0.8,
        top_k=50,
        num_predict=30
    )
    
    print("Top-K = 5 (Focused):")
    print(response_narrow)
    print("\nTop-K = 50 (Exploratory):")
    print(response_wide)

demonstrate_topk()
```

**Top-P sampling** (also called nucleus sampling) is more sophisticated. Instead of a fixed number of tokens, it selects from the smallest set of tokens whose cumulative probability exceeds P:

```python
def demonstrate_topp():
    """
    Show how top-p creates dynamic token sets.
    """
    prompt = "In conclusion, the most important factor is"
    
    # Conservative: only most likely tokens (90% probability mass)
    response_conservative = call_ollama(
        prompt,
        temperature=0.8,
        top_p=0.5,
        num_predict=30
    )
    
    # Exploratory: include less likely tokens (95% probability mass)
    response_exploratory = call_ollama(
        prompt,
        temperature=0.8,
        top_p=0.95,
        num_predict=30
    )
    
    print("Top-P = 0.5 (Conservative):")
    print(response_conservative)
    print("\nTop-P = 0.95 (Exploratory):")
    print(response_exploratory)

demonstrate_topp()
```

**The key insight:** Top-K gives you a fixed-size pool of options at each step. Top-P adapts the pool size based on how confident the model is. When the model is very sure (like completing "The capital of France is..."), top-p might only consider 2-3 tokens. When it's less certain, it considers more options.

### 3.3.3 Combining Parameters: The Recipe for Success

These parameters interact in interesting ways. Ollama applies them in sequence:

1. **Top-K** filters down to the K most likely tokens
2. **Top-P** further filters based on cumulative probability
3. **Temperature** is applied to the remaining tokens to determine final selection

Here's a practical guide for common scenarios:

```python
def parameter_recipes():
    """
    Demonstrate parameter combinations for different use cases.
    """
    test_prompt = "Explain quantum entanglement"
    
    scenarios = {
        "Factual (code, documentation)": {
            "temperature": 0.1,
            "top_k": 20,
            "top_p": 0.5
        },
        "Balanced (general chat)": {
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9
        },
        "Creative (brainstorming)": {
            "temperature": 0.9,
            "top_k": 50,
            "top_p": 0.95
        },
        "Deterministic (testing)": {
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0
        }
    }
    
    for scenario, params in scenarios.items():
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario}")
        print(f"Parameters: {params}")
        print(f"{'='*60}")
        
        response = call_ollama(
            test_prompt,
            num_predict=50,
            **params
        )
        print(response)

parameter_recipes()
```

## 3.4 The Software Development Lifecycle: A Parameter Perspective

One of the most practical applications of understanding these parameters is knowing when to use which settings during software development. Different phases of the development lifecycle call for different levels of creativity and consistency.

Let me share a story. Last semester, one of my students—let's call her Maya—was using an LLM to help build a web application. She was frustrated because the code the model generated during implementation kept changing every time she ran it. Meanwhile, when she asked it to brainstorm features, the responses felt stale and repetitive.

The problem? She was using the same parameters for everything: temperature 0.5, which is perfectly middling for general chat but suboptimal for specialized tasks.

Here's how to think about parameters across the development lifecycle:

```python
def sdlc_parameter_guide():
    """
    Demonstrate optimal parameters for each SDLC phase.
    """
    phases = {
        "Requirements & Ideation": {
            "description": "Exploring possibilities, gathering creative solutions",
            "parameters": {"temperature": 0.9, "top_p": 0.95, "top_k": 50},
            "prompt": "Brainstorm 5 innovative features for a task management app"
        },
        "System Design": {
            "description": "Balance creativity with technical soundness",
            "parameters": {"temperature": 0.6, "top_p": 0.85, "top_k": 30},
            "prompt": "Suggest database schemas for a multi-tenant SaaS application"
        },
        "Implementation": {
            "description": "Precise, deterministic code generation",
            "parameters": {"temperature": 0.2, "top_p": 0.7, "top_k": 15},
            "prompt": "Write a Python function to validate email addresses with regex"
        },
        "Testing & QA": {
            "description": "Explore edge cases creatively",
            "parameters": {"temperature": 0.8, "top_p": 0.9, "top_k": 40},
            "prompt": "Generate 10 edge cases for testing a login function"
        },
        "Deployment": {
            "description": "Reliable, repeatable automation",
            "parameters": {"temperature": 0.1, "top_p": 0.6, "top_k": 10},
            "prompt": "Write a CI/CD pipeline configuration for GitHub Actions"
        }
    }
    
    for phase, config in phases.items():
        print(f"\n{'='*70}")
        print(f"PHASE: {phase}")
        print(f"Purpose: {config['description']}")
        print(f"Parameters: {config['parameters']}")
        print(f"{'='*70}")
        
        response = call_ollama(
            config['prompt'],
            num_predict=100,
            **config['parameters']
        )
        print(f"\nExample Output:\n{response}\n")

sdlc_parameter_guide()
```

The pattern should be clear: **increase temperature and sampling diversity when you want exploration; decrease them when you want consistency**.

This isn't just academic—it has real implications for your work. Maya eventually adjusted her approach: high temperature (0.9) during brainstorming sessions, low temperature (0.1-0.2) during code generation, and back to high temperature (0.8) when generating test cases. Her productivity improved dramatically, and more importantly, she stopped fighting the tool.

## 3.5 The Prompt Itself: Zero-Shot, One-Shot, and Few-Shot Learning

Now that we understand how to control the *way* the model generates responses, let's focus on controlling *what* it generates. This is where prompt engineering becomes truly powerful.

### 3.5.1 Zero-Shot Prompting: The Direct Approach

**Zero-shot prompting** means asking the model to perform a task without providing any examples. You rely entirely on the model's training to understand what you want:

```python
def zero_shot_classification():
    """
    Classify text using only instructions, no examples.
    """
    def classify_sentiment(review):
        prompt = f"""Classify this movie review as POSITIVE, NEGATIVE, or NEUTRAL.
Return only the classification label.

Review: {review}

Classification:"""
        
        return call_ollama(
            prompt,
            temperature=0.1,
            num_predict=10
        ).strip()
    
    # Test reviews
    reviews = [
        "This movie was absolutely amazing! Best film of the year!",
        "Terrible waste of time. I want my money back.",
        "It was okay. Nothing special but not bad either.",
        "A masterpiece of cinema.",
        "I fell asleep halfway through."
    ]
    
    print("Zero-Shot Sentiment Classification\n" + "="*50)
    for review in reviews:
        sentiment = classify_sentiment(review)
        print(f"\nReview: {review}")
        print(f"Sentiment: {sentiment}")

zero_shot_classification()
```

Zero-shot works surprisingly well for many tasks because modern LLMs have been trained on such diverse data. But it has limitations:

- **Complex reasoning**: Multi-step problems often fail
- **Specific formats**: Getting exact JSON or structured output is unreliable
- **Domain knowledge**: Specialized terminology may be misunderstood
- **Ambiguity**: Unclear instructions lead to unpredictable results

### 3.5.2 One-Shot and Few-Shot: Teaching by Example

When zero-shot fails, we add examples. This is called **few-shot learning**—not because the model is learning in the traditional sense (its weights don't change), but because it's adapting its behavior based on the pattern you establish.

Here's a one-shot example:

```python
def one_shot_extraction():
    """
    Extract structured data using one example.
    """
    def extract_order(text):
        prompt = f"""Parse pizza orders into JSON format.

EXAMPLE:
Input: I want a small pizza with cheese and pepperoni.
Output: {{"size": "small", "toppings": ["cheese", "pepperoni"]}}

Now parse this:
Input: {text}
Output:"""
        
        return call_ollama(prompt, temperature=0.1, num_predict=100)
    
    orders = [
        "Large pizza with mushrooms and olives",
        "I'd like a medium with just cheese please",
        "Extra large with everything"
    ]
    
    print("One-Shot Order Parsing\n" + "="*50)
    for order in orders:
        result = extract_order(order)
        print(f"\nOrder: {order}")
        print(f"Parsed: {result}")

one_shot_extraction()
```

And here's a more powerful few-shot version:

```python
def few_shot_classification():
    """
    Classify emails using multiple examples to establish pattern.
    """
    def classify_email(email_body):
        prompt = f"""Classify emails as SPAM, IMPORTANT, or NORMAL.

Example 1:
Email: "Congratulations! You've won $1,000,000! Click here now!"
Classification: SPAM

Example 2:
Email: "Meeting with CEO rescheduled to tomorrow 9am. Please confirm."
Classification: IMPORTANT

Example 3:
Email: "Weekly newsletter: Here are this week's top articles."
Classification: NORMAL

Example 4:
Email: "Your account will be closed unless you verify within 24 hours!"
Classification: SPAM

Example 5:
Email: "Board meeting agenda attached. Review before Friday."
Classification: IMPORTANT

Now classify:
Email: {email_body}
Classification:"""
        
        return call_ollama(prompt, temperature=0.1, num_predict=10).strip()
    
    test_emails = [
        "URGENT: Limited time offer! Buy now!",
        "Q4 financial results ready for your review. Call me.",
        "Thanks for subscribing to our blog updates.",
        "Your package has been shipped and will arrive Tuesday.",
        "You are a winner! Claim your free iPhone now!"
    ]
    
    print("Few-Shot Email Classification\n" + "="*50)
    for email in test_emails:
        classification = classify_email(email)
        print(f"\nEmail: {email[:60]}...")
        print(f"Classification: {classification}")

few_shot_classification()
```

**Key principles for few-shot prompting:**

1. **3-6 examples is the sweet spot**: Too few and the pattern isn't clear; too many and you waste context window space.

2. **Diversity matters**: Your examples should cover the range of inputs you expect. Don't use five examples that are all basically the same.

3. **Quality over quantity**: One excellent, clear example is worth three mediocre ones.

4. **Order can matter**: Some models are sensitive to example order, though this varies.

Let's see how example quality affects results:

```python
def example_quality_comparison():
    """
    Compare good vs. poor few-shot examples.
    """
    task = "Extract key information into JSON"
    
    # Poor examples: inconsistent, unclear
    poor_prompt = """Extract information into JSON.

Ex: John is 30
{"name": "John", "age": 30}

Ex: Sarah teacher
{"name": "Sarah", "job": "teacher"}

Extract: {text}"""
    
    # Good examples: consistent, comprehensive
    good_prompt = """Extract person information into JSON with fields: name, age, occupation.

Example 1:
Input: My name is John Smith, I'm 30 years old, and I work as an engineer.
Output: {{"name": "John Smith", "age": 30, "occupation": "engineer"}}

Example 2:
Input: Sarah is 25 and teaches mathematics.
Output: {{"name": "Sarah", "age": 25, "occupation": "mathematics teacher"}}

Example 3:
Input: Dr. Mike Chen, 42, physician
Output: {{"name": "Mike Chen", "age": 42, "occupation": "physician"}}

Now extract:
Input: {text}
Output:"""
    
    test_text = "Alice Johnson is a 28-year-old software developer."
    
    print("POOR Examples:")
    print(call_ollama(
        poor_prompt.format(text=test_text),
        temperature=0.1
    ))
    
    print("\n" + "="*60)
    print("\nGOOD Examples:")
    print(call_ollama(
        good_prompt.format(text=test_text),
        temperature=0.1
    ))

example_quality_comparison()
```

## 3.6 Role Playing and System Prompts

One of the most powerful techniques in prompt engineering is giving the model a **role** or **persona**. This isn't just theatrical—it's a way of activating different patterns in the model's training data.

### 3.6.1 The Power of System Prompts

In most modern LLM APIs, prompts are structured as conversations with different **roles**:

- **system**: Sets overall behavior and constraints
- **user**: Represents the human's input
- **assistant**: The model's previous responses (for context)

The system prompt is particularly powerful because it establishes the framing for the entire conversation:

```python
def chat_ollama(messages, model="llama3.2", **options):
    """
    Send a chat-formatted conversation to Ollama.
    """
    response = client.chat(
        model=model,
        messages=messages,
        options=options
    )
    return response['message']['content']

def demonstrate_system_prompts():
    """
    Show how system prompts change model behavior.
    """
    question = "Explain how neural networks learn."
    
    scenarios = [
        {
            "name": "No System Prompt",
            "system": None,
            "description": "Baseline response"
        },
        {
            "name": "Concise Expert",
            "system": "You are a concise technical expert. Maximum 3 sentences.",
            "description": "Brief, dense explanation"
        },
        {
            "name": "Patient Teacher",
            "system": "You are a patient teacher explaining to someone new to the field. Use analogies and simple language.",
            "description": "Accessible explanation"
        },
        {
            "name": "Socratic Questioner",
            "system": "You answer questions by asking clarifying questions to help the user think through the problem themselves.",
            "description": "Guides rather than tells"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario['name']}")
        print(f"Goal: {scenario['description']}")
        print(f"{'='*70}\n")
        
        messages = []
        
        if scenario['system']:
            messages.append({
                'role': 'system',
                'content': scenario['system']
            })
        
        messages.append({
            'role': 'user',
            'content': question
        })
        
        response = chat_ollama(messages, temperature=0.7)
        print(response)

demonstrate_system_prompts()
```

### 3.6.2 Role-Based Prompting for Domain Expertise

You can use roles to activate domain-specific knowledge:

```python
def role_based_consultation():
    """
    Get perspectives from different professional roles.
    """
    def consult_expert(question, role, expertise):
        messages = [
            {
                'role': 'system',
                'content': f"You are a {role} with deep expertise in {expertise}. Provide advice from your professional perspective."
            },
            {
                'role': 'user',
                'content': question
            }
        ]
        return chat_ollama(messages, temperature=0.6)
    
    problem = "Our web application is slow. How should we diagnose and fix it?"
    
    experts = [
        ("Database Administrator", "query optimization and indexing"),
        ("Frontend Developer", "client-side performance and rendering"),
        ("DevOps Engineer", "infrastructure and scaling")
    ]
    
    print(f"Problem: {problem}\n")
    print("="*70)
    
    for role, expertise in experts:
        print(f"\nConsulting: {role} ({expertise})")
        print("-"*70)
        advice = consult_expert(problem, role, expertise)
        print(advice)
        print()

role_based_consultation()
```

This technique is incredibly useful when you need specialized perspectives on a problem. The model has been trained on text from many different domains, and role prompts help surface relevant patterns.

## 3.7 Chain of Thought: Teaching the Model to Reason

Here's where things get really interesting. One of the most significant discoveries in prompt engineering is that you can dramatically improve model performance on reasoning tasks by asking it to "show its work."

This technique is called **Chain of Thought (CoT)** prompting.

### 3.7.1 Zero-Shot Chain of Thought

The simplest version is almost embarrassingly effective. Just add "Let's think step by step" to your prompt:

```python
def zero_shot_cot():
    """
    Demonstrate zero-shot Chain of Thought reasoning.
    """
    problem = """When I was 6 years old, my sister was half my age.
Now I'm 70 years old. How old is my sister?"""
    
    # Without CoT
    print("WITHOUT Chain of Thought:")
    print("-"*60)
    response = call_ollama(problem, temperature=0.0, num_predict=100)
    print(response)
    
    # With CoT
    print("\n\nWITH Chain of Thought:")
    print("-"*60)
    cot_prompt = f"{problem}\n\nLet's think step by step:"
    response = call_ollama(cot_prompt, temperature=0.0, num_predict=150)
    print(response)

zero_shot_cot()
```

Why does this work? When you ask the model to think step-by-step, you're actually asking it to generate intermediate reasoning tokens before the final answer. This changes the computational path the model takes through the problem. Without CoT, the model tries to jump directly from question to answer—which works for simple problems but fails for complex reasoning. With CoT, it generates a sequence of reasoning steps, and each step provides context that helps generate the next step.

### 3.7.2 Few-Shot Chain of Thought

For even better results, provide examples that include reasoning:

```python
def few_shot_cot():
    """
    Demonstrate few-shot Chain of Thought with reasoning examples.
    """
    prompt = """Solve these math word problems step by step.

Example 1:
Problem: A train travels 60 mph for 2.5 hours. How far does it go?
Solution:
Step 1: Identify the given information
  - Speed = 60 mph
  - Time = 2.5 hours
Step 2: Apply the distance formula
  - Distance = Speed × Time
  - Distance = 60 × 2.5
Step 3: Calculate
  - Distance = 150 miles
Answer: 150 miles

Example 2:
Problem: Jane has $50. She spends $15 on lunch and $20 on a book. How much remains?
Solution:
Step 1: Identify starting amount and expenses
  - Starting: $50
  - Lunch: $15
  - Book: $20
Step 2: Calculate total spent
  - Total spent = $15 + $20 = $35
Step 3: Calculate remaining
  - Remaining = $50 - $35 = $15
Answer: $15

Now solve this problem:
Problem: A rectangle is 8 cm long and 5 cm wide. What is its area and perimeter?
Solution:"""
    
    response = call_ollama(prompt, temperature=0.0, num_predict=250)
    print("Few-Shot CoT Response:")
    print("="*60)
    print(response)

few_shot_cot()
```

### 3.7.3 When Chain of Thought Fails: The Limits of Reasoning

It's crucial to understand that CoT isn't magic. LLMs still don't "think" or "reason" in the way humans do. They're generating text that *looks like* reasoning, which often leads to correct answers, but can also lead to confident-sounding nonsense.

Let me show you a famous failure case:

```python
def cot_failure_example():
    """
    Demonstrate where CoT reasoning can fail.
    """
    # Modified river crossing problem
    problem = """A farmer is on one side of a river with a wolf, a goat, and a cabbage.
When crossing in a boat, he can only take one item at a time.
The wolf will eat the goat if left alone together.
The goat will eat the cabbage if left alone together.

How can the farmer transport the goat across the river without it being eaten?

Let's think through this step by step:"""
    
    response = call_ollama(problem, temperature=0.0, num_predict=300)
    print("CoT Response to Modified Problem:")
    print("="*60)
    print(response)
    print("\n" + "="*60)
    print("Analysis: The problem only asks about transporting the GOAT,")
    print("not all three items. The correct answer is simply:")
    print("'Put the goat in the boat and transport it across.'")
    print("\nBut the model may provide an unnecessarily complex solution")
    print("because it pattern-matches to the classic river-crossing puzzle.")

cot_failure_example()
```

This example illustrates a fundamental limitation: LLMs can be led astray by **misguided attention**. They pattern-match to familiar problems even when the actual problem is different. The model has seen many river-crossing puzzles in its training data, so it activates those patterns even though the question is much simpler.

This is why careful prompt engineering matters, and why you should never trust an LLM's output without verification, especially for critical applications.

