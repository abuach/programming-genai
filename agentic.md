# Chapter 7: Tool Calling & Agentic AI

## Introduction: From Conversation to Action

Imagine you're working on a group project, and instead of just talking about what needs to be done, your teammate can actually *do* things—check the calendar, send emails, look up information, even write code. That's the leap we're making in this chapter: from language models that can only *talk* about actions to AI agents that can *take* actions.

In the early days of large language models, we were thrilled when they could answer questions and generate text. But there was always a gap: they lived in a world of words, disconnected from the tools and systems we use every day. Tool calling bridges that gap, transforming language models from eloquent conversationalists into capable assistants.

> **A Quick Joke**: Why did the AI agent break up with the chatbot? Because the chatbot could only talk about changing the world, but the agent could actually do it! 🔧

## The Core Idea: Structured Function Calling

At its heart, tool calling is about giving language models a structured way to say "I need to use this specific function with these specific parameters." Instead of just generating text that *describes* what should happen, the model generates structured data that your code can execute.

Let's see this in action with Ollama and the Qwen model:

```python
import ollama
import json

# Define a simple tool: get the current weather
tools = [{
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': 'Get the current weather for a location',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'City name, e.g. San Francisco'
                },
                'unit': {
                    'type': 'string',
                    'enum': ['celsius', 'fahrenheit'],
                    'description': 'Temperature unit'
                }
            },
            'required': ['location']
        }
    }
}]

response = ollama.chat(
    model='qwen2.5:latest',
    messages=[{'role': 'user', 'content': 'What is the weather in Paris?'}],
    tools=tools
)

print(response['message']['tool_calls'])
```

What's happening here? We've given the model a *schema*—a formal description of what the `get_weather` function expects. When the model sees "What is the weather in Paris?", it recognizes it needs to call a tool and outputs structured JSON rather than freeform text.

## Anatomy of a Tool Definition

Tool definitions follow a specific structure that tells the model everything it needs to know:

```python
weather_tool = {
    'type': 'function',  # Currently, 'function' is the only type
    'function': {
        'name': 'get_weather',  # Unique identifier
        'description': 'Get weather for a location',  # Helps model decide when to use it
        'parameters': {  # JSON Schema for the parameters
            'type': 'object',
            'properties': {
                'location': {'type': 'string', 'description': 'City name'},
                'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}
            },
            'required': ['location']
        }
    }
}
```

The `description` field is crucial—it's how the model decides *when* to use the tool. Good descriptions are clear, specific, and include examples when helpful.

## Building Your First Agent Loop

An agent isn't just one tool call—it's a *conversation* between the model and your tools. Here's the basic loop:

1. User sends a message
2. Model decides if it needs a tool
3. If yes, model generates tool call(s)
4. You execute the tool(s)
5. You send results back to model
6. Model responds to user (or calls more tools!)

Let's implement this:

```python
def get_weather(location, unit='celsius'):
    """Simulated weather API"""
    return {
        'location': location,
        'temperature': 22 if unit == 'celsius' else 72,
        'conditions': 'sunny',
        'unit': unit
    }

def run_agent(user_message):
    messages = [{'role': 'user', 'content': user_message}]
    
    response = ollama.chat(
        model='qwen2.5:latest',
        messages=messages,
        tools=[weather_tool]
    )
    
    # Check if model wants to call a tool
    if response['message'].get('tool_calls'):
        # Add model's response to messages
        messages.append(response['message'])
        
        # Execute each tool call
        for tool in response['message']['tool_calls']:
            if tool['function']['name'] == 'get_weather':
                args = tool['function']['arguments']
                result = get_weather(**args)
                
                # Add tool result to messages
                messages.append({
                    'role': 'tool',
                    'content': json.dumps(result),
                })
        
        # Get final response with tool results
        final_response = ollama.chat(
            model='qwen2.5:latest',
            messages=messages
        )
        return final_response['message']['content']
    
    return response['message']['content']

# Try it!
print(run_agent("What's the weather like in Tokyo?"))
```

Notice the message flow: user message → model response with tool call → tool result → final model response. This is the fundamental pattern of agentic AI.

## Multiple Tools: Expanding Capabilities

Real agents have access to multiple tools. Let's create a more interesting agent:

```python
tools = [
    {
        'type': 'function',
        'function': {
            'name': 'calculate',
            'description': 'Perform mathematical calculations',
            'parameters': {
                'type': 'object',
                'properties': {
                    'expression': {
                        'type': 'string',
                        'description': 'Math expression like "2 + 2" or "sqrt(16)"'
                    }
                },
                'required': ['expression']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'search_database',
            'description': 'Search a product database',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string', 'description': 'Search query'},
                    'max_results': {'type': 'integer', 'description': 'Max results'}
                },
                'required': ['query']
            }
        }
    }
]

def calculate(expression):
    """Safe calculator"""
    try:
        return {'result': eval(expression, {'__builtins__': {}}, 
                              {'sqrt': __import__('math').sqrt})}
    except:
        return {'error': 'Invalid expression'}

def search_database(query, max_results=5):
    """Simulated database"""
    products = {
        'laptop': {'name': 'UltraBook Pro', 'price': 1299},
        'phone': {'name': 'SmartPhone X', 'price': 899}
    }
    return [v for k, v in products.items() if query.lower() in k]
```

The model will now automatically choose which tool to use based on the user's request!

## Parallel Tool Calls: Efficiency Matters

Sometimes an agent needs to call multiple tools at once. Modern models support parallel tool calls:

```python
# User asks: "What's the weather in London and Paris?"
response = ollama.chat(
    model='qwen2.5:latest',
    messages=[{
        'role': 'user',
        'content': 'What is the weather in London and Paris?'
    }],
    tools=[weather_tool]
)

# Model might return multiple tool calls at once!
for tool_call in response['message'].get('tool_calls', []):
    print(f"Calling {tool_call['function']['name']} with args:")
    print(tool_call['function']['arguments'])
```

This is more efficient than sequential calls and shows how agents can be surprisingly sophisticated in their planning.

## Error Handling: When Tools Fail

Tools don't always succeed. Your agent needs to handle errors gracefully:

```python
def robust_agent_loop(user_message, max_turns=5):
    messages = [{'role': 'user', 'content': user_message}]
    
    for turn in range(max_turns):
        response = ollama.chat(
            model='qwen2.5:latest',
            messages=messages,
            tools=tools
        )
        
        if not response['message'].get('tool_calls'):
            return response['message']['content']
        
        messages.append(response['message'])
        
        for tool_call in response['message']['tool_calls']:
            func_name = tool_call['function']['name']
            args = tool_call['function']['arguments']
            
            try:
                # Execute tool
                if func_name == 'calculate':
                    result = calculate(args['expression'])
                elif func_name == 'search_database':
                    result = search_database(args['query'])
                else:
                    result = {'error': f'Unknown tool: {func_name}'}
            except Exception as e:
                result = {'error': str(e)}
            
            messages.append({
                'role': 'tool',
                'content': json.dumps(result)
            })
    
    return "Max turns reached. Task incomplete."
```

The `max_turns` parameter prevents infinite loops—a critical safety feature for autonomous agents.

## Chain of Thought with Tools

Sometimes agents need to "think" before acting. We can encourage this:

```python
def thoughtful_agent(user_message):
    messages = [
        {'role': 'system', 'content': 
         'Think step-by-step before using tools. Explain your reasoning.'},
        {'role': 'user', 'content': user_message}
    ]
    
    response = ollama.chat(
        model='qwen2.5:latest',
        messages=messages,
        tools=tools
    )
    
    # Model might respond with reasoning before tool calls
    if response['message'].get('content'):
        print("Agent's reasoning:", response['message']['content'])
    
    # Then handle tool calls as before...
```

This approach leads to more transparent and debuggable agents.

## Real-World Example: Research Assistant

Let's build something practical—a research assistant that can search and summarize:

```python
import requests

research_tools = [
    {
        'type': 'function',
        'function': {
            'name': 'search_arxiv',
            'description': 'Search academic papers on arXiv',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string', 'description': 'Search terms'},
                    'max_results': {'type': 'integer', 'description': 'Papers to return'}
                },
                'required': ['query']
            }
        }
    }
]

def search_arxiv(query, max_results=3):
    """Search arXiv (simplified example)"""
    # In real implementation, use arxiv API
    return {
        'papers': [
            {'title': 'Attention Is All You Need', 'year': 2017},
            {'title': 'BERT: Pre-training Deep Bidirectional Transformers', 'year': 2018}
        ]
    }

# Agent can now help with research queries!
result = run_agent("Find recent papers on transformer architectures")
```

## Security Considerations: Sandboxing Tools

Never give an AI agent unlimited access to your system! Here's a safer approach:

```python
class SafeToolExecutor:
    def __init__(self):
        self.allowed_tools = {
            'get_weather': get_weather,
            'calculate': calculate
        }
        self.call_count = 0
        self.max_calls = 10
    
    def execute(self, tool_name, args):
        if self.call_count >= self.max_calls:
            raise Exception("Tool call limit exceeded")
        
        if tool_name not in self.allowed_tools:
            raise Exception(f"Tool {tool_name} not allowed")
        
        self.call_count += 1
        return self.allowed_tools[tool_name](**args)

# Use in agent loop
executor = SafeToolExecutor()
result = executor.execute('get_weather', {'location': 'Paris'})
```

Key security principles:
- **Whitelist tools** rather than blacklist
- **Rate limit** tool calls
- **Validate inputs** before execution
- **Audit logs** for all tool usage
- **Principle of least privilege**: only give tools necessary permissions

## Stateful Agents: Memory and Context

Agents become more powerful when they remember past interactions:

```python
class StatefulAgent:
    def __init__(self):
        self.conversation_history = []
        self.facts_learned = {}
    
    def add_fact(self, key, value):
        """Tool for agent to remember things"""
        self.facts_learned[key] = value
        return {'status': 'success', 'key': key}
    
    def recall_fact(self, key):
        """Tool to retrieve remembered information"""
        return {'value': self.facts_learned.get(key, 'Not found')}
    
    def chat(self, user_message):
        self.conversation_history.append({
            'role': 'user',
            'content': user_message
        })
        
        # Include memory tools
        memory_tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'remember',
                    'description': 'Store a fact for later recall',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'key': {'type': 'string'},
                            'value': {'type': 'string'}
                        },
                        'required': ['key', 'value']
                    }
                }
            }
        ]
        
        # Agent can now remember things across turns!
```

This enables agents that learn about users' preferences, project context, and domain knowledge.

## Evaluation: How Good Is Your Agent?

Building agents is one thing; knowing if they work well is another:

```python
def evaluate_agent(test_cases):
    results = []
    
    for case in test_cases:
        try:
            response = run_agent(case['input'])
            
            # Check if response contains expected elements
            success = all(
                expected in response.lower() 
                for expected in case['expected_keywords']
            )
            
            results.append({
                'input': case['input'],
                'success': success,
                'response': response
            })
        except Exception as e:
            results.append({
                'input': case['input'],
                'success': False,
                'error': str(e)
            })
    
    success_rate = sum(r['success'] for r in results) / len(results)
    return success_rate, results

# Example test cases
tests = [
    {
        'input': 'What is 15 * 23?',
        'expected_keywords': ['345']
    },
    {
        'input': 'Weather in Tokyo',
        'expected_keywords': ['tokyo', 'temperature']
    }
]

score, details = evaluate_agent(tests)
print(f"Agent success rate: {score:.1%}")
```

## Advanced Pattern: Agent Reflection

One of the most powerful patterns is giving agents the ability to reflect on their actions:

```python
def reflective_agent(user_message):
    messages = [{'role': 'user', 'content': user_message}]
    
    # First attempt
    response = ollama.chat(model='qwen2.5:latest', 
                          messages=messages, tools=tools)
    
    # Execute tools...
    # (tool execution code here)
    
    # Ask agent to reflect
    messages.append({
        'role': 'user',
        'content': 'Did your tool calls fully answer the question? '
                   'If not, what additional information is needed?'
    })
    
    reflection = ollama.chat(model='qwen2.5:latest', messages=messages)
    
    # Agent can identify gaps in its own reasoning!
    return reflection['message']['content']
```

This meta-cognitive ability helps agents correct mistakes and handle complex queries.

## The Future: Multimodal Agents

The frontier of agentic AI involves tools that process images, audio, and more:

```python
multimodal_tools = [
    {
        'type': 'function',
        'function': {
            'name': 'analyze_image',
            'description': 'Analyze an image and describe its contents',
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_url': {'type': 'string', 'description': 'URL to image'}
                },
                'required': ['image_url']
            }
        }
    }
]

# Future agents will seamlessly integrate vision, audio, and text!
```

## Conclusion: From Tools to Teammates

Tool calling transforms language models from impressive mimics into genuine assistants. But with this power comes responsibility: we must design agents that are safe, transparent, and aligned with human values.

As you build your own agents, remember:
- Start simple, add complexity gradually
- Test extensively with diverse inputs
- Build in safety limits from the start
- Make agent reasoning visible when possible
- Never grant tools more access than necessary

The joke about AI agents isn't that they'll replace programmers—it's that they'll make us all better at programming by handling the tedious parts while we focus on the creative and strategic work. The future of software development isn't human *or* AI; it's human *and* AI, working together as a team.

## Exercises

1. **Build a File Manager Agent**: Create an agent with tools to list, read, and create text files in a sandboxed directory.

2. **Implement Tool Versioning**: Design a system where tools can have multiple versions, and the agent specifies which version to use.

3. **Create an Agent Debugger**: Build a tool that logs all agent decisions, tool calls, and results for later analysis.

4. **Multi-Agent Collaboration**: Design two agents with different tool sets that can delegate tasks to each other.

5. **Privacy-Preserving Agent**: Implement an agent that can redact sensitive information from tool calls and results.

---

*"The best thing about boolean logic is that even if you're wrong, you're only off by a bit!"* 💻

Happy agent building!