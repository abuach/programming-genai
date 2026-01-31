One of the most appealing things about modern Generative AI is of course the promise of Agentic: the potential to have fully autonomous AI that goes beyond words to enact actions in the real world that follow our direction. Watching an Agentic AI analyze its environment (in a local or cloud computing environment) and perform several tasks to reach its goal can feel very thrilling and satisfying. I can't match that experience in this humble Jupyter notebook, but instead I would like to discuss and demonstrate the fundamentals of implementic Agentic AI. All computation can be boiled down to the execution of certain functions with certain parameters, as seen in something like the simply-typed lambda calculus, for example. Similarly, we can reduce Agentic AI to the basic model of having a machine recommend: 
- given a certain environment state, 
- and also a certain goal, 
- which requires a sequence of actions to be realized
- what particular action needs to happen *next*
- what function `f` out of a set of possible functions best represents this action 
- and also what sequence of arguments `**kwargs` to pass to that function if it requires any

We basically encode the above steps as the body of the Agentic loop, which continues until the goal has been achieved and we can exit the loop. 
The agentic loop is in its simplest form a `while-break` pattern as we will see below.