# Loop Agent Example

A simulated customer support agent that uses the **LoopAgent** pattern
(LLM + iterative reasoning loop) to handle incoming support tickets.

The agent analyses each ticket through multiple reasoning steps -- classifying
the issue, checking policies, formulating a response, and reviewing it --
before producing a final answer.

## Architecture

```
User Message
    |
    v
+-----------------+
|   LoopAgent     |
|  (ReasonLoop)   |
|                 |
|  step 1: parse  |
|  step 2: classify|
|  step 3: draft  |
|  step 4: review |
|  step 5: done   |
+-----------------+
    |
    v
Final Response
```

## Running

```bash
cd examples/loop
python main.py
```

No API keys required -- all LLM calls are simulated locally.
