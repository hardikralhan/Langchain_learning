# LangChain Learning Repo

A hands-on learning repository for exploring core LangChain concepts using Google Gemini. 

## What's Inside

| File | Purpose |
|------|---------|
| `main.py` | Primary demo file (rate-limit safe, use this one) |

## Demos

`main.py` contains 6 progressive demos that cover the building blocks of agentic AI:

**Demo 1 — Messages & Statelessness**
Shows that LLMs have no memory between calls. You must resend the full message history on every call. The "memory" is the list you maintain, not the model.

**Demo 2 — LCEL Chains (`prompt | model | parser`)**
Demonstrates LangChain Expression Language. Components snap together with `|` to form pipelines. You get `.batch()`, `.stream()`, and `.ainvoke()` for free on any chain.

**Demo 3 — Tools + Agent (think → act → observe)**
An agent wraps the model in a loop and lets it call Python functions (tools). The model decides which tool to call, reads the result, and reasons again until it has a final answer.

**Demo 4 — Short-term Memory (checkpointer + thread_id)**
Attaching an `InMemorySaver` checkpointer makes the agent automatically save and reload conversation state per `thread_id`. Different thread = fresh conversation.

**Demo 5 — Summarization Middleware (token trigger)**
When conversation history grows past a token threshold, middleware compresses old messages into a summary and keeps only the most recent turns verbatim. Prevents blowing past the model's context window.

**Demo 6 — Long-term Memory (Store + user_id)**
An `InMemoryStore` persists facts across all threads, scoped by `user_id` instead of `thread_id`. This is how you remember things about a person across multiple separate conversations.

## Setup

**1. Install dependencies**

```bash
pip install langchain langgraph langchain-google-genai python-dotenv
```

Or with `uv`:

```bash
uv sync
```

**2. Set your API key**

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_key_here
```

Get a free key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).

## Running the Demos

```bash
# Run all demos in order
python main.py

# Run a single demo (e.g. Demo 3)
python main.py 3

# Run multiple specific demos
python main.py 1 4 5
```

> Each demo makes real Gemini API calls. Running single demos while studying keeps costs minimal (fractions of a cent per run). The free tier is 15 requests/min — `main.py` adds sleep delays to stay within limits.

## Model

Uses `gemini-2.5-flash-lite` by default — fast, cheap, and available on the free tier (15 req/min, 1000 req/day). Change `MODEL_NAME` at the top of `main.py` to switch models.
