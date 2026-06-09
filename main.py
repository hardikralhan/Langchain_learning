"""
============================================================================
 LangChain / LangGraph CONCEPTS — a single runnable learning file
============================================================================

PURPOSE
    Run this file to *see* the core ideas of agentic AI work, one at a time:
      DEMO 1  Messages & why LLMs are "stateless"
      DEMO 2  LCEL chains  (prompt | model | parser)
      DEMO 3  Tools + Agent (the think -> act -> observe loop)
      DEMO 4  Short-term memory (checkpointer + thread_id)
      DEMO 5  Summarization middleware (triggered by TOKEN count)
      DEMO 6  Long-term memory (Store, scoped by user_id)

REQUIREMENTS  (you said your project already has these)
    pip install langchain langgraph langchain-google-genai

API KEY
    This file uses Google Gemini. Set your key as an environment variable:
        export GOOGLE_API_KEY="your-key-here"      (mac/linux)
        setx   GOOGLE_API_KEY "your-key-here"       (windows)
    ...or just paste it into GOOGLE_API_KEY below.

HOW TO RUN
    python langchain_concepts_demo.py          # runs every demo in order
    python langchain_concepts_demo.py 3        # runs only DEMO 3
    python langchain_concepts_demo.py 1 4 5    # runs DEMOs 1, 4 and 5

NOTE: every demo makes real (cheap) Gemini calls, so each run costs a few
fractions of a cent. Run single demos while studying to keep it minimal.
============================================================================
"""

import os
import sys
import time
from dotenv import load_dotenv
load_dotenv()   # loads env vars from a .env file if you have one

# --- core message + chain building blocks ---------------------------------
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

# --- the model -------------------------------------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI

# --- agent + middleware + memory (these live in langchain / langgraph) -----
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver   # short-term memory
from langgraph.store.memory import InMemoryStore         # long-term memory


# ===========================================================================
# SETUP
# ===========================================================================

# Paste your key here as a fallback, or leave it and use the env variable.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# gemini-2.0-flash was RETIRED on June 1 2026.
# gemini-2.5-flash-lite is the best free-tier model now:
#   15 req/min  ·  1,000 req/day  ·  fast  ·  cheap
# Change to "gemini-2.5-flash" if you want a slightly smarter model (10 RPM).
MODEL_NAME = "gemini-2.5-flash-lite"


def build_model(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """Create one model instance. temperature=0 keeps answers deterministic."""
    if not GOOGLE_API_KEY:
        sys.exit(
            "\nNo Google API key found.\n"
            "Set GOOGLE_API_KEY as an environment variable, or paste it into\n"
            "the GOOGLE_API_KEY variable near the top of this file.\n"
        )
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
    )


def banner(text: str) -> None:
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


def show_messages(messages) -> None:
    """Pretty-print a list of messages so you can SEE the conversation flow.

    Every message has a .type: 'system', 'human', 'ai', or 'tool'.
    An AI message may also carry .tool_calls (the agent deciding to act).
    """
    for m in messages:
        role = getattr(m, "type", "?")
        if role == "ai" and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                print(f"   [AI -> wants tool] {tc['name']}({tc['args']})")
            if m.content:
                print(f"   [AI] {m.content}")
        elif role == "tool":
            print(f"   [TOOL result] {m.content}")
        else:
            text = m.content if isinstance(m.content, str) else str(m.content)
            print(f"   [{role.upper()}] {text}")


# ===========================================================================
# DEMO 1 — Messages & statelessness
#   Key idea: the LLM remembers NOTHING between calls. YOU carry the history.
# ===========================================================================

def demo_1_messages():
    banner("DEMO 1 — Messages & why the LLM is 'stateless'")
    model = build_model()

    print("\n(A) A conversation is just a typed list of messages:")
    conversation = [
        SystemMessage(content="You are a concise tutor. Answer in one sentence."),
        HumanMessage(content="What is an LLM?"),
    ]
    reply = model.invoke(conversation)
    print(f"   [AI] {reply.content}")

    time.sleep(5)
    print("\n(B) Now we ask a follow-up WITHOUT sending the history.")
    print("    The model has no idea what 'it' refers to — it forgot:")
    forgetful = model.invoke([HumanMessage(content="Give me one example of it.")])
    print(f"   [AI] {forgetful.content}")

    time.sleep(5)
    print("\n(C) Same follow-up, but THIS time we resend the whole history.")
    print("    We append the previous AI reply, then the new question:")
    conversation.append(AIMessage(content=reply.content))
    conversation.append(HumanMessage(content="Give me one example of it."))
    remembered = model.invoke(conversation)
    print(f"   [AI] {remembered.content}")

    print("\nLESSON: 'Memory' is not the model remembering — it's you resending")
    print("the message list every call. Everything later automates this for you.")


# ===========================================================================
# DEMO 2 — LCEL chains:  prompt | model | parser
#   Key idea: the pipe '|' streams data left to right through components.
# ===========================================================================

def demo_2_lcel():
    banner("DEMO 2 — LCEL chains (prompt | model | parser)")
    model = build_model()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful tutor who explains simply."),
        ("human", "Explain {topic} in exactly one short sentence."),
    ])
    parser = StrOutputParser()           # pulls the clean string out of the reply

    # The whole point of LCEL: snap pieces together with |
    chain = prompt | model | parser

    print("\nSingle input:")
    print("   ", chain.invoke({"topic": "vectors"}))

    # Free tier has a low RPM cap — .batch() fires requests in parallel by
    # default (thread pool), which is exactly how you hit the 429 limit fast.
    # max_concurrency=1 makes it sequential. On a paid API key, remove that
    # and the requests truly fire in parallel — THAT is the superpower.
    time.sleep(5)
    print("\n.batch() — multiple inputs, one chain call (sequential on free tier):")
    results = chain.batch(
        [{"topic": "tokens"}, {"topic": "embeddings"}, {"topic": "an AI agent"}],
        config={"max_concurrency": 1},   # remove this on a paid key for parallelism
    )
    for r in results:
        print("   -", r)

    print("\nLESSON: input -> prompt (fills {topic}) -> model -> parser -> string.")
    print("You also get .stream() and async .ainvoke() for free, no extra code.")


# ===========================================================================
# DEMO 3 — Tools + Agent (the think -> act -> observe loop)
#   Key idea: an agent is the model in a loop, allowed to CALL functions.
# ===========================================================================

# Simple tool functions. The docstring is what the model reads to decide
# when to use the tool, so write it clearly.

@tool
def add(a: int, b: int) -> int:
    """Add two integers together and return the sum."""
    return a + b


@tool
def count_letters(word: str) -> int:
    """Return how many letters are in a single word."""
    return len(word)


@tool
def get_weather(city: str) -> str:
    """Return today's (fake, hard-coded) weather for a given city."""
    fake = {"delhi": "38C and sunny", "london": "14C and rainy"}
    return fake.get(city.lower(), "weather data not available")


def demo_3_agent():
    banner("DEMO 3 — Tools + Agent (think -> act -> observe)")
    model = build_model()

    agent = create_agent(
        model=model,
        tools=[add, count_letters, get_weather],
        system_prompt="You are a helpful assistant. Use tools when they help.",
    )

    question = ("What is 12 + 30? Also how many letters are in 'agent'? "
                "And what's the weather in Delhi?")
    print(f"\nUser asks: {question}\n")

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    print("Full loop (watch the agent decide, call tools, then answer):")
    show_messages(result["messages"])

    print("\nLESSON: the agent reasoned about which tools to call, called them,")
    print("read the results, then composed a final answer — all in one loop.")


# ===========================================================================
# DEMO 4 — Short-term memory (checkpointer + thread_id)
#   Key idea: a checkpointer auto-saves state PER thread_id. Same id = remembers.
# ===========================================================================

def demo_4_short_term_memory():
    banner("DEMO 4 — Short-term memory (checkpointer + thread_id)")
    model = build_model()

    # Attaching a checkpointer makes the agent remember automatically.
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="You are a friendly assistant.",
        checkpointer=InMemorySaver(),      # short-term memory engine
    )

    thread_a = {"configurable": {"thread_id": "chat-A"}}

    print("\nThread A, turn 1: introduce ourselves")
    agent.invoke({"messages": [{"role": "user",
                  "content": "My name is Alex and I love pizza."}]}, thread_a)

    print("Thread A, turn 2: ask it to recall (we send ONLY the new message)")
    r = agent.invoke({"messages": [{"role": "user",
                      "content": "What is my name and what do I like?"}]}, thread_a)
    print(f"   [AI] {r['messages'][-1].content}")
    print("   -> It remembered, because thread_id matched and the checkpointer")
    print("      reloaded the earlier messages for us automatically.")

    print("\nNow a DIFFERENT thread_id (a brand-new conversation):")
    thread_b = {"configurable": {"thread_id": "chat-B"}}
    r2 = agent.invoke({"messages": [{"role": "user",
                       "content": "What is my name?"}]}, thread_b)
    print(f"   [AI] {r2['messages'][-1].content}")
    print("   -> Amnesia. Short-term memory is scoped to ONE thread_id.")

    print("\nLESSON: you never rebuilt the history by hand — the checkpointer did.")
    print("But this memory dies when the thread changes. (See DEMO 6 to fix that.)")


# ===========================================================================
# DEMO 5 — Summarization middleware, triggered by TOKEN count
#   Key idea: when history gets big, auto-compress old messages into a summary
#             so you stay under the context limit (and cut cost/latency).
# ===========================================================================

def demo_5_summarization_middleware():
    banner("DEMO 5 — Summarization middleware (trigger = tokens)")
    model = build_model()

    # trigger=("tokens", N): summarize once the conversation passes N tokens.
    # keep=("messages", M):  after summarizing, keep the last M messages intact.
    # We use a small token threshold so it actually fires during this demo.
    summarizer = SummarizationMiddleware(
        model=model,
        trigger=("tokens", 250),     # <-- the TOKEN trigger you asked about
        keep=("messages", 4),        # keep the 4 most recent messages verbatim
    )

    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="You are a concise assistant.",
        middleware=[summarizer],
        checkpointer=InMemorySaver(),
    )

    config = {"configurable": {"thread_id": "long-chat"}}

    # Send several turns to grow the history past the token trigger.
    turns = [
        "Tell me a fun fact about the planet Mars.",
        "Now one about the Moon.",
        "Now one about the Sun.",
        "Now one about Jupiter.",
        "Now one about Saturn's rings.",
        "Finally, summarize what we discussed.",
    ]

    for i, t in enumerate(turns, 1):
        agent.invoke({"messages": [{"role": "user", "content": t}]}, config)
        # Inspect the SAVED state after each turn to watch the history.
        state = agent.get_state(config)
        msgs = state.values["messages"]
        approx = count_tokens_approximately(msgs)
        print(f"Turn {i}: stored messages = {len(msgs):2d}, "
              f"~tokens = {approx:4d}")
        time.sleep(5)   # stay well inside the 15 req/min free-tier rate limit

    print("\nFinal stored conversation (older turns get folded into a summary):")
    show_messages(agent.get_state(config).values["messages"])

    print("\nLESSON: middleware hooked into the loop and compressed history once it")
    print("crossed 250 tokens — keeping recent turns, summarizing the rest. This is")
    print("how long-running agents avoid blowing past the model's context window.")


# ===========================================================================
# DEMO 6 — Long-term memory (Store, scoped by user_id)
#   Key idea: the Store survives across ALL threads. Namespace by user, not thread.
# ===========================================================================

def demo_6_long_term_memory():
    banner("DEMO 6 — Long-term memory (Store, scoped by user_id)")

    store = InMemoryStore()
    user_id = "user-42"

    # A Store is explicit: YOU choose what to write and when to read.
    # The namespace is a tuple — put the user_id in it (NOT a thread_id),
    # so the same person is remembered across every future conversation.
    namespace = ("memories", user_id)

    print(f"\nWriting two facts about {user_id} to the Store...")
    store.put(namespace, "diet", {"fact": "is vegetarian"})
    store.put(namespace, "plan", {"fact": "is on the enterprise plan"})

    print("Later — in a totally different session/thread — we read them back:")
    hits = store.search(namespace)            # all memories for this user
    for item in hits:
        print(f"   - {item.key}: {item.value['fact']}")

    print("\nNow we feed those memories into a fresh agent as context:")
    model = build_model()
    facts = ", ".join(item.value["fact"] for item in hits)
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt=f"You are a shop assistant. Known facts about the user: {facts}.",
    )
    r = agent.invoke({"messages": [{"role": "user",
                      "content": "Suggest a meal for me and confirm my plan."}]})
    print(f"   [AI] {r['messages'][-1].content}")

    print("\nLESSON: short-term (checkpointer) = this conversation, per thread_id.")
    print("Long-term (Store) = facts about a PERSON, per user_id, across all chats.")
    print("Tip: to make the agent manage its own memory, give it memory tools")
    print("(e.g. the LangMem package) so it decides what to save during reasoning.")


# ===========================================================================
# RUNNER
# ===========================================================================

DEMOS = {
    "1": demo_1_messages,
    "2": demo_2_lcel,
    "3": demo_3_agent,
    "4": demo_4_short_term_memory,
    "5": demo_5_summarization_middleware,
    "6": demo_6_long_term_memory,
}


def main():
    # Decide which demos to run from command-line args (default: all).
    chosen = sys.argv[1:] or list(DEMOS.keys())
    for key in chosen:
        fn = DEMOS.get(key)
        if fn is None:
            print(f"Unknown demo '{key}'. Valid options: {', '.join(DEMOS)}")
            continue
        fn()
        if len(chosen) > 1:
            print("   (pausing 8s before next demo to respect rate limits...)")
            time.sleep(8)
    print("\nDone. Re-run a single demo with, e.g.:  python "
          f"{os.path.basename(__file__)} 5\n")


if __name__ == "__main__":
    main()