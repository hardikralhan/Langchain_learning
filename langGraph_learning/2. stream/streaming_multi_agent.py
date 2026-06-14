"""
============================================================================
 MULTI-AGENT in LangGraph — the SUPERVISOR pattern (explicit wiring)
============================================================================

WHAT THIS SHOWS
    How several agents are CONNECTED in one LangGraph graph. We build it with
    raw nodes + edges (no prebuilt supervisor helper) so the wiring is visible.

    Topology:
        START -> supervisor -> (researcher | math) -> supervisor -> ... -> END

    * supervisor  = an agent that READS the conversation and DECIDES who acts
                    next ("researcher", "math", or "FINISH").
    * researcher  = a specialist agent with a get_population tool.
    * math        = a specialist agent with add / multiply tools.

    Each specialist does its part, then control RETURNS to the supervisor,
    which decides the next move. That return-to-supervisor loop is the heart
    of the supervisor pattern.

KEY IDEA
    An "agent" is just a node. Agents are connected the same way any nodes are:
      - normal edges  (A always goes to B):        add_edge("researcher","supervisor")
      - conditional edges (router picks the next): add_conditional_edges(...)
    The shared STATE (the message list) is how one agent sees what the others did.

REQUIREMENTS
    pip install langchain langgraph langchain-google-genai

RUN
    export GOOGLE_API_KEY="your-key"
    python multi_agent_supervisor.py

    Try changing USER_QUERY at the bottom to see different routing.

NOTE
    This makes several LLM calls (supervisor + each specialist). Small sleeps
    are added inside nodes to stay under the free-tier rate limit, so a full
    run takes ~30-60s. Run it on its own.
============================================================================
"""

import os
import sys
import time
from typing import TypedDict, Annotated
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env file

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ----------------------------------------------------------------------------
# SETUP
# ----------------------------------------------------------------------------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
MODEL_NAME = "gemini-2.5-flash-lite"   # current free-tier model (15 req/min)

# A small pause inside each node keeps us under the free-tier rate limit.
THROTTLE_SECONDS = 4


def build_model(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    if not GOOGLE_API_KEY:
        sys.exit(
            "\nNo Google API key found. Set GOOGLE_API_KEY as an environment "
            "variable, or paste it into GOOGLE_API_KEY near the top of this file.\n"
        )
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=temperature
    )


# ----------------------------------------------------------------------------
# TOOLS for the specialist agents (kept simple + fake, no external calls)
# ----------------------------------------------------------------------------
@tool
def get_population(city: str) -> str:
    """Return the (approximate, hard-coded) population of a city, in millions."""
    data = {"tokyo": "37", "delhi": "33", "paris": "11", "new york": "19"}
    return data.get(city.lower(), "unknown")


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


# ----------------------------------------------------------------------------
# SHARED STATE
#   messages : the conversation, shared by ALL agents (how they see each
#              other's work). add_messages reducer => appends, never replaces.
#   next     : set by the supervisor, read by the router edge.
#   visits   : a safety counter so the supervisor loop can't run forever.
# ----------------------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]
    next: str
    visits: int


# ----------------------------------------------------------------------------
# BUILD THE GRAPH
# ----------------------------------------------------------------------------
def build_graph():
    model = build_model()

    # --- two SPECIALIST AGENTS. Each is a full create_agent (its own little
    #     think->act->observe loop). They are nodes in our bigger graph. ---
    from langchain.agents import create_agent

    researcher = create_agent(
        model=model,
        tools=[get_population],
        system_prompt=("You are a research expert. Use your tools to find facts. "
                       "Answer in one short sentence with the number."),
    )
    mathematician = create_agent(
        model=model,
        tools=[add, multiply],
        system_prompt=("You are a math expert. Use your tools to calculate. "
                       "Answer in one short sentence with the result."),
    )

    # --- NODE: supervisor. Reads the conversation, decides who goes next. ---
    def supervisor_node(state: State) -> dict:
        time.sleep(THROTTLE_SECONDS)
        visits = state.get("visits", 0) + 1

        # Safety stop so the loop can't spin forever.
        if visits > 4:
            return {"next": "FINISH", "visits": visits}

        system = (
            "You are a SUPERVISOR coordinating two workers:\n"
            "  - 'researcher': looks up facts like city populations\n"
            "  - 'math': performs calculations\n"
            "Look at the conversation so far. Reply with ONE word only:\n"
            "  RESEARCHER  -> if a fact still needs looking up\n"
            "  MATH        -> if a calculation still needs doing\n"
            "  FINISH      -> if the user's request is fully answered\n"
        )
        decision = model.invoke([SystemMessage(content=system)] + state["messages"])
        choice = decision.content.strip().upper()

        if "RESEARCH" in choice:
            nxt = "researcher"
        elif "MATH" in choice:
            nxt = "math"
        else:
            nxt = "FINISH"

        print(f"   [supervisor] decision -> {nxt}")
        return {"next": nxt, "visits": visits}

    # --- NODE: researcher agent wrapper. Runs the sub-agent, hands its answer
    #     back into the shared messages so the supervisor/math can see it. ---
    def researcher_node(state: State) -> dict:
        time.sleep(THROTTLE_SECONDS)
        result = researcher.invoke({"messages": state["messages"]})
        answer = result["messages"][-1].content
        print(f"   [researcher] {answer}")
        return {"messages": [AIMessage(content=answer, name="researcher")]}

    # --- NODE: math agent wrapper. Same idea. ---
    def math_node(state: State) -> dict:
        time.sleep(THROTTLE_SECONDS)
        result = mathematician.invoke({"messages": state["messages"]})
        answer = result["messages"][-1].content
        print(f"   [math] {answer}")
        return {"messages": [AIMessage(content=answer, name="math")]}

    # --- ROUTER: a plain function the conditional edge uses to pick the path.
    #     It just reads state["next"] that the supervisor set. ---
    def route(state: State) -> str:
        return state["next"]

    # --- ASSEMBLE: nodes + edges. THIS is "how agents are connected". ---
    builder = StateGraph(State)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("math", math_node)

    builder.add_edge(START, "supervisor")              # entry

    # Conditional edge: supervisor -> whichever worker (or END) route() returns.
    builder.add_conditional_edges(
        "supervisor",
        route,
        {"researcher": "researcher", "math": "math", "FINISH": END},
    )

    # Normal edges: after a worker finishes, ALWAYS go back to the supervisor.
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("math", "supervisor")

    return builder.compile()


# ----------------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------------
USER_QUERY = ("What is the population of Tokyo (in millions)? "
              "Then multiply that number by 2.")


def main():
    graph = build_graph()

    print("\n" + "=" * 70)
    print("MULTI-AGENT SUPERVISOR  —  watch the handoffs between agents")
    print("=" * 70)
    print(f"\nUser: {USER_QUERY}\n")

    initial = {
        "messages": [HumanMessage(content=USER_QUERY)],
        "next": "",
        "visits": 0,
    }

    # stream_mode="updates" yields {node_name: update} after each node runs,
    # so we can watch the handoffs live. We also keep the latest agent answer.
    last_answer = None
    for chunk in graph.stream(initial, stream_mode="updates"):
        for node_name, update in chunk.items():
            print(f"-> node '{node_name}' just ran")
            # capture the most recent message any agent produced
            msgs = update.get("messages") if isinstance(update, dict) else None
            if msgs:
                last_answer = msgs[-1].content

    print("\n" + "-" * 70)
    print(f"FINAL ANSWER: {last_answer}")
    print("-" * 70)
    print("\nKEY TAKEAWAY: each agent is a NODE. They're connected by edges and")
    print("a conditional router, and they share STATE (the message list), which")
    print("is how each agent sees what the others already did.\n")


if __name__ == "__main__":
    main()