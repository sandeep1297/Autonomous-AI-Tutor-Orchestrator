
import os
import re
import json
from typing import Dict, Any, TypedDict, Optional
from dotenv import load_dotenv

# LangChain / LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END

# Project-local imports
from tools import TOOL_MAP, LC_TOOLS
from context import MOCK_USER_INFO, MOCK_STUDENT_CONTEXT

load_dotenv()

# ---------------------------------------------
# Env check
# ---------------------------------------------
if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError("Missing GOOGLE_API_KEY in environment (.env).")

# ---------------------------------------------
# System Prompt for Planner Agent
# ---------------------------------------------
SYSTEM_PROMPT = f"""
You are the YoLearn Autonomous AI Tutor Orchestrator.

Your ONLY task:
→ Select one correct tool and return a structured JSON specifying the tool and its parameters.

Available tools:
- note_maker(topic: str, note_taking_style: str, subject: str, include_examples: bool, include_analogies: bool)
- flashcard_generator(topic: str, count: int, difficulty: str, subject: str)
- concept_explainer(concept_to_explain: str, desired_depth: str, current_topic: str)

Context:
User Info: {MOCK_USER_INFO}
Student Context: {MOCK_STUDENT_CONTEXT}

Rules:
- Output ONLY valid JSON (no extra text).
- Return exactly ONE tool call.
- If any parameter is missing, infer a reasonable default.
"""

# ---------------------------------------------
# Initialize LLM and Agent
# ---------------------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        MessagesPlaceholder(variable_name="intermediate_steps"),
        ("human", "{input}"),
    ]
)

_agent = create_tool_calling_agent(llm, LC_TOOLS, prompt)
_agent_executor = AgentExecutor(agent=_agent, tools=LC_TOOLS, verbose=False)


# ---------------------------------------------
# Orchestrator State
# ---------------------------------------------
class OrchestratorState(TypedDict, total=False):
    user_message: str
    tool_name: Optional[str]
    tool_args: Optional[Dict[str, Any]]
    tool_output: Optional[Dict[str, Any]]
    status: Optional[str]
    final_response: Optional[str]
    llm_raw: Optional[str]
    fallback_used: Optional[bool]
    error: Optional[str]


# ---------------------------------------------
# Rule-based Planner (fallback)
# ---------------------------------------------
def rule_based_planner(user_input: str) -> Dict[str, Any]:
    s = user_input.lower()

    if any(k in s for k in ["note", "summary", "outline", "study"]):
        return {
            "tool_name": "note_maker",
            "tool_args": {
                "topic": user_input,
                "note_taking_style": "structured",
                "subject": MOCK_STUDENT_CONTEXT.get("subject", "General"),
                "include_examples": True,
                "include_analogies": "confuse" in s or "confused" in s,
            },
        }

    if any(k in s for k in ["flashcard", "quiz", "practice"]):
        count = 5
        m = re.search(r"\b(\d{1,2})\b", s)
        if m:
            count = int(m.group(1))
        return {
            "tool_name": "flashcard_generator",
            "tool_args": {
                "topic": user_input,
                "count": max(1, min(count, 20)),
                "difficulty": "medium",
                "subject": MOCK_STUDENT_CONTEXT.get("subject", "General"),
            },
        }

    if any(k in s for k in ["explain", "define", "describe", "clarify"]):
        depth = "intermediate"
        if "simple" in s or "basic" in s:
            depth = "basic"
        if "advanced" in s or "detailed" in s:
            depth = "advanced"
        return {
            "tool_name": "concept_explainer",
            "tool_args": {
                "concept_to_explain": user_input,
                "desired_depth": depth,
                "current_topic": MOCK_STUDENT_CONTEXT.get("last_topic", "General"),
            },
        }

    return {"tool_name": None, "tool_args": None}


# ---------------------------------------------
# Planner Node
# ---------------------------------------------
def planner_node(state: OrchestratorState) -> OrchestratorState:
    """Planner node that first tries LLM (Gemini) → then falls back to rule-based planner if needed."""
    user_input = state.get("user_message", "")
    state["fallback_used"] = False

    try:
        result = _agent_executor.invoke({"input": user_input})
        state["llm_raw"] = str(result)

        # --- Extract the raw text from LLM output ---
        output = result.get("output") or result

        # If output is a string, Gemini may wrap JSON in ```json ... ```
        if isinstance(output, str):
            import re, json

            # Clean out markdown code fences
            clean_output = re.sub(r"```(?:json)?", "", output, flags=re.IGNORECASE)
            clean_output = clean_output.replace("```", "").strip()

            # Try parsing JSON if possible
            try:
                parsed = json.loads(clean_output)
                name = parsed.get("tool_name") or parsed.get("tool") or parsed.get("name")
                args = (
                    parsed.get("tool_args")
                    or parsed.get("args")
                    or parsed.get("tool_input")
                    or parsed.get("parameters")  # ✅ Gemini sometimes uses this key
                    or {}
                )
                if name:
                    state["tool_name"] = name
                    state["tool_args"] = args
                    state["status"] = "FOUND_TOOL"
                    return state
            except Exception:
                pass  # fall through to other parsing methods

        # If still a dict, use the direct extraction path
        if isinstance(output, dict):
            name = output.get("name") or output.get("tool_name")
            args = (
                output.get("args")
                or output.get("tool_args")
                or output.get("tool_input")
                or output.get("parameters")
                or {}
            )
            if name:
                state["tool_name"] = name
                state["tool_args"] = args
                state["status"] = "FOUND_TOOL"
                return state

        # If we reach here, the LLM didn’t provide a structured tool call
        raise ValueError("LLM failed to produce structured tool call.")

    except Exception as e:
        print(f"[Planner] ⚠️ Falling back due to error: {e}")
        fb = rule_based_planner(user_input)
        state["tool_name"] = fb.get("tool_name")
        state["tool_args"] = fb.get("tool_args")
        state["fallback_used"] = True
        state["status"] = "FOUND_TOOL" if fb.get("tool_name") else "NO_TOOL"
        state["error"] = str(e)
        return state



# ---------------------------------------------
# Executor Node (✅ fixed .invoke() call)
# ---------------------------------------------
def executor_node(state: OrchestratorState) -> OrchestratorState:
    name = state.get("tool_name")
    args = state.get("tool_args") or {}

    if not name:
        state["status"] = "NO_TOOL"
        state["final_response"] = "⚠️ Unable to determine a suitable tool."
        return state

    tool_func = TOOL_MAP.get(name)
    if not tool_func:
        state["status"] = "UNKNOWN_TOOL"
        state["final_response"] = f"❌ Unknown tool: {name}"
        return state

    try:
        # ✅ FIXED: LangChain tools use .invoke(dict)
        result = tool_func.invoke(args)
        state["tool_output"] = result
        state["status"] = "TOOL_SUCCESS"
        print(f"[Executor] ✅ {name} executed successfully.")
    except Exception as e:
        state["status"] = "TOOL_ERROR"
        state["error"] = str(e)
        state["final_response"] = f"❌ Tool execution failed: {e}"
        print(f"[Executor] ❌ Tool failed: {e}")
    return state


# ---------------------------------------------
# Formatter Node
# ---------------------------------------------
def formatter_node(state: OrchestratorState) -> OrchestratorState:
    status = state.get("status")

    if status == "TOOL_SUCCESS":
        name = state["tool_name"]
        args = state.get("tool_args", {})
        if name == "note_maker":
            topic = args.get("topic", "your topic")
            msg = f"📘 Here are your **{topic}** notes (structured). Ready to start?"
        elif name == "flashcard_generator":
            msg = f"🃏 Generated {args.get('count', 5)} flashcards on {args.get('topic', 'this topic')}."
        elif name == "concept_explainer":
            msg = f"🧠 Explanation for **{args.get('concept_to_explain', 'the concept')}** ready!"
        else:
            msg = f"✅ Tool {name} executed successfully."
        state["final_response"] = msg
        state["status"] = "SUCCESS"
        return state

    if status in ("NO_TOOL", "LLM_ERROR"):
        state["final_response"] = "⚠️ Unable to determine a suitable tool."
        return state

    if status == "TOOL_ERROR":
        return state

    state["final_response"] = state.get("final_response") or "⚠️ Could not complete request."
    return state


# ---------------------------------------------
# Build LangGraph
# ---------------------------------------------
graph = StateGraph(OrchestratorState)
graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("formatter", formatter_node)
graph.add_edge("planner", "executor")
graph.add_edge("executor", "formatter")
graph.set_entry_point("planner")
graph.set_finish_point("formatter")
orchestrator_graph = graph.compile()


# ---------------------------------------------
# Public Entrypoint
# ---------------------------------------------
def run_orchestrator_turn(user_message: str) -> Dict[str, Any]:
    initial_state = {"user_message": user_message}
    final = orchestrator_graph.invoke(initial_state)

    return {
        "status": final.get("status"),
        "final_response": final.get("final_response"),
        "tool_name": final.get("tool_name"),
        "tool_args": final.get("tool_args"),
        "raw_state": final,
        "llm_raw": final.get("llm_raw"),
        "fallback_used": final.get("fallback_used", False),
        "error": final.get("error"),
    }
