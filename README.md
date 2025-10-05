
# 🧠 YoLearn — Autonomous AI Tutor Orchestrator

## 📘 Overview
YoLearn is an **autonomous orchestration engine** built with **LangGraph** + **LangChain** that decides *which educational tool to use* based on a student’s message, extracts all required parameters intelligently, executes the tool, and returns a natural tutor-style response.

**Tech Stack**
- `FastAPI` — REST interface  
- `LangGraph` — state orchestration (`planner → executor → formatter`)  
- `LangChain` — LLM agent & tool invocation  
- `Gemini 2.0 Flash (ChatGoogleGenerativeAI)` — reasoning + structured tool planning  
- `Python 3.10+`, `dotenv` for config  

---

## ⚙️ System Architecture

The orchestrator uses a 3-node LangGraph pipeline for modular, scalable control flow.

```text
┌─────────────────────────────────────────────────────────────┐
│                  YoLearn Orchestrator Flow                  │
└─────────────────────────────────────────────────────────────┘

[ Student Message ]
        │
        ▼
[Planner Node] → [Executor Node] → [Formatter Node] → [Final Response]


### 🔹 Planner Node

Uses Gemini via LangChain’s `AgentExecutor` to interpret natural language and output structured JSON tool calls.
If the LLM fails or returns un-parsable output, a deterministic `rule_based_planner()` infers the tool and its parameters.

### 🔹 Executor Node

Dynamically looks up the callable in `TOOL_MAP` and invokes it with validated arguments.

### 🔹 Formatter Node

Converts structured tool results into clear, friendly tutor replies.

---

## 🚀 Quick Start

### 1️⃣ Setup Environment

```bash
git clone <repo>
cd yolearn.ai
python -m venv hack
source hack/bin/activate  # Windows: hack\Scripts\activate
pip install -r requirements.txt
```

Create **.env**

```bash
GOOGLE_API_KEY=<your-gemini-api-key>
```

### 2️⃣ Run Server

```bash
uvicorn main:app --reload
```

Then open:
📍 Swagger UI: http://127.0.0.1:8000/docs
📍 OpenAPI JSON: http://127.0.0.1:8000/openapi.json

---

## 💬 Example Interactions

### 🧾 1. Note Maker

**Input**

```
Generate structured notes on protein synthesis and include examples.
```

**Response**

```json
{
  "tool_name": "note_maker",
  "tool_args": {
    "topic": "Protein Synthesis",
    "note_taking_style": "structured",
    "subject": "Biology",
    "include_examples": true,
    "include_analogies": false
  },
  "final_response": "📘 Here are your **Protein Synthesis** notes (structured). Ready to start?"
}
```

### 🃏 2. Flashcard Generator

**Input**

```
Generate 10 flashcards on Newton's Laws of Motion for quick revision.
```

**Expected Output**

```json
{
  "tool_name": "flashcard_generator",
  "tool_args": {
    "topic": "Newton's Laws of Motion",
    "count": 10,
    "difficulty": "medium",
    "subject": "Physics"
  },
  "final_response": "🃏 Generated 10 flashcards on Newton's Laws of Motion. Start reviewing?"
}
```

### 🧠 3. Concept Explainer

**Input**

```
Explain quantum entanglement in simple terms.
```

**Expected Output**

```json
{
  "tool_name": "concept_explainer",
  "tool_args": {
    "concept_to_explain": "Quantum Entanglement",
    "desired_depth": "basic",
    "current_topic": "Physics"
  },
  "final_response": "🧠 Explanation for Quantum Entanglement ready. Would you like examples?"
}
```



## 🧩 Future Enhancements

* 🔁 Add memory node for multi-turn conversations
* ⚡ Implement caching of frequent tool results
* 🧮 Integrate adaptive difficulty selection for flashcards

