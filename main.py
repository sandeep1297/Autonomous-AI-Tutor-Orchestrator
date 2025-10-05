"""FastAPI interface for YoLearn Orchestrator."""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Query
from orchestrator_graph import run_orchestrator_turn

app = FastAPI(
    title="YoLearn AI Tutor Orchestrator",
    description="LangGraph + LangChain autonomous tutor middleware",
    version="1.0"
)

@app.get("/")
def root():
    return {"message": "YoLearn Orchestrator is running 🚀"}

@app.post("/api/orchestrate")
async def orchestrate(message: str = Query(..., description="Student message")):
    result = run_orchestrator_turn(message)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
