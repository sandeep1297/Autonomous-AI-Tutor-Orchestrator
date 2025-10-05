
from langchain_core.tools import tool
from typing import Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum

# ---------- ENUMS ----------
class NoteStyleEnum(str, Enum):
    outline = "outline"
    bullet_points = "bullet_points"
    narrative = "narrative"
    structured = "structured"


class FlashcardDifficultyEnum(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class ExplanationDepthEnum(str, Enum):
    basic = "basic"
    intermediate = "intermediate"
    advanced = "advanced"
    comprehensive = "comprehensive"


# ---------- INPUT SCHEMAS ----------
class NoteMakerInput(BaseModel):
    topic: str
    note_taking_style: NoteStyleEnum
    subject: str
    include_examples: bool = True
    include_analogies: bool = False


class FlashcardGeneratorInput(BaseModel):
    topic: str
    count: int
    difficulty: FlashcardDifficultyEnum
    subject: str

    @validator("count")
    def validate_count(cls, v):
        if not 1 <= v <= 20:
            raise ValueError("Count must be between 1 and 20")
        return v


class ConceptExplainerInput(BaseModel):
    concept_to_explain: str
    desired_depth: ExplanationDepthEnum
    current_topic: str


# ---------- TOOL FUNCTIONS ----------
@tool("note_maker", args_schema=NoteMakerInput)
def note_maker_tool(**kwargs) -> Dict[str, Any]:
    """Generates structured study notes."""
    data = NoteMakerInput(**kwargs)
    return {
        "result": {
            "topic": data.topic,
            "style": data.note_taking_style,
            "examples": data.include_examples,
            "analogies": data.include_analogies,
        }
    }


@tool("flashcard_generator", args_schema=FlashcardGeneratorInput)
def flashcard_generator_tool(**kwargs) -> Dict[str, Any]:
    """Generates flashcards for a topic."""
    data = FlashcardGeneratorInput(**kwargs)
    flashcards = [
        {"q": f"What is point {i} about {data.topic}?", "a": f"Answer {i}."}
        for i in range(1, data.count + 1)
    ]
    return {"result": {"flashcards": flashcards, "difficulty": data.difficulty}}


@tool("concept_explainer", args_schema=ConceptExplainerInput)
def concept_explainer_tool(**kwargs) -> Dict[str, Any]:
    """Explains a concept with desired depth."""
    data = ConceptExplainerInput(**kwargs)
    return {
        "result": {
            "concept": data.concept_to_explain,
            "depth": data.desired_depth,
            "explanation": f"{data.desired_depth.value.title()} explanation of {data.concept_to_explain}.",
        }
    }


# ---------- TOOL REGISTRY ----------
LC_TOOLS = [note_maker_tool, flashcard_generator_tool, concept_explainer_tool]
TOOL_MAP = {t.name: t for t in LC_TOOLS}
