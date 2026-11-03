"""LangGraph wiring for the agent roles, and the single entry point `run_agent`."""

from langgraph.graph import END, START, StateGraph

from docqa.agents import nodes
from docqa.agents.nodes import AgentContext, AgentState, BudgetedLLM
from docqa.errors import DocQAError
from docqa.guardrails import validate_question
from docqa.llm import LLM
from docqa.models import AgentResult
from docqa.workspace import Workspace


def _after_planner(state: AgentState) -> str:
    return {"unrelated": END, "table": "analyst"}.get(state["route"], "retriever")


def _after_retriever(state: AgentState) -> str:
    if state.get("final") is not None:
        return END
    return "answerer" if state.get("hits") else "retriever"  # no hits yet: query was rewritten


def _after_analyst(state: AgentState) -> str:
    return "answerer" if state["route"] == "table" else "retriever"


def _after_answerer(state: AgentState) -> str:
    return END if state.get("final") is not None else "validator"


def build_graph(ctx: AgentContext):
    graph = StateGraph(AgentState)
    graph.add_node("planner", lambda state: nodes.planner(state, ctx))
    graph.add_node("retriever", lambda state: nodes.retriever(state, ctx))
    graph.add_node("analyst", lambda state: nodes.analyst(state, ctx))
    graph.add_node("answerer", lambda state: nodes.answerer(state, ctx))
    graph.add_node("validator", lambda state: nodes.validator(state, ctx))
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", _after_planner, ["analyst", "retriever", END])
    graph.add_conditional_edges("retriever", _after_retriever, ["answerer", "retriever", END])
    graph.add_conditional_edges("analyst", _after_analyst, ["answerer", "retriever"])
    graph.add_conditional_edges("answerer", _after_answerer, ["validator", END])
    graph.add_edge("validator", END)
    return graph.compile()


def run_agent(
    question: str, workspace: Workspace, llm: LLM, max_llm_calls: int = 8
) -> AgentResult:
    """Answer a question from the workspace's files. Raises DocQAError for user-facing failures."""
    question = validate_question(question)
    if workspace.is_empty():
        raise DocQAError("Please upload at least one file first.")
    budget = BudgetedLLM(llm, max_llm_calls)
    state = build_graph(AgentContext(budget, workspace)).invoke(
        {"question": question, "trace": [], "rewrites": 0}
    )
    return AgentResult(
        answer=state["final"],
        sources=state.get("sources", []),
        trace=state["trace"],
        llm_calls=budget.calls,
    )
