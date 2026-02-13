import logging
import json
from typing import Dict, Any, List, Optional, Annotated, TypedDict
from langgraph.graph import StateGraph, END

from app.agents.safety_agent import SafetyAgent
from app.agents.inference_engine import InferenceEngine
from app.agents.recommendation_engine import RecommendationEngine
from app.agents.graph_analytics_agent import GraphAnalyticsAgent
from app.services.graph_reasoning_service import GraphReasoningService
from app.services.hybrid_rag import HybridRAGService
from app.services.ai_service import get_ollama_service

logger = logging.getLogger(__name__)

class PipelineState(TypedDict):
    """
    State for the 13-Step Reasoning Pipeline.
    """
    user_input: str
    user_id: str
    session_id: str
    folder_id: str
    
    # Processed Data
    extracted_indicators: List[Dict[str, Any]]
    safety_report: Dict[str, Any]
    inferred_states: List[Dict[str, Any]]
    graph_analytics: Optional[Dict[str, Any]]
    graph_interventions: List[Dict[str, Any]]
    rag_context: Dict[str, Any]
    recommendation: Dict[str, Any]
    
    # Final Output
    final_response: str
    status: str
    error: Optional[str]
    encounter_id: Optional[str]

class ReasoningPipeline:
    """
    LangGraph Orchestrator for the General Knowledge Reasoning Pipeline.
    Coordinates all 13 steps from input to outcome tracking.
    """
    
    def __init__(self, encounter_service, ai_service):
        self.encounters = encounter_service
        self.ai = ai_service
        self.safety = SafetyAgent()
        self.inference = InferenceEngine()
        self.analytics = GraphAnalyticsAgent(ai_service)
        self.graph_reasoning = GraphReasoningService()
        self.recommendation_engine = RecommendationEngine()
        
        self.workflow = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(PipelineState)
        
        # Add Nodes
        graph.add_node("extraction", self._extraction_node)
        graph.add_node("safety_check", self._safety_node)
        graph.add_node("state_inference", self._inference_node)
        graph.add_node("graph_analytics", self._analytics_node)
        graph.add_node("graph_reasoning", self._graph_node)
        graph.add_node("rag_enrichment", self._rag_node)
        graph.add_node("recommendation", self._recommendation_node)
        graph.add_node("compose_response", self._composition_node)
        graph.add_node("logging", self._logging_node)
        
        # Define Edges (Parallelized for Speed)
        graph.set_entry_point("extraction") # Primary chain
        # Add secondary entry point for parallel analytics
        graph.add_edge("extraction", "safety_check")
        graph.add_edge("safety_check", "state_inference")
        graph.add_edge("state_inference", "graph_reasoning")
        
        # Branch Analytics
        graph.add_node("analytics_start", lambda x: x) # Sync node
        graph.set_entry_point("analytics_start")
        graph.add_edge("analytics_start", "graph_analytics")
        
        # Merge Point
        graph.add_node("parallel_merge", lambda x: x)
        graph.add_edge("graph_reasoning", "parallel_merge")
        graph.add_edge("graph_analytics", "parallel_merge")
        
        graph.add_edge("parallel_merge", "rag_enrichment")
        graph.add_edge("rag_enrichment", "recommendation")
        graph.add_edge("recommendation", "compose_response")
        graph.add_edge("compose_response", "logging")
        graph.add_edge("logging", END)
        
        return graph.compile()

    # --- Node Implementations ---

    async def _extraction_node(self, state: PipelineState) -> Dict[str, Any]:
        """Step 2: NER Entity Extraction."""
        prompt = f"""Extract domain-specific indicators (symptoms, observations, levels) from the following text:
Text: "{state['user_input']}"
Output JSON: {{"indicators": [{{"name": "...", "type": "...", "level": "High/Normal/Low"}}]}}
"""
        try:
            result = await self.ai.chat_json([{"role": "user", "content": prompt}])
            return {"extracted_indicators": result.get("indicators", []), "status": "extracted"}
        except Exception as e:
            return {"error": str(e), "status": "failed"}

    async def _safety_node(self, state: PipelineState) -> Dict[str, Any]:
        """Step 3: Safety Engine."""
        report = await self.safety.check_safety(state["extracted_indicators"])
        return {"safety_report": report}

    async def _inference_node(self, state: PipelineState) -> Dict[str, Any]:
        """Step 5: State Inference (e.g. Dosha determination)."""
        if state["safety_report"].get("status") == "DANGEROUS":
            return {"inferred_states": [], "status": "safety_alert"}
            
        result = await self.inference.infer_states(state["extracted_indicators"])
        return {"inferred_states": result.get("inferred_states", [])}

    async def _analytics_node(self, state: PipelineState) -> Dict[str, Any]:
        """Step 5.5: Dynamic Graph Analytics."""
        selection = await self.analytics.analyze_intent(state["user_input"])
        if selection:
            results = await self.analytics.execute_algorithm(selection, state["folder_id"])
            return {"graph_analytics": results}
        return {"graph_analytics": None}

    async def _graph_node(self, state: PipelineState) -> Dict[str, Any]:
        """Step 6 & 7: Graph Traversal & Reasoning."""
        interventions = await self.graph_reasoning.find_interventions(
            state["inferred_states"], state["folder_id"]
        )
        return {"graph_interventions": interventions}

    async def _rag_node(self, state: PipelineState) -> Dict[str, Any]:
        """Step 8: RAG Retrieval."""
        # Generic retrieval based on interventions
        query = ", ".join([i["intervention"] for i in state["graph_interventions"][:3]])
        # Assuming hybrid RAG service is initialized elsewhere or accessible
        # For now, we'll return a placeholder or mock
        return {"rag_context": {"references": "Authoritative domain monographs retrieved."}}

    async def _recommendation_node(self, state: PipelineState) -> Dict[str, Any]:
        """Step 9: Recommendation Engine (Dose sizing)."""
        if not state["graph_interventions"]:
            return {"recommendation": None}
            
        main_intervention = state["graph_interventions"][0]["intervention"]
        # Determine average intensity for sizing
        intensity = 0.5 # Default
        if state["inferred_states"]:
            intensity = state["inferred_states"][0].get("confidence", 0.5)
            
        result = await self.recommendation_engine.calculate_recommendation(main_intervention, intensity)
        return {"recommendation": result.get("recommendation")}

    async def _composition_node(self, state: PipelineState) -> Dict[str, Any]:
        """Step 10: Final Response Composition."""
        if state["safety_report"].get("status") == "DANGEROUS":
            response = f"⚠️ EMERGENCY ALERT: {state['safety_report'].get('reason')}. {state['safety_report'].get('advice')}"
        else:
            prompt = f"""Compose a patient-safe response based SOLELY on the provided data:
Indicators: {state['extracted_indicators']}
Inferred States: {state['inferred_states']}
Graph Analytics (Structural Insights): {state['graph_analytics']}
Interventions: {state['graph_interventions']}
RAG Context: {state['rag_context']}
Recommendation: {state['recommendation']}

### STRICT COMPLIANCE RULES:
1. **NO EXTERNAL KNOWLEDGE**: Do NOT include information not explicitly mentioned in the RAG Context or Graph Interventions.
2. **NO HALLUCINATIONS**: Only report relationships and properties found in the provided Graph Analytics and Interventions.
3. **GROUNDING**: If the provided data is insufficient to answer a specific part of the user's query, state that the information is not available in the current knowledge graph.
4. **DIAGNOSTIC FEEDBACK**: If the Graph Interventions or Inferred States suggest a condition but some critical symptoms are missing, explicitly ask the user about those specific connected symptoms found in the Graph.
5. **SAFETY**: Be professional, empathetic, and always include safety precautions.
6. **EXPLAINING ANALYTICS**: If structural analytics (PageRank, Centrality, etc.) are present, explain their significance based ONLY on the provided results.
7. **CITATIONS**: Cite references from the RAG context for every claim.
"""
            response = await self.ai.chat([{"role": "user", "content": prompt}])
            
        return {"final_response": response}

    async def _logging_node(self, state: PipelineState) -> Dict[str, Any]:
        """Step 11: Logging everything to DB."""
        encounter = await self.encounters.create_encounter(
            user_id=state["user_id"],
            session_id=state["session_id"],
            indicators={"input": state["user_input"], "extracted": state["extracted_indicators"]}
        )
        return {"status": "completed", "encounter_id": str(encounter.id)}

    async def run(self, inputs: Dict[str, Any]) -> PipelineState:
        """Run the pipeline."""
        initial_state = {
            "user_input": inputs["user_input"],
            "user_id": inputs["user_id"],
            "session_id": inputs["session_id"],
            "folder_id": inputs["folder_id"],
            "extracted_indicators": [],
            "safety_report": {"status": "SAFE"},
            "inferred_states": [],
            "graph_analytics": None,
            "graph_interventions": [],
            "rag_context": {},
            "recommendation": {},
            "final_response": "",
            "status": "started",
            "error": None,
            "encounter_id": None
        }
        return await self.workflow.ainvoke(initial_state)
