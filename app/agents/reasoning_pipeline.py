import logging
import json
import operator
from typing import Dict, Any, List, Optional, Annotated, TypedDict
from langgraph.graph import StateGraph, END

from app.agents.safety_agent import SafetyAgent
from app.agents.inference_engine import InferenceEngine
from app.agents.recommendation_engine import RecommendationEngine
from app.agents.graph_analytics_agent import GraphAnalyticsAgent
from app.services.graph_reasoning_service import GraphReasoningService
from app.services.hybrid_rag import HybridRAGService
from app.services.ai_service import get_ollama_service
from app.db.connections import get_neo4j_driver

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
    extracted_indicators: Annotated[List[Dict[str, Any]], operator.add]
    safety_report: Dict[str, Any]
    inferred_states: Annotated[List[Dict[str, Any]], operator.add]
    graph_analytics: Optional[Dict[str, Any]]
    graph_interventions: Annotated[List[Dict[str, Any]], operator.add]
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
        
        # Initialize HybridRAGService for real knowledge retrieval
        neo4j_driver = get_neo4j_driver()
        self.rag_service = HybridRAGService(neo4j_driver, ai_service)
        
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
        
        # Define Edges
        graph.set_entry_point("extraction")
        
        # Fan out from extraction: safety chain + analytics in parallel
        graph.add_edge("extraction", "safety_check")
        graph.add_edge("extraction", "graph_analytics")
        
        # Safety chain continues sequentially
        graph.add_edge("safety_check", "state_inference")
        graph.add_edge("state_inference", "graph_reasoning")
        
        # Merge point: both parallel branches must complete before continuing
        graph.add_node("parallel_merge", lambda x: {})
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
        """Step 8: RAG Retrieval using HybridRAGService."""
        try:
            # Use the original user question for RAG - this is critical
            # for knowledge queries like "What are the symptoms of menopause?"
            query = state["user_input"]
            
            # If we have graph interventions, enrich the query with them
            if state["graph_interventions"]:
                intervention_names = ", ".join(
                    [i["intervention"] for i in state["graph_interventions"][:3]]
                )
                query = f"{state['user_input']} (related interventions: {intervention_names})"
            
            # Build scope from folder_id
            scope = {"folder_id": state["folder_id"]} if state.get("folder_id") else None
            
            rag_result = await self.rag_service.query(
                question=query,
                session_id=state["session_id"],
                scope=scope
            )
            
            return {
                "rag_context": {
                    "answer": rag_result.get("answer", ""),
                    "citations": rag_result.get("citations", []),
                    "related_nodes": rag_result.get("related_nodes", []),
                }
            }
        except Exception as e:
            logger.error(f"RAG enrichment failed: {e}")
            return {"rag_context": {"answer": "", "citations": [], "error": str(e)}}

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
            # Extract the RAG answer if available
            rag_answer = ""
            rag_citations = []
            if isinstance(state.get("rag_context"), dict):
                rag_answer = state["rag_context"].get("answer", "")
                rag_citations = state["rag_context"].get("citations", [])
            
            prompt = f"""You are a knowledgeable assistant. The user asked: "{state['user_input']}"

Below is all the data gathered from the knowledge graph and RAG system. Use it to compose a comprehensive, accurate response.

--- RETRIEVED DATA ---
RAG Answer: {rag_answer}
RAG Citations: {rag_citations}
Extracted Indicators: {state['extracted_indicators']}
Inferred States: {state['inferred_states']}
Graph Analytics (Structural Insights): {state['graph_analytics']}
Graph Interventions: {state['graph_interventions']}
Recommendation: {state['recommendation']}
--- END DATA ---

### RESPONSE RULES:
1. **PRIORITIZE RAG ANSWER**: If the RAG Answer contains relevant information, use it as the primary basis for your response.
2. **SUPPLEMENT WITH GRAPH DATA**: Enrich the response with Graph Interventions, Analytics, and Inferred States where available.
3. **NO HALLUCINATIONS**: Only report information found in the provided data above.
4. **GROUNDING**: If the provided data is insufficient to fully answer the user's query, state what information is available and what is not found in the current knowledge graph.
5. **SAFETY**: Be professional, empathetic, and always include safety precautions where relevant.
6. **CITATIONS**: Cite references from the RAG citations for claims when available.
7. **EXPLAINING ANALYTICS**: If structural analytics (PageRank, Centrality, etc.) are present, explain their significance based ONLY on the provided results.
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
