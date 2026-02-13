import logging
from typing import List, Dict, Any
from app.db.connections import get_neo4j_driver

logger = logging.getLogger(__name__)

class GraphReasoningService:
    """
    Step 6: Graph Intelligence Layer
    Performs deterministic reasoning using graph traversals and pattern expansion.
    """
    
    def __init__(self):
        self.driver = get_neo4j_driver()

    async def find_interventions(self, inferred_states: List[Dict[str, Any]], folder_id: str) -> List[Dict[str, Any]]:
        """
        Query the KG to find interventions that modulate the inferred states.
        Implements: State -> [OPPOSED_BY/MODULATED_BY] -> Quality/Property <- [HAS_QUALITY] - Intervention
        """
        state_names = [s["state"] for s in inferred_states]
        
        query = """
        UNWIND $state_names AS stateName
        MATCH (s:State {folder_id: $folder_id})
        WHERE s.name CONTAINS stateName OR stateName CONTAINS s.name
        
        // Expansion: Find opposing/balancing properties
        MATCH (s)-[r:OPPOSED_BY|MODULATED_BY|REQUIRES_QUALITY]->(p:Property)
        
        // Expansion: Find interventions with those properties
        MATCH (iv:Intervention)-[:HAS_QUALITY|EXPRESSES_KARMA]->(p)
        WHERE iv.folder_id = $folder_id
        
        RETURN 
            s.name AS source_state,
            p.name AS target_property,
            iv.name AS intervention,
            iv.type AS intervention_type,
            labels(iv) AS labels,
            count(*) AS strength
        ORDER BY strength DESC
        LIMIT 10
        """
        
        try:
            async with self.driver.session() as session:
                result = await session.run(query, state_names=state_names, folder_id=folder_id)
                records = await result.data()
                logger.info(f"Graph reasoning found {len(records)} intervention paths.")
                return records
        except Exception as e:
            logger.error(f"Graph reasoning failed: {e}")
            return []

    async def trace_outcome_path(self, intervention_name: str, folder_id: str) -> List[Dict[str, Any]]:
        """
        Trace the path from an intervention to its recorded outcomes.
        Intervention -> [PRODUCES/LEADS_TO] -> Outcome
        """
        query = """
        MATCH (iv:Intervention {name: $name, folder_id: $folder_id})
        OPTIONAL MATCH (iv)-[:PRODUCES|LEADS_TO|ENABLES_KARMA*1..3]->(o:Outcome)
        RETURN iv.name AS name, collect(DISTINCT o.name) AS outcomes
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(query, name=intervention_name, folder_id=folder_id)
                record = await result.single()
                return record.data() if record else {}
        except Exception as e:
            logger.error(f"Outcome tracing failed: {e}")
            return {}
