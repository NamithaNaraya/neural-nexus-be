"""
Graph Service - Neo4j GDS Integration

Handles advanced graph algorithms like FastRP for node embeddings.
"""
import logging
from app.db.connections import get_neo4j_driver

logger = logging.getLogger(__name__)

class GraphService:
    """Service for advanced Neo4j Graph Data Science (GDS) operations."""
    
    def __init__(self):
        self.driver = get_neo4j_driver()

    async def run_fastrp_node_embeddings(self, folder_id: str):
        """
        Run FastRP algorithm to generate structural node embeddings.
        
        This generates embeddings based on the graph topology within a folder.
        """
        logger.info(f"Running FastRP for folder {folder_id}")
        
        # We'll project a graph, run FastRP, and write results back to nodes
        projection_name = f"graph_{folder_id.replace('-', '_')}"
        
        async with self.driver.session() as session:
            try:
                # 1. Clean up old projection if it exists
                await session.run(f"CALL gds.graph.drop('{projection_name}', false)")
                
                # 2. Create projection (Entity nodes and any relationships)
                # We filter by folder_id to keep structural context relevant
                await session.run("""
                    CALL gds.graph.project(
                        $projection_name,
                        ['Entity'],
                        {
                            RELATED_TO: {orientation: 'UNDIRECTED'},
                            PART_OF: {orientation: 'UNDIRECTED'},
                            LOCATED_IN: {orientation: 'UNDIRECTED'}
                        },
                        {
                            nodeProperties: { folder_id: $folder_id }
                        }
                    )
                """, projection_name=projection_name, folder_id=folder_id)
                
                # 3. Run FastRP
                # embeddingDimension 256 is a good balance for structural info
                await session.run(f"""
                    CALL gds.fastRP.write(
                        '{projection_name}',
                        {{
                            writeProperty: 'fastrp_embedding',
                            embeddingDimension: 256,
                            randomSeed: 42
                        }}
                    )
                """)
                
                logger.info(f"FastRP completed for folder {folder_id}")
                return True
                
            except Exception as e:
                error_str = str(e)
                if "ProcedureNotFound" in error_str and "gds" in error_str:
                    logger.warning("Neo4j GDS plugin not detected. Skipping FastRP embeddings.")
                else:
                    logger.error(f"Failed to run FastRP: {e}")
                # Don't raise, as this is an enrichment step
                return False
            finally:
                # Clean up projection
                try:
                    await session.run(f"CALL gds.graph.drop('{projection_name}', false)")
                except:
                    pass

_graph_service = None

def get_graph_service() -> GraphService:
    global _graph_service
    if _graph_service is None:
        _graph_service = GraphService()
    return _graph_service
