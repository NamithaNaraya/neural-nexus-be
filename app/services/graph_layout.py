"""
Graph Layout Service

Provides server-side layout calculations using Neo4j GDS algorithms.
Used to pre-calculate node positions for large graphs.
"""
import logging
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class LayoutPosition:
    """Node position in 3D space."""
    x: float
    y: float
    z: float


class GraphLayoutService:
    """
    Server-side graph layout service.
    
    Uses Neo4j GDS algorithms when available, with fallback to
    deterministic positioning based on node properties.
    """
    
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
        self._gds_available: Optional[bool] = None
    
    async def check_gds_availability(self) -> bool:
        """Check if GDS is installed and available."""
        if self._gds_available is not None:
            return self._gds_available
        
        try:
            result = await self.driver.execute_query(
                "CALL gds.version() YIELD version RETURN version"
            )
            if result.records:
                logger.info(f"GDS version: {result.records[0]['version']}")
                self._gds_available = True
            else:
                self._gds_available = False
        except Exception as e:
            logger.warning(f"GDS not available: {e}")
            self._gds_available = False
        
        return self._gds_available
    
    async def calculate_layout(
        self,
        folder_id: str,
        algorithm: str = "forceAtlas2",
        iterations: int = 100,
        scale: float = 500.0,
    ) -> Dict[str, LayoutPosition]:
        """
        Calculate layout for nodes in a folder with 'Floating Island' offset.
        """
        # Calculate base positions
        if await self.check_gds_availability():
            positions = await self._calculate_gds_layout(
                folder_id, algorithm, iterations, scale
            )
        else:
            positions = await self._calculate_fallback_layout(folder_id, scale)
            
        # Apply 'Floating Island' offset based on folder_id
        # This ensures that different folders appear in different regions of the graph
        offset = self._get_folder_offset(folder_id, scale * 3)
        
        for node_id in positions:
            pos = positions[node_id]
            positions[node_id] = LayoutPosition(
                x=pos.x + offset[0],
                y=pos.y + offset[1],
                z=pos.z + offset[2]
            )
            
        return positions

    def _get_folder_offset(self, folder_id: str, island_gap: float) -> Tuple[float, float, float]:
        """Generate a deterministic 3D offset for a folder's 'island'."""
        hash_bytes = hashlib.md5(folder_id.encode()).digest()
        
        # Use parts of the hash to create distinct X, Y, Z offsets
        x_val = (int.from_bytes(hash_bytes[0:4], 'big') / (2**32)) - 0.5
        y_val = (int.from_bytes(hash_bytes[4:8], 'big') / (2**32)) - 0.5
        z_val = (int.from_bytes(hash_bytes[8:12], 'big') / (2**32)) - 0.5
        
        return (x_val * island_gap, y_val * island_gap, z_val * island_gap)
    
    async def _calculate_gds_layout(
        self,
        folder_id: str,
        algorithm: str,
        iterations: int,
        scale: float,
    ) -> Dict[str, LayoutPosition]:
        """Calculate layout using Neo4j GDS."""
        graph_name = f"layout_{folder_id.replace('-', '_')}"
        positions: Dict[str, LayoutPosition] = {}
        
        try:
            # Create a projected graph for GDS
            projection_query = """
            CALL gds.graph.project.cypher(
                $graph_name,
                'MATCH (n) WHERE n.folder_id = $folder_id RETURN id(n) AS id',
                'MATCH (a)-[r]-(b) WHERE a.folder_id = $folder_id AND b.folder_id = $folder_id 
                 RETURN id(a) AS source, id(b) AS target'
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
            """
            
            await self.driver.execute_query(projection_query, {
                "graph_name": graph_name,
                "folder_id": folder_id,
            })
            
            # Run ForceAtlas2 or fallback to FR layout
            if algorithm == "forceAtlas2":
                layout_query = """
                CALL gds.alpha.forceAtlas2.stream($graph_name, {
                    maxIterations: $iterations,
                    outboundAttractionDistribution: true,
                    scalingRatio: 2.0
                })
                YIELD nodeId, x, y
                MATCH (n) WHERE id(n) = nodeId
                RETURN n.id AS nodeId, n.entity_id AS entityId, x, y
                """
            else:
                # Fruchterman-Reingold fallback
                layout_query = """
                CALL gds.alpha.randomWalk.stream($graph_name, {
                    walkLength: 10,
                    walksPerNode: 5
                })
                YIELD nodeIds
                UNWIND nodeIds AS nodeId
                MATCH (n) WHERE id(n) = nodeId
                RETURN DISTINCT n.id AS nodeId, n.entity_id AS entityId,
                       rand() * $scale AS x, rand() * $scale AS y
                """
            
            result = await self.driver.execute_query(layout_query, {
                "graph_name": graph_name,
                "iterations": iterations,
                "scale": scale,
            })
            
            for record in result.records:
                node_id = record["nodeId"] or record["entityId"]
                if node_id:
                    # Scale and convert to 3D (z based on hash for consistency)
                    z = self._deterministic_z(node_id, scale)
                    positions[node_id] = LayoutPosition(
                        x=record["x"] * scale,
                        y=record["y"] * scale,
                        z=z
                    )
            
        except Exception as e:
            logger.error(f"GDS layout failed: {e}")
            # Fall back to deterministic layout
            positions = await self._calculate_fallback_layout(folder_id, scale)
        
        finally:
            # Cleanup projected graph
            try:
                await self.driver.execute_query(
                    "CALL gds.graph.drop($graph_name)",
                    {"graph_name": graph_name}
                )
            except:
                pass
        
        return positions
    
    async def _calculate_fallback_layout(
        self,
        folder_id: str,
        scale: float,
    ) -> Dict[str, LayoutPosition]:
        """
        Fallback deterministic layout when GDS is not available.
        Uses a combination of node type clustering and hash-based positioning.
        """
        positions: Dict[str, LayoutPosition] = {}
        
        try:
            # Fetch nodes with their types and degrees
            query = """
            MATCH (n)
            WHERE n.folder_id = $folder_id OR n.folderId = $folder_id
            OPTIONAL MATCH (n)-[r]-()
            WITH n, count(DISTINCT r) as degree
            RETURN n.id AS nodeId, n.entity_id AS entityId, 
                   n.type AS type, COALESCE(n.name, '') AS name, degree
            """
            
            result = await self.driver.execute_query(query, {"folder_id": folder_id})
            
            # Group nodes by type for clustering
            type_groups: Dict[str, List[Tuple[str, str, int]]] = {}
            
            for record in result.records:
                node_id = record["nodeId"] or record["entityId"]
                node_type = record["type"] or "Unknown"
                node_name = record["name"] or ""
                degree = record["degree"] or 0
                
                if node_type not in type_groups:
                    type_groups[node_type] = []
                type_groups[node_type].append((node_id, node_name, degree))
            
            # Calculate type cluster centers (arranged in a circle)
            num_types = len(type_groups)
            type_centers: Dict[str, Tuple[float, float]] = {}
            
            for i, node_type in enumerate(type_groups.keys()):
                angle = (i / num_types) * 2 * math.pi
                center_x = math.cos(angle) * scale * 0.5
                center_y = math.sin(angle) * scale * 0.5
                type_centers[node_type] = (center_x, center_y)
            
            # Position nodes within their type clusters
            for node_type, nodes in type_groups.items():
                center_x, center_y = type_centers[node_type]
                cluster_radius = min(scale * 0.3, len(nodes) * 10)
                
                for j, (node_id, node_name, degree) in enumerate(nodes):
                    if not node_id:
                        continue
                    
                    # Spiral placement within cluster
                    angle = (j / max(len(nodes), 1)) * 2 * math.pi * 3  # 3 rotations
                    radius = (j / max(len(nodes), 1)) * cluster_radius
                    
                    # Add deterministic jitter based on node name
                    jitter = self._hash_to_float(node_name) * 20
                    
                    x = center_x + math.cos(angle) * radius + jitter
                    y = center_y + math.sin(angle) * radius + jitter
                    z = self._deterministic_z(node_id, scale)
                    
                    # High-degree nodes closer to center
                    if degree > 5:
                        x = x * 0.7 + center_x * 0.3
                        y = y * 0.7 + center_y * 0.3
                    
                    positions[node_id] = LayoutPosition(x=x, y=y, z=z)
            
        except Exception as e:
            logger.error(f"Fallback layout failed: {e}")
        
        return positions
    
    def _deterministic_z(self, node_id: str, scale: float) -> float:
        """Generate deterministic Z coordinate based on node ID."""
        hash_bytes = hashlib.md5(node_id.encode()).digest()
        # Use first 4 bytes as float between -scale/2 and scale/2
        int_value = int.from_bytes(hash_bytes[:4], 'big')
        normalized = (int_value / (2**32)) - 0.5
        return normalized * scale * 0.3
    
    def _hash_to_float(self, text: str) -> float:
        """Convert text to a deterministic float between -1 and 1."""
        hash_bytes = hashlib.md5(text.encode()).digest()
        int_value = int.from_bytes(hash_bytes[:4], 'big')
        return (int_value / (2**32)) * 2 - 1


# Singleton instance
_layout_service: Optional[GraphLayoutService] = None


def get_layout_service(neo4j) -> GraphLayoutService:
    """Get or create the layout service singleton."""
    global _layout_service
    if _layout_service is None:
        _layout_service = GraphLayoutService(neo4j)
    return _layout_service
