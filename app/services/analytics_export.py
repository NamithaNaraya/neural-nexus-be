"""
Analytics Export Service

Generates PDF and JSON exports with analytics supplement tables.
Includes centrality, clustering, and other graph metrics.
"""
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
import json

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for analytics export."""
    include_centrality: bool = True
    include_clustering: bool = True
    include_degree_distribution: bool = True
    include_ghost_lines: bool = True
    include_health_score: bool = True
    format: str = "pdf"  # pdf, json, csv


class AnalyticsExportService:
    """
    Service for exporting graph analytics to various formats.
    
    Features:
    - PDF reports with analytics supplement
    - JSON data exports
    - CSV for spreadsheet analysis
    """
    
    def __init__(self, neo4j_driver):
        self.neo4j = neo4j_driver
    
    async def export(
        self,
        folder_id: str,
        config: Optional[ExportConfig] = None,
    ) -> Dict[str, Any]:
        """
        Generate an analytics export for a folder.
        
        Returns:
            Dict with export data and format info
        """
        if config is None:
            config = ExportConfig()
        
        try:
            # Gather all analytics data
            analytics_data = await self._gather_analytics(folder_id, config)
            
            # Generate export based on format
            if config.format == "json":
                return self._export_json(analytics_data, folder_id)
            elif config.format == "csv":
                return self._export_csv(analytics_data, folder_id)
            else:  # PDF
                return await self._export_pdf(analytics_data, folder_id)
                
        except Exception as e:
            logger.error(f"Analytics export failed: {e}")
            return {
                "error": str(e),
                "folder_id": folder_id,
            }
    
    async def _gather_analytics(
        self,
        folder_id: str,
        config: ExportConfig,
    ) -> Dict[str, Any]:
        """Gather all requested analytics data."""
        data = {
            "folder_id": folder_id,
            "generated_at": datetime.utcnow().isoformat(),
            "sections": {},
        }
        
        # Basic graph stats
        data["overview"] = await self._get_overview(folder_id)
        
        # Centrality metrics
        if config.include_centrality:
            data["sections"]["centrality"] = await self._get_centrality_data(folder_id)
        
        # Clustering/community data
        if config.include_clustering:
            data["sections"]["clustering"] = await self._get_clustering_data(folder_id)
        
        # Degree distribution
        if config.include_degree_distribution:
            data["sections"]["degree_distribution"] = await self._get_degree_distribution(folder_id)
        
        # Ghost lines (blind spots)
        if config.include_ghost_lines:
            data["sections"]["blind_spots"] = await self._get_blind_spots(folder_id)
        
        # Health score
        if config.include_health_score:
            data["sections"]["health"] = await self._get_health_score(folder_id)
        
        return data
    
    async def _get_overview(self, folder_id: str) -> Dict[str, Any]:
        """Get basic graph statistics."""
        query = """
        MATCH (n)
        WHERE n.folder_id = $folder_id OR n.folderId = $folder_id
        WITH count(n) as nodeCount
        MATCH (a)-[r]->(b)
        WHERE (a.folder_id = $folder_id OR a.folderId = $folder_id)
          AND (b.folder_id = $folder_id OR b.folderId = $folder_id)
        WITH nodeCount, count(r) as linkCount
        RETURN nodeCount, linkCount
        """
        
        try:
            result = self.neo4j.execute_query(query, {"folder_id": folder_id})
            
            if result.records:
                record = result.records[0]
                node_count = record["nodeCount"]
                link_count = record["linkCount"]
                density = (2 * link_count) / (node_count * (node_count - 1)) if node_count > 1 else 0
                
                return {
                    "node_count": node_count,
                    "link_count": link_count,
                    "density": round(density, 4),
                    "avg_degree": round(2 * link_count / node_count, 2) if node_count > 0 else 0,
                }
        except Exception as e:
            logger.error(f"Overview query failed: {e}")
        
        return {"node_count": 0, "link_count": 0, "density": 0, "avg_degree": 0}
    
    async def _get_centrality_data(self, folder_id: str) -> Dict[str, Any]:
        """Get top nodes by various centrality metrics."""
        # Degree centrality (simplified)
        degree_query = """
        MATCH (n)
        WHERE n.folder_id = $folder_id OR n.folderId = $folder_id
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) as degree
        ORDER BY degree DESC
        LIMIT 10
        RETURN 
            COALESCE(n.id, elementId(n)) as id,
            n.name as name,
            n.type as type,
            degree
        """
        
        try:
            result = self.neo4j.execute_query(degree_query, {"folder_id": folder_id})
            
            degree_top = [
                {
                    "id": r["id"],
                    "name": r["name"] or r["id"],
                    "type": r["type"] or "Unknown",
                    "score": r["degree"],
                }
                for r in result.records
            ]
            
            return {
                "degree_centrality": {
                    "description": "Nodes with the most connections",
                    "top_nodes": degree_top,
                },
                "betweenness_centrality": {
                    "description": "Nodes that act as bridges (requires GDS)",
                    "top_nodes": [],
                    "note": "Enable Neo4j GDS for accurate betweenness calculation",
                },
                "pagerank": {
                    "description": "Most influential nodes (requires GDS)",
                    "top_nodes": [],
                    "note": "Enable Neo4j GDS for PageRank calculation",
                },
            }
        except Exception as e:
            logger.error(f"Centrality query failed: {e}")
            return {}
    
    async def _get_clustering_data(self, folder_id: str) -> Dict[str, Any]:
        """Get community/clustering information."""
        # Get node type distribution as a proxy for clustering
        type_query = """
        MATCH (n)
        WHERE n.folder_id = $folder_id OR n.folderId = $folder_id
        RETURN n.type as type, count(*) as count
        ORDER BY count DESC
        """
        
        try:
            result = self.neo4j.execute_query(type_query, {"folder_id": folder_id})
            
            type_distribution = [
                {"type": r["type"] or "Unknown", "count": r["count"]}
                for r in result.records
            ]
            
            return {
                "type_distribution": type_distribution,
                "community_detection": {
                    "algorithm": "Louvain (requires GDS)",
                    "communities": [],
                    "note": "Enable Neo4j GDS for community detection",
                },
            }
        except Exception as e:
            logger.error(f"Clustering query failed: {e}")
            return {}
    
    async def _get_degree_distribution(self, folder_id: str) -> Dict[str, Any]:
        """Get degree distribution statistics."""
        query = """
        MATCH (n)
        WHERE n.folder_id = $folder_id OR n.folderId = $folder_id
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) as degree
        RETURN 
            min(degree) as min_degree,
            max(degree) as max_degree,
            avg(degree) as avg_degree,
            percentileCont(degree, 0.5) as median_degree,
            percentileCont(degree, 0.9) as p90_degree,
            stdev(degree) as std_degree
        """
        
        try:
            result = self.neo4j.execute_query(query, {"folder_id": folder_id})
            
            if result.records:
                r = result.records[0]
                return {
                    "min": r["min_degree"],
                    "max": r["max_degree"],
                    "mean": round(r["avg_degree"], 2),
                    "median": r["median_degree"],
                    "p90": r["p90_degree"],
                    "std_dev": round(r["std_degree"], 2) if r["std_degree"] else 0,
                }
        except Exception as e:
            logger.error(f"Degree distribution query failed: {e}")
        
        return {}
    
    async def _get_blind_spots(self, folder_id: str) -> Dict[str, Any]:
        """Get predicted missing relationships."""
        from app.services.blind_spot_discovery import get_discovery_service
        
        try:
            discovery = get_discovery_service(self.neo4j)
            result = await discovery.discover(folder_id=folder_id, limit=20)
            
            return {
                "count": result.get("total_count", 0),
                "top_predictions": result.get("ghost_lines", [])[:10],
                "by_category": result.get("by_category", {}),
            }
        except Exception as e:
            logger.error(f"Blind spots query failed: {e}")
            return {"count": 0, "top_predictions": []}
    
    async def _get_health_score(self, folder_id: str) -> Dict[str, Any]:
        """Get graph health assessment."""
        overview = await self._get_overview(folder_id)
        
        # Simple health scoring
        score = 100.0
        issues = []
        
        node_count = overview.get("node_count", 0)
        link_count = overview.get("link_count", 0)
        density = overview.get("density", 0)
        
        # Check for isolated nodes
        if node_count > 0 and link_count == 0:
            score -= 30
            issues.append("No relationships exist in the graph")
        elif density < 0.01 and node_count > 10:
            score -= 15
            issues.append("Graph density is very low - many isolated nodes")
        
        # Check for minimum size
        if node_count < 5:
            score -= 20
            issues.append("Very few nodes in the graph")
        
        # Check for relationship diversity
        rel_query = """
        MATCH (a)-[r]->(b)
        WHERE (a.folder_id = $folder_id OR a.folderId = $folder_id)
        RETURN count(DISTINCT type(r)) as rel_types
        """
        
        try:
            result = self.neo4j.execute_query(rel_query, {"folder_id": folder_id})
            if result.records:
                rel_types = result.records[0]["rel_types"]
                if rel_types == 1 and link_count > 10:
                    score -= 10
                    issues.append("Only one relationship type - consider adding diversity")
        except:
            pass
        
        return {
            "score": max(0, min(100, score)),
            "grade": self._score_to_grade(score),
            "issues": issues,
            "metrics": overview,
        }
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _export_json(
        self,
        data: Dict[str, Any],
        folder_id: str,
    ) -> Dict[str, Any]:
        """Export as JSON."""
        return {
            "format": "json",
            "folder_id": folder_id,
            "filename": f"analytics_{folder_id}_{datetime.utcnow().strftime('%Y%m%d')}.json",
            "data": data,
            "content_type": "application/json",
        }
    
    def _export_csv(
        self,
        data: Dict[str, Any],
        folder_id: str,
    ) -> Dict[str, Any]:
        """Export as CSV (overview table)."""
        lines = ["Metric,Value"]
        
        # Overview
        overview = data.get("overview", {})
        lines.append(f"Node Count,{overview.get('node_count', 0)}")
        lines.append(f"Link Count,{overview.get('link_count', 0)}")
        lines.append(f"Density,{overview.get('density', 0)}")
        lines.append(f"Average Degree,{overview.get('avg_degree', 0)}")
        
        # Health
        health = data.get("sections", {}).get("health", {})
        lines.append(f"Health Score,{health.get('score', 0)}")
        lines.append(f"Health Grade,{health.get('grade', 'N/A')}")
        
        csv_content = "\n".join(lines)
        
        return {
            "format": "csv",
            "folder_id": folder_id,
            "filename": f"analytics_{folder_id}_{datetime.utcnow().strftime('%Y%m%d')}.csv",
            "content": csv_content,
            "content_type": "text/csv",
        }
    
    async def _export_pdf(
        self,
        data: Dict[str, Any],
        folder_id: str,
    ) -> Dict[str, Any]:
        """
        Export as PDF.
        
        Note: Full PDF generation requires reportlab or similar.
        This returns structured data for frontend PDF generation.
        """
        # Structure for frontend PDF rendering
        pdf_structure = {
            "title": f"Knowledge Graph Analytics Report",
            "subtitle": f"Folder: {folder_id}",
            "generated_at": data.get("generated_at"),
            "sections": [
                {
                    "title": "Overview",
                    "type": "metrics",
                    "data": data.get("overview", {}),
                },
            ],
        }
        
        # Add centrality section
        centrality = data.get("sections", {}).get("centrality", {})
        if centrality:
            pdf_structure["sections"].append({
                "title": "Centrality Analysis",
                "type": "table",
                "description": "Top nodes by degree centrality",
                "columns": ["Rank", "Name", "Type", "Score"],
                "rows": [
                    [i + 1, n["name"], n["type"], n["score"]]
                    for i, n in enumerate(centrality.get("degree_centrality", {}).get("top_nodes", []))
                ],
            })
        
        # Add clustering section
        clustering = data.get("sections", {}).get("clustering", {})
        if clustering:
            pdf_structure["sections"].append({
                "title": "Entity Type Distribution",
                "type": "chart_data",
                "chart_type": "pie",
                "data": clustering.get("type_distribution", []),
            })
        
        # Add blind spots
        blind_spots = data.get("sections", {}).get("blind_spots", {})
        if blind_spots and blind_spots.get("top_predictions"):
            pdf_structure["sections"].append({
                "title": "Predicted Missing Relationships (Blind Spots)",
                "type": "table",
                "description": f"Found {blind_spots.get('count', 0)} potential missing connections",
                "columns": ["Source", "Target", "Confidence", "Reason"],
                "rows": [
                    [p["source_name"], p["target_name"], f"{p['confidence']:.0%}", p["reason"]]
                    for p in blind_spots.get("top_predictions", [])[:10]
                ],
            })
        
        # Add health score
        health = data.get("sections", {}).get("health", {})
        if health:
            pdf_structure["sections"].append({
                "title": "Graph Health Assessment",
                "type": "health_score",
                "score": health.get("score", 0),
                "grade": health.get("grade", "N/A"),
                "issues": health.get("issues", []),
            })
        
        return {
            "format": "pdf",
            "folder_id": folder_id,
            "filename": f"analytics_{folder_id}_{datetime.utcnow().strftime('%Y%m%d')}.pdf",
            "structure": pdf_structure,
            "raw_data": data,
            "content_type": "application/json",  # Frontend will render the PDF
        }


# Singleton
_export_service: Optional[AnalyticsExportService] = None


def get_export_service(neo4j) -> AnalyticsExportService:
    """Get or create the export service singleton."""
    global _export_service
    if _export_service is None:
        _export_service = AnalyticsExportService(neo4j)
    return _export_service
