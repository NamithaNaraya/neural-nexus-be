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
            result = await self.neo4j.execute_query(query, {"folder_id": folder_id})
            
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
            result = await self.neo4j.execute_query(degree_query, {"folder_id": folder_id})
            
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
            result = await self.neo4j.execute_query(type_query, {"folder_id": folder_id})
            
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
            result = await self.neo4j.execute_query(query, {"folder_id": folder_id})
            
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
            result = await self.neo4j.execute_query(rel_query, {"folder_id": folder_id})
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
                    "title": "Executive Summary",
                    "type": "summary",
                    "data": self._build_executive_summary(data),
                },
                {
                    "title": "Graph Overview",
                    "type": "metrics_grid",
                    "columns": 4,
                    "metrics": [
                        {
                            "label": "Total Nodes",
                            "value": data.get("overview", {}).get("node_count", 0),
                            "icon": "nodes",
                        },
                        {
                            "label": "Total Relationships",
                            "value": data.get("overview", {}).get("link_count", 0),
                            "icon": "links",
                        },
                        {
                            "label": "Graph Density",
                            "value": f"{data.get('overview', {}).get('density', 0):.4f}",
                            "icon": "density",
                        },
                        {
                            "label": "Average Degree",
                            "value": data.get("overview", {}).get("avg_degree", 0),
                            "icon": "degree",
                        },
                    ],
                },
            ],
        }
        
        # Add centrality section with enhanced table
        centrality = data.get("sections", {}).get("centrality", {})
        if centrality:
            degree_nodes = centrality.get("degree_centrality", {}).get("top_nodes", [])
            pdf_structure["sections"].append({
                "title": "Centrality Analysis",
                "type": "enhanced_table",
                "description": "Top 10 most connected nodes in the knowledge graph",
                "columns": [
                    {"key": "rank", "label": "#", "width": 40, "align": "center"},
                    {"key": "name", "label": "Entity Name", "width": 200, "align": "left"},
                    {"key": "type", "label": "Type", "width": 100, "align": "center"},
                    {"key": "score", "label": "Connections", "width": 80, "align": "right"},
                    {"key": "bar", "label": "Relative", "width": 100, "type": "progress_bar"},
                ],
                "rows": [
                    {
                        "rank": i + 1,
                        "name": n["name"],
                        "type": n["type"],
                        "score": n["score"],
                        "bar": n["score"] / max(1, degree_nodes[0]["score"] if degree_nodes else 1) * 100,
                    }
                    for i, n in enumerate(degree_nodes)
                ],
                "footer": f"Showing top {len(degree_nodes)} nodes by degree centrality",
            })
        
        # Add entity type distribution pie chart
        clustering = data.get("sections", {}).get("clustering", {})
        if clustering:
            type_dist = clustering.get("type_distribution", [])
            pdf_structure["sections"].append({
                "title": "Entity Type Distribution",
                "type": "pie_chart",
                "description": "Distribution of entities across different types",
                "data": [
                    {"label": t["type"], "value": t["count"]}
                    for t in type_dist
                ],
                "total": sum(t["count"] for t in type_dist),
            })
        
        # Add degree distribution histogram
        degree_dist = data.get("sections", {}).get("degree_distribution", {})
        if degree_dist:
            pdf_structure["sections"].append({
                "title": "Degree Distribution Statistics",
                "type": "stats_box",
                "description": "Statistical analysis of node connectivity",
                "stats": [
                    {"label": "Minimum Degree", "value": degree_dist.get("min", 0)},
                    {"label": "Maximum Degree", "value": degree_dist.get("max", 0)},
                    {"label": "Mean Degree", "value": degree_dist.get("mean", 0)},
                    {"label": "Median Degree", "value": degree_dist.get("median", 0)},
                    {"label": "90th Percentile", "value": degree_dist.get("p90", 0)},
                    {"label": "Standard Deviation", "value": degree_dist.get("std_dev", 0)},
                ],
            })
            
            # Add histogram data for rendering
            pdf_structure["sections"].append({
                "title": "Degree Distribution Histogram",
                "type": "histogram",
                "description": "Visual distribution of node degrees",
                "bins": await self._get_degree_histogram_data(folder_id),
            })
        
        # Add blind spots with enhanced formatting
        blind_spots = data.get("sections", {}).get("blind_spots", {})
        if blind_spots and blind_spots.get("top_predictions"):
            predictions = blind_spots.get("top_predictions", [])
            pdf_structure["sections"].append({
                "title": "Predicted Missing Relationships (Blind Spots)",
                "type": "enhanced_table",
                "description": f"AI-identified potential connections ({blind_spots.get('count', 0)} total predictions)",
                "columns": [
                    {"key": "source", "label": "Source Entity", "width": 150, "align": "left"},
                    {"key": "arrow", "label": "", "width": 40, "type": "icon", "icon": "arrow-right"},
                    {"key": "target", "label": "Target Entity", "width": 150, "align": "left"},
                    {"key": "confidence", "label": "Confidence", "width": 80, "type": "percentage"},
                    {"key": "reason", "label": "Reasoning", "width": 200, "align": "left"},
                ],
                "rows": [
                    {
                        "source": p.get("source_name", "Unknown"),
                        "arrow": "→",
                        "target": p.get("target_name", "Unknown"),
                        "confidence": p.get("confidence", 0),
                        "reason": p.get("reason", "Structural similarity"),
                    }
                    for p in predictions[:10]
                ],
                "highlight_rows": True,
            })
            
            # Add category breakdown
            by_category = blind_spots.get("by_category", {})
            if by_category:
                pdf_structure["sections"].append({
                    "title": "Blind Spots by Category",
                    "type": "bar_chart",
                    "orientation": "horizontal",
                    "data": [
                        {"label": cat, "value": count}
                        for cat, count in by_category.items()
                    ],
                })
        
        # Add health score section with visual gauge
        health = data.get("sections", {}).get("health", {})
        if health:
            score = health.get("score", 0)
            grade = health.get("grade", "N/A")
            issues = health.get("issues", [])
            
            pdf_structure["sections"].append({
                "title": "Graph Health Assessment",
                "type": "health_gauge",
                "score": score,
                "grade": grade,
                "max_score": 100,
                "zones": [
                    {"from": 0, "to": 60, "color": "#EF4444", "label": "Needs Work"},
                    {"from": 60, "to": 80, "color": "#F59E0B", "label": "Good"},
                    {"from": 80, "to": 100, "color": "#10B981", "label": "Excellent"},
                ],
                "issues": issues if issues else ["No issues detected - graph health is excellent!"],
                "recommendations": self._get_health_recommendations(score, issues),
            })
        
        return {
            "format": "pdf",
            "folder_id": folder_id,
            "filename": f"analytics_{folder_id}_{datetime.utcnow().strftime('%Y%m%d')}.pdf",
            "structure": pdf_structure,
            "raw_data": data,
            "content_type": "application/json",  # Frontend will render the PDF
        }
    
    def _build_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build executive summary text."""
        overview = data.get("overview", {})
        health = data.get("sections", {}).get("health", {})
        blind_spots = data.get("sections", {}).get("blind_spots", {})
        
        node_count = overview.get("node_count", 0)
        link_count = overview.get("link_count", 0)
        score = health.get("score", 0)
        grade = health.get("grade", "N/A")
        predictions = blind_spots.get("count", 0)
        
        return {
            "text": f"This knowledge graph contains {node_count:,} entities connected by {link_count:,} relationships. "
                    f"The overall health score is {score}/100 (Grade: {grade}). "
                    f"AI analysis has identified {predictions} potential missing connections that could enhance the graph.",
            "highlights": [
                f"{node_count:,} entities",
                f"{link_count:,} relationships",
                f"Health: {grade} ({score}%)",
                f"{predictions} blind spots",
            ],
        }
    
    def _get_health_recommendations(self, score: float, issues: List[str]) -> List[str]:
        """Generate recommendations based on health score and issues."""
        recommendations = []
        
        if score < 60:
            recommendations.append("Consider adding more documents to increase graph density")
            recommendations.append("Review isolated entities and create connecting relationships")
        elif score < 80:
            recommendations.append("Good foundation - consider diversifying relationship types")
            recommendations.append("Review blind spot predictions to fill knowledge gaps")
        else:
            recommendations.append("Excellent graph health! Continue maintaining data quality")
            recommendations.append("Consider periodic review of new entities for accuracy")
        
        if "density is very low" in " ".join(issues):
            recommendations.append("Many entities are disconnected - review for missing relationships")
        
        if "Very few nodes" in " ".join(issues):
            recommendations.append("Add more documents to build a comprehensive knowledge base")
        
        return recommendations
    
    async def _get_degree_histogram_data(self, folder_id: str) -> List[Dict[str, Any]]:
        """Get histogram bin data for degree distribution."""
        query = """
        MATCH (n)
        WHERE n.folder_id = $folder_id OR n.folderId = $folder_id
        OPTIONAL MATCH (n)-[r]-()
        WITH count(r) as degree
        RETURN degree, count(*) as count
        ORDER BY degree
        """
        
        try:
            result = await self.neo4j.execute_query(query, {"folder_id": folder_id})
            
            # Group into bins
            degree_counts = {}
            for r in result.records:
                degree = r["degree"]
                count = r["count"]
                # Create bins: 0, 1-2, 3-5, 6-10, 11-20, 21+
                if degree == 0:
                    bin_label = "0"
                elif degree <= 2:
                    bin_label = "1-2"
                elif degree <= 5:
                    bin_label = "3-5"
                elif degree <= 10:
                    bin_label = "6-10"
                elif degree <= 20:
                    bin_label = "11-20"
                else:
                    bin_label = "21+"
                
                degree_counts[bin_label] = degree_counts.get(bin_label, 0) + count
            
            # Order bins properly
            bin_order = ["0", "1-2", "3-5", "6-10", "11-20", "21+"]
            return [
                {"bin": b, "count": degree_counts.get(b, 0)}
                for b in bin_order
            ]
            
        except Exception as e:
            logger.error(f"Histogram query failed: {e}")
            return []


# Singleton
_export_service: Optional[AnalyticsExportService] = None


def get_export_service(neo4j) -> AnalyticsExportService:
    """Get or create the export service singleton."""
    global _export_service
    if _export_service is None:
        _export_service = AnalyticsExportService(neo4j)
    return _export_service
