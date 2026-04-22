"""
Dashboard Service

Provides global statistics and recent activity for the main dashboard.
Aggregates data from both Neo4j (graph stats) and PostgreSQL (audit logs/activity).
"""
import logging
from typing import Dict, Any, List
from sqlalchemy import text
from app.db.connections import get_neo4j_driver, get_postgres_session
from app.services.cache_service import get_cache_service

logger = logging.getLogger(__name__)

class DashboardService:
    """Service for dashboard data aggregation."""
    
    async def get_stats(self) -> List[Dict[str, Any]]:
        """Get global graph statistics."""
        cache = get_cache_service()
        cache_key = f"{cache.analytics_prefix}dashboard:stats"
        cached = await cache.get_cached_value(cache_key)
        if cached is not None:
            return cached

        driver = get_neo4j_driver()
        
        async with driver.session() as session:
            # 1. Count Total Nodes
            node_result = await session.run("MATCH (n:Entity) RETURN count(n) as count")
            node_count = (await node_result.single())["count"]
            
            # 2. Count Relationships
            rel_result = await session.run("MATCH ()-[r:RELATIONSHIP]->() RETURN count(r) as count")
            rel_count = (await rel_result.single())["count"]
            
            # 3. Count Unique Types
            type_result = await session.run("MATCH (n:Entity) RETURN count(DISTINCT n.type) as count")
            type_count = (await type_result.single())["count"]
            
        # 4. Count Documents from PostgreSQL
        async with get_postgres_session() as session:
            doc_result = await session.execute(text("SELECT count(*) FROM neural_nexus.files WHERE status = 'completed'"))
            doc_count = doc_result.scalar()
            
        payload = [
            {"label": "Total Nodes", "value": f"{node_count:,}", "change": "Live", "trend": "up"},
            {"label": "Relationships", "value": f"{rel_count:,}", "change": "Live", "trend": "up"},
            {"label": "Entity Types", "value": str(type_count), "change": "Active", "trend": "up"},
            {"label": "Stored Documents", "value": str(doc_count), "change": "Processed", "trend": "up"},
        ]
        await cache.set_cached_value(cache_key, payload, ttl=60)
        return payload

    async def get_recent_activity(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent activity from audit logs."""
        async with get_postgres_session() as session:
            query = text("""
                SELECT 
                    id, 
                    action, 
                    target_type, 
                    target_id, 
                    details, 
                    created_at
                FROM neural_nexus.audit_logs
                ORDER BY created_at DESC
                LIMIT :limit
            """)
            result = await session.execute(query, {"limit": limit})
            rows = result.fetchall()
            
            activity = []
            for row in rows:
                # Format time relative or simple string
                # For now, let's just return formatted data
                details = row.details or {}
                target_name = details.get("filename") or details.get("name") or row.target_id or "Item"
                
                activity.append({
                    "id": str(row.id),
                    "action": row.action.capitalize(),
                    "target": str(target_name),
                    "time": self._format_time(row.created_at)
                })
                
            return activity

    def _format_time(self, dt):
        """Simple relative time formatter."""
        from datetime import datetime
        now = datetime.utcnow()
        diff = now - dt
        
        if diff.days > 0:
            return f"{diff.days}d ago"
        seconds = diff.seconds
        if seconds < 60:
            return "just now"
        if seconds < 3600:
            return f"{seconds // 60}m ago"
        return f"{seconds // 3600}h ago"

# Singleton
_dashboard_service = DashboardService()

def get_dashboard_service():
    return _dashboard_service
