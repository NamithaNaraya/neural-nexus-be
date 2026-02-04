# Services Package
# Core business logic (AI processing, Azure, Neo4j interactions)

from .ai_service import AIService, get_ai_service, get_ollama_service

# Phase 5
from .graph_layout import GraphLayoutService, get_layout_service

# Phase 6: Advanced Intelligence
from .hybrid_rag import HybridRAGService, get_rag_service
from .cluster_comparison import ClusterComparisonService, get_comparison_service
from .blind_spot_discovery import BlindSpotDiscovery, get_discovery_service
from .analytics_export import AnalyticsExportService, get_export_service

# Phase 7: Final Features & Polish
from .permission_service import (
    PermissionService,
    PermissionLevel,
    ContextualSearchLock,
    get_permission_service,
)
from .deletion_service import (
    DeletionService,
    DeletionJob,
    DeletionStatus,
    ReferenceCounter,
    get_deletion_service,
)
from .health_service import (
    SystemHealthService,
    HealthStatus,
    ServiceStatus,
    IndexStatus,
    get_health_service,
)

__all__ = [
    # AI
    "AIService",
    "get_ai_service",
    "get_ollama_service",
    # Phase 5
    "GraphLayoutService",
    "get_layout_service",
    # Phase 6
    "HybridRAGService",
    "get_rag_service",
    "ClusterComparisonService",
    "get_comparison_service",
    "BlindSpotDiscovery",
    "get_discovery_service",
    "AnalyticsExportService",
    "get_export_service",
    # Phase 7
    "PermissionService",
    "PermissionLevel",
    "ContextualSearchLock",
    "get_permission_service",
    "DeletionService",
    "DeletionJob",
    "DeletionStatus",
    "ReferenceCounter",
    "get_deletion_service",
    "SystemHealthService",
    "HealthStatus",
    "ServiceStatus",
    "IndexStatus",
    "get_health_service",
]
