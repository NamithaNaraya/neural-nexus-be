"""
Celery Worker Configuration

Sets up the distributed task queue for:
- File ingestion pipeline
- Background AI processing
- Long-running graph operations
"""
from celery import Celery

from app.core.config import settings

# Initialize Celery app
celery_app = Celery(
    "neural_nexus",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Concurrency
    worker_concurrency=settings.CELERY_WORKER_CONCURRENCY,
    
    # Task routing
    task_routes={
        "app.tasks.ingestion.*": {"queue": "ingestion"},
        "app.tasks.analytics.*": {"queue": "analytics"},
        "app.tasks.maintenance.*": {"queue": "maintenance"},
    },
    
    # Retry settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Result expiration (24 hours)
    result_expires=86400,
    
    # Rate limiting
    task_annotations={
        "app.tasks.ingestion.process_chunk": {
            "rate_limit": f"{settings.MAX_CHUNK_PARALLEL}/m"
        },
    },
)


# Task discovery
celery_app.autodiscover_tasks([
    "app.tasks",
])
