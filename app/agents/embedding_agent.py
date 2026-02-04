"""
Embedding Agent - Phase 7 of the Ingestion Pipeline

Generates vector embeddings for entities and chunks.
Uses Ollama's embedding models for semantic similarity.
"""
import logging
from typing import Any, Dict, List, Optional

from app.services.ai_service import get_ollama_service

logger = logging.getLogger(__name__)


class EmbeddingAgent:
    """
    Phase 7: Embedding Generation Agent
    
    Responsibilities:
    - Generate embeddings for entities
    - Generate embeddings for text chunks
    - Prepare data for vector similarity search
    """
    
    def __init__(self, model_name: str = None):
        self.ollama = get_ollama_service()
        self.model_name = model_name
        
    async def embed_entities(
        self,
        entities: List[Any],
    ) -> List[Any]:
        """
        Generate embeddings for entities.
        
        Args:
            entities: List of validated entities
            
        Returns:
            Entities with embeddings added
        """
        logger.info(f"Generating embeddings for {len(entities)} entities")
        
        embedded_entities = []
        
        for entity in entities:
            try:
                # Build text for embedding
                name = entity.name if hasattr(entity, 'name') else entity.get('name', '')
                entity_type = entity.type if hasattr(entity, 'type') else entity.get('type', '')
                description = entity.description if hasattr(entity, 'description') else entity.get('description', '')
                
                embed_text = f"{name}. Type: {entity_type}. {description}"
                
                # Generate embedding
                embedding = await self.ollama.embed(embed_text)
                
                # Add embedding to entity
                if hasattr(entity, 'embedding'):
                    entity.embedding = embedding
                else:
                    entity['embedding'] = embedding
                
                embedded_entities.append(entity)
                
            except Exception as e:
                logger.warning(f"Failed to embed entity {name}: {e}")
                # Still include entity without embedding
                embedded_entities.append(entity)
        
        logger.info(f"Generated embeddings for {len(embedded_entities)} entities")
        return embedded_entities
    
    async def embed_chunks(
        self,
        chunks: List[Any],
    ) -> List[Any]:
        """
        Generate embeddings for text chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Chunks with embeddings added
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        embedded_chunks = []
        
        for chunk in chunks:
            try:
                content = chunk.content if hasattr(chunk, 'content') else chunk.get('content', '')
                
                if not content.strip():
                    embedded_chunks.append(chunk)
                    continue
                
                # Generate embedding
                embedding = await self.ollama.embed(content)
                
                # Add embedding to chunk
                if hasattr(chunk, 'embedding'):
                    chunk.embedding = embedding
                else:
                    chunk['embedding'] = embedding
                
                embedded_chunks.append(chunk)
                
            except Exception as e:
                logger.warning(f"Failed to embed chunk: {e}")
                embedded_chunks.append(chunk)
        
        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
        return embedded_chunks
    
    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding vector
        """
        try:
            return await self.ollama.embed(query)
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Return the dimension of embeddings for the current model."""
        # mxbai-embed-large returns 1024-dimensional embeddings
        # Other models may vary
        model = self.model_name or self.ollama.embed_model
        
        dimension_map = {
            "mxbai-embed-large": 1024,
            "nomic-embed-text": 768,
            "all-minilm": 384,
        }
        
        return dimension_map.get(model, 768)  # Default to 768
