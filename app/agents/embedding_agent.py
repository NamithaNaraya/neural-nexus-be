"""
Embedding Agent - Phase 7 of the Ingestion Pipeline

Generates vector embeddings for entities and chunks.
Uses Ollama's embedding models for semantic similarity.
"""
import logging
from typing import Any, Dict, List, Optional
from app.core.config import settings
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
    async def embed_entities(
        self,
        entities: List[Any],
    ) -> List[Any]:
        """
        Generate embeddings for a list of entities in batch for performance.
        """
        if not entities:
            return []
            
        logger.info(f"Ollama: Starting batch embedding for {len(entities)} entities...")
        
        # 1. Prepare entity dicts and text for embedding
        prepared_data = []
        texts_to_embed = []
        
        for entity in entities:
            try:
                # Convert dataclass to dict if needed
                if hasattr(entity, '__dataclass_fields__'):
                    entity_dict = {
                        'id': entity.id,
                        'name': entity.name,
                        'type': entity.type,
                        'description': getattr(entity, 'description', ''),
                        'properties': getattr(entity, 'properties', {}),
                        'source_chunk_id': getattr(entity, 'source_chunk_id', None),
                        'source_text': getattr(entity, 'source_text', ''),
                        'confidence': getattr(entity, 'confidence', 1.0),
                    }
                elif isinstance(entity, dict):
                    entity_dict = entity.copy()
                else:
                    entity_dict = dict(entity) if hasattr(entity, '__iter__') else {'name': str(entity)}
                
                name = entity_dict.get('name', '')
                entity_type = entity_dict.get('type', '')
                description = entity_dict.get('description', '')
                
                embed_text = f"{name}. Type: {entity_type}. {description}"
                
                prepared_data.append(entity_dict)
                texts_to_embed.append(embed_text)
                
            except Exception as e:
                logger.warning(f"Failed to prepare entity for embedding: {e}")
                # Add a dummy entry to keep indices aligned if we really have to, 
                # but better to skip if it's broken.
        
        # 2. Call Ollama Batch
        try:
            logger.info(f"Ollama: Sending {len(texts_to_embed)} texts to {settings.OLLAMA_BASE_URL}...")
            embeddings = await self.ollama.embed_batch(texts_to_embed)
            logger.info(f"Ollama: Successfully received {len(embeddings)} embeddings")
            
            # 3. Assign embeddings back
            for i, embedding in enumerate(embeddings):
                if i < len(prepared_data):
                    prepared_data[i]['embedding'] = embedding
                    
        except Exception as e:
            logger.error(f"Ollama: Batch embedding failed: {e}")
            logger.warning("Proceeding with entities without embeddings")
            # We already have prepared_data without 'embedding' key, which is fine as fallback
            
        return prepared_data
    
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
                # Convert dataclass to dict if needed
                if hasattr(chunk, '__dataclass_fields__'):
                    chunk_dict = {
                        'chunk_id': getattr(chunk, 'chunk_id', None),
                        'content': getattr(chunk, 'content', ''),
                        'section_type': getattr(chunk, 'section_type', 'paragraph'),
                        'start_position': getattr(chunk, 'start_position', 0),
                        'end_position': getattr(chunk, 'end_position', 0),
                        'metadata': getattr(chunk, 'metadata', {}),
                    }
                elif isinstance(chunk, dict):
                    chunk_dict = chunk.copy()
                else:
                    chunk_dict = {'content': str(chunk)}
                
                content = chunk_dict.get('content', '')
                
                if not content.strip():
                    embedded_chunks.append(chunk_dict)
                    continue
                
                # Generate embedding
                embedding = await self.ollama.embed(content)
                
                # Add embedding to chunk dict
                chunk_dict['embedding'] = embedding
                
                embedded_chunks.append(chunk_dict)
                
            except Exception as e:
                logger.warning(f"Failed to embed chunk: {e}")
                # Still include chunk - convert if needed
                if hasattr(chunk, '__dataclass_fields__'):
                    chunk_dict = {
                        'chunk_id': getattr(chunk, 'chunk_id', None),
                        'content': getattr(chunk, 'content', ''),
                        'section_type': getattr(chunk, 'section_type', 'paragraph'),
                    }
                    embedded_chunks.append(chunk_dict)
                elif isinstance(chunk, dict):
                    embedded_chunks.append(chunk)
                else:
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
