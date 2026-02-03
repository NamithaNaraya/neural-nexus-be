"""
Chunking Agent - Phase 2 of the Ingestion Pipeline

Intelligent text segmentation based on semantic boundaries rather than character count.
Uses layout information from Phase 1 to create meaningful chunks.
"""
import logging
import re
import uuid
from typing import Any, Dict, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a semantic chunk of text."""
    chunk_id: str
    content: str
    start_position: int
    end_position: int
    section_type: str
    section_title: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChunkingAgent:
    """
    Phase 2: Semantic Chunking Agent
    
    Responsibilities:
    - Segment text based on semantic boundaries
    - Respect document structure from layout analysis
    - Maintain context continuity across chunks
    - Optimize chunk size for embedding and retrieval
    """
    
    def __init__(self, 
                 max_chunk_size: int = 1000,
                 min_chunk_size: int = 100,
                 chunk_overlap: int = 100,
                 model_name: str = None):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        
    async def chunk(self, layout_result: Dict[str, Any]) -> List[TextChunk]:
        """
        Create semantic chunks from layout-analyzed document.
        
        Args:
            layout_result: Output from LayoutAgent.analyze()
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        sections = layout_result.get("sections", [])
        
        if not sections:
            logger.warning("No sections found in layout result")
            return chunks
        
        current_header = None
        
        for section in sections:
            section_type = section.get("section_type", "paragraph")
            content = section.get("content", "")
            
            if section_type == "header":
                current_header = content
                continue
            
            if not content.strip():
                continue
                
            # Chunk this section
            section_chunks = self._chunk_section(
                content=content,
                section_type=section_type,
                section_title=current_header,
                start_position=section.get("start_position", 0),
                metadata=section.get("metadata", {}),
            )
            chunks.extend(section_chunks)
        
        # Add overlap between chunks for context continuity
        chunks = self._add_overlap(chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(sections)} sections")
        return chunks
    
    def _chunk_section(
        self,
        content: str,
        section_type: str,
        section_title: str,
        start_position: int,
        metadata: Dict[str, Any],
    ) -> List[TextChunk]:
        """Chunk a single section respecting semantic boundaries."""
        chunks = []
        
        # If content is small enough, return as single chunk
        if len(content) <= self.max_chunk_size:
            chunks.append(TextChunk(
                chunk_id=str(uuid.uuid4()),
                content=content.strip(),
                start_position=start_position,
                end_position=start_position + len(content),
                section_type=section_type,
                section_title=section_title,
                metadata=metadata,
            ))
            return chunks
        
        # Find semantic boundaries (sentences, paragraphs)
        boundaries = self._find_semantic_boundaries(content)
        
        current_chunk_content = []
        current_chunk_start = start_position
        current_length = 0
        
        for boundary in boundaries:
            segment = boundary["text"]
            segment_length = len(segment)
            
            if current_length + segment_length > self.max_chunk_size:
                # Save current chunk if it has content
                if current_chunk_content:
                    chunk_text = ' '.join(current_chunk_content).strip()
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(TextChunk(
                            chunk_id=str(uuid.uuid4()),
                            content=chunk_text,
                            start_position=current_chunk_start,
                            end_position=current_chunk_start + len(chunk_text),
                            section_type=section_type,
                            section_title=section_title,
                            metadata=metadata,
                        ))
                
                # Start new chunk
                current_chunk_content = [segment]
                current_chunk_start = start_position + boundary["start"]
                current_length = segment_length
            else:
                current_chunk_content.append(segment)
                current_length += segment_length
        
        # Add final chunk
        if current_chunk_content:
            chunk_text = ' '.join(current_chunk_content).strip()
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(TextChunk(
                    chunk_id=str(uuid.uuid4()),
                    content=chunk_text,
                    start_position=current_chunk_start,
                    end_position=current_chunk_start + len(chunk_text),
                    section_type=section_type,
                    section_title=section_title,
                    metadata=metadata,
                ))
        
        return chunks
    
    def _find_semantic_boundaries(self, text: str) -> List[Dict[str, Any]]:
        """Find natural break points in text (sentences, paragraphs)."""
        boundaries = []
        
        # Split by paragraphs first (double newlines)
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_pos = 0
        for para in paragraphs:
            if not para.strip():
                current_pos += len(para) + 2  # +2 for double newline
                continue
            
            # Split paragraphs into sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            
            for sentence in sentences:
                if sentence.strip():
                    boundaries.append({
                        "text": sentence.strip(),
                        "start": current_pos,
                        "end": current_pos + len(sentence),
                        "type": "sentence",
                    })
                current_pos += len(sentence) + 1
            
            current_pos += 2  # For paragraph break
        
        return boundaries
    
    def _add_overlap(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Add context overlap between chunks."""
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]
            
            # Only add overlap if chunks are from same section
            if prev_chunk.section_title == current_chunk.section_title:
                # Get last N characters from previous chunk
                overlap_text = prev_chunk.content[-self.chunk_overlap:]
                
                # Find sentence boundary in overlap
                if '. ' in overlap_text:
                    overlap_text = overlap_text.split('. ', 1)[-1]
                
                # Prepend to current chunk
                current_chunk.content = f"[...] {overlap_text} {current_chunk.content}"
                current_chunk.metadata["has_overlap"] = True
        
        return chunks
