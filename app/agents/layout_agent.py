"""
Layout Agent - Phase 1 of the Ingestion Pipeline

Analyzes document structure, identifying headers, tables, and sections.
This agent understands the visual and semantic layout of documents.
"""
import logging
import re
from typing import Any, Dict, List
from dataclasses import dataclass, field

from app.services.ai_service import get_ollama_service

logger = logging.getLogger(__name__)


@dataclass
class LayoutSection:
    """Represents a section identified in the document layout."""
    section_type: str  # 'header', 'paragraph', 'table', 'list', 'image', 'footer'
    content: str
    start_position: int
    end_position: int
    level: int  # For headers: 1-6, for others: 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class LayoutAgent:
    """
    Phase 1: Layout Analysis Agent
    
    Responsibilities:
    - Analyze document structure
    - Identify headers, paragraphs, tables, lists
    - Detect document hierarchy
    - Extract metadata (title, author, date if available)
    """
    
    SYSTEM_PROMPT = """You are a document structure analyst. Analyze the given text and identify its structural elements.

OUTPUT FORMAT (JSON):
{
    "document_type": "report|article|datasheet|form|other",
    "title": "extracted title or null",
    "author": "extracted author or null",
    "date": "extracted date or null",
    "sections": [
        {
            "type": "header|paragraph|table|list|code|quote",
            "level": 1-6 for headers else 0,
            "title": "section title if header",
            "summary": "brief summary of content"
        }
    ],
    "has_tables": true/false,
    "has_lists": true/false,
    "language": "detected language"
}

Be precise and only include what you can clearly identify from the text."""
    
    def __init__(self, model_name: str = None):
        self.ollama = get_ollama_service()
        self.model_name = model_name
        
    async def analyze(self, document_content: str, file_type: str) -> Dict[str, Any]:
        """
        Analyze the layout of a document.
        
        Args:
            document_content: Raw text content of the document
            file_type: Type of document (pdf, csv, txt, etc.)
            
        Returns:
            Dictionary containing layout analysis results
        """
        logger.info(f"Analyzing layout for {file_type} document ({len(document_content)} chars)")
        
        # For CSV/TSV, use rule-based detection
        if file_type in ['csv', 'tsv']:
            return self._analyze_tabular(document_content, file_type)
        
        # For text documents, use AI analysis
        sections = await self._detect_sections(document_content, file_type)
        hierarchy = self._build_hierarchy(sections)
        metadata = await self._extract_metadata(document_content)
        
        return {
            "sections": [s.__dict__ for s in sections],
            "hierarchy": hierarchy,
            "metadata": metadata,
            "document_type": file_type,
            "total_sections": len(sections),
            "content_length": len(document_content),
        }
    
    def _analyze_tabular(self, content: str, file_type: str) -> Dict[str, Any]:
        """Analyze tabular data (CSV/TSV)."""
        delimiter = '\t' if file_type == 'tsv' else ','
        lines = content.strip().split('\n')
        
        if not lines:
            return {"sections": [], "hierarchy": {}, "metadata": {}}
        
        # First line is typically headers
        headers = lines[0].split(delimiter)
        
        return {
            "sections": [{
                "section_type": "table",
                "content": content,
                "start_position": 0,
                "end_position": len(content),
                "level": 0,
                "metadata": {
                    "columns": headers,
                    "row_count": len(lines) - 1,
                    "delimiter": delimiter,
                }
            }],
            "hierarchy": {"table": {"columns": headers}},
            "metadata": {
                "document_type": "tabular_data",
                "columns": headers,
                "row_count": len(lines) - 1,
            },
            "document_type": file_type,
            "total_sections": 1,
            "content_length": len(content),
        }
    
    async def _detect_sections(self, content: str, file_type: str) -> List[LayoutSection]:
        """Detect different sections in the document using AI and rules."""
        sections = []
        
        # Rule-based section detection for common patterns
        lines = content.split('\n')
        current_pos = 0
        current_section_start = 0
        current_section_content = []
        current_section_type = "paragraph"
        
        for i, line in enumerate(lines):
            line_start = current_pos
            line_end = current_pos + len(line)
            
            # Detect headers (lines that are short and followed by blank or different content)
            is_header = (
                len(line.strip()) > 0 and
                len(line.strip()) < 100 and
                (line.strip().isupper() or 
                 line.strip().endswith(':') or
                 re.match(r'^#{1,6}\s', line) or  # Markdown headers
                 re.match(r'^\d+\.\s', line.strip()))  # Numbered sections
            )
            
            if is_header and current_section_content:
                # Save previous section
                section_content = '\n'.join(current_section_content)
                if section_content.strip():
                    sections.append(LayoutSection(
                        section_type=current_section_type,
                        content=section_content,
                        start_position=current_section_start,
                        end_position=current_pos - 1,
                        level=0,
                    ))
                
                # Start new header section
                level = 1
                if re.match(r'^#{1,6}\s', line):
                    level = len(re.match(r'^(#{1,6})\s', line).group(1))
                
                sections.append(LayoutSection(
                    section_type="header",
                    content=line.strip(),
                    start_position=line_start,
                    end_position=line_end,
                    level=level,
                ))
                
                current_section_content = []
                current_section_start = line_end + 1
                current_section_type = "paragraph"
            else:
                current_section_content.append(line)
            
            current_pos = line_end + 1  # +1 for newline
        
        # Add final section
        if current_section_content:
            section_content = '\n'.join(current_section_content)
            if section_content.strip():
                sections.append(LayoutSection(
                    section_type=current_section_type,
                    content=section_content,
                    start_position=current_section_start,
                    end_position=current_pos,
                    level=0,
                ))
        
        logger.info(f"Detected {len(sections)} sections")
        return sections
    
    def _build_hierarchy(self, sections: List[LayoutSection]) -> Dict[str, Any]:
        """Build a hierarchical structure from detected sections."""
        hierarchy = {
            "root": [],
            "depth": 0,
        }
        
        current_level = [hierarchy["root"]]
        
        for section in sections:
            if section.section_type == "header":
                hierarchy["depth"] = max(hierarchy["depth"], section.level)
                
        return hierarchy
    
    async def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract document metadata using AI."""
        # Take first 2000 chars for metadata extraction
        sample = content[:2000]
        
        try:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this document excerpt:\n\n{sample}"}
            ]
            
            result = await self.ollama.chat_json(messages)
            
            return {
                "title": result.get("title"),
                "author": result.get("author"),
                "date": result.get("date"),
                "document_type": result.get("document_type", "other"),
                "language": result.get("language", "en"),
                "has_tables": result.get("has_tables", False),
                "has_lists": result.get("has_lists", False),
            }
            
        except Exception as e:
            logger.warning(f"AI metadata extraction failed: {e}, using defaults")
            return {
                "title": None,
                "author": None,
                "date": None,
                "document_type": "other",
                "language": "en",
            }
