"""
Hybrid RAG Service

Vector + Graph reasoning chain for natural language queries.
Uses LangGraph for orchestration with sliding window context.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Sliding window size for conversation context
SLIDING_WINDOW_SIZE = 5


@dataclass
class ChatMessage:
    """Single chat message."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    citations: List[Dict[str, Any]] = None
    

@dataclass
class Citation:
    """Source citation for an answer."""
    node_id: str
    node_name: str
    chunk_text: str
    confidence: float
    source_file: str = None


class ConversationMemory:
    """
    Sliding window conversation memory.
    Keeps last N messages for context, optimizing for performance.
    """
    
    def __init__(self, window_size: int = SLIDING_WINDOW_SIZE):
        self.window_size = window_size
        self._sessions: Dict[str, List[ChatMessage]] = {}
    
    def add_message(self, session_id: str, role: str, content: str, citations: List = None):
        """Add a message to the session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            citations=citations,
        )
        self._sessions[session_id].append(message)
        
        # Trim to window size (keeping last N pairs = 2N messages)
        max_messages = self.window_size * 2
        if len(self._sessions[session_id]) > max_messages:
            self._sessions[session_id] = self._sessions[session_id][-max_messages:]
    
    def get_context(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation context for the session."""
        if session_id not in self._sessions:
            return []
        
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self._sessions[session_id]
        ]
    
    def clear_session(self, session_id: str):
        """Clear a session's history."""
        if session_id in self._sessions:
            del self._sessions[session_id]
    
    def get_last_n_questions(self, session_id: str, n: int = 5) -> List[str]:
        """Get the last N questions for context."""
        if session_id not in self._sessions:
            return []
        
        questions = [
            msg.content for msg in self._sessions[session_id]
            if msg.role == 'user'
        ]
        return questions[-n:]


class HybridRAGService:
    """
    Hybrid RAG (Retrieval Augmented Generation) Service.
    
    Combines:
    - Vector search for semantic similarity
    - Graph traversal for relationship context
    - LLM for answer generation
    """
    
    def __init__(self, neo4j_driver, ai_service):
        self.neo4j = neo4j_driver
        self.ai = ai_service
        self.memory = ConversationMemory()
    
    async def query(
        self,
        question: str,
        session_id: str,
        scope: Optional[Dict[str, Any]] = None,
        clear_history: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a hybrid RAG query.
        
        Steps:
        1. Get conversation context (sliding window)
        2. Vector search for relevant chunks
        3. Graph traversal for relationship context
        4. LLM generation with citations
        """
        # Clear history if requested
        if clear_history:
            self.memory.clear_session(session_id)
        
        try:
            # Step 1: Get conversation context
            conversation_context = self.memory.get_context(session_id)
            last_questions = self.memory.get_last_n_questions(session_id)
            
            # Build context-aware query
            enhanced_question = self._build_enhanced_question(question, last_questions)
            
            # Step 2: Vector search
            vector_results = await self._vector_search(enhanced_question, scope)
            
            # Step 3: Graph context expansion
            graph_context = await self._expand_graph_context(vector_results, scope)
            
            # Step 4: Generate answer with LLM
            answer, citations = await self._generate_answer(
                question=question,
                vector_context=vector_results,
                graph_context=graph_context,
                conversation_context=conversation_context,
            )
            
            # Store in memory
            self.memory.add_message(session_id, 'user', question)
            self.memory.add_message(session_id, 'assistant', answer, citations)
            
            # Get related nodes from citations
            related_nodes = [c['node_id'] for c in citations if 'node_id' in c]
            
            return {
                "answer": answer,
                "citations": citations,
                "session_id": session_id,
                "related_nodes": related_nodes,
                "context_used": len(conversation_context),
            }
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return {
                "answer": f"I encountered an error processing your question: {str(e)}",
                "citations": [],
                "session_id": session_id,
                "related_nodes": [],
                "error": str(e),
            }
    
    def _build_enhanced_question(self, question: str, last_questions: List[str]) -> str:
        """
        Build an enhanced question using conversation context.
        Helps resolve pronouns and references.
        """
        if not last_questions:
            return question
        
        # Create context summary for entity resolution
        context_summary = " | ".join(last_questions[-3:])  # Last 3 questions
        return f"Previous context: {context_summary}\n\nCurrent question: {question}"
    
    async def _vector_search(
        self,
        query: str,
        scope: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search in Neo4j.
        """
        try:
            # Build scope filter
            scope_filter = ""
            params = {"query": query, "top_k": top_k}
            
            if scope:
                scope_type = scope.get("type")
                scope_id = scope.get("id")
                
                if scope_type == "folder" and scope_id:
                    scope_filter = "AND (n.folder_id = $scope_id OR n.folderId = $scope_id)"
                    params["scope_id"] = scope_id
                elif scope_type == "file" and scope_id:
                    scope_filter = "AND (n.file_id = $scope_id OR n.fileId = $scope_id)"
                    params["scope_id"] = scope_id
                elif scope_type == "selection":
                    node_ids = scope.get("node_ids", [])
                    if node_ids:
                        scope_filter = "AND n.id IN $node_ids"
                        params["node_ids"] = node_ids
            
            # Query using vector index (if available) or text search fallback
            vector_query = f"""
            CALL db.index.fulltext.queryNodes('entity_search', $query)
            YIELD node, score
            WHERE score > 0.3 {scope_filter.replace('n.', 'node.')}
            RETURN 
                COALESCE(node.id, node.entity_id, elementId(node)) as node_id,
                node.name as name,
                node.description as description,
                node.type as type,
                score
            ORDER BY score DESC
            LIMIT $top_k
            """
            
            result = self.neo4j.execute_query(vector_query, params)
            
            return [
                {
                    "node_id": record["node_id"],
                    "name": record["name"],
                    "description": record["description"] or "",
                    "type": record["type"],
                    "score": record["score"],
                }
                for record in result.records
            ]
            
        except Exception as e:
            logger.warning(f"Vector search failed, using fallback: {e}")
            # Fallback to simple text matching
            return await self._fallback_text_search(query, scope, top_k)
    
    async def _fallback_text_search(
        self,
        query: str,
        scope: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Fallback text-based search when vector search unavailable."""
        scope_filter = ""
        params = {"query": f".*{query}.*", "top_k": top_k}
        
        if scope and scope.get("type") == "folder" and scope.get("id"):
            scope_filter = "AND (n.folder_id = $scope_id OR n.folderId = $scope_id)"
            params["scope_id"] = scope.get("id")
        
        fallback_query = f"""
        MATCH (n)
        WHERE (n.name =~ $query OR n.description =~ $query)
        {scope_filter}
        RETURN 
            COALESCE(n.id, n.entity_id, elementId(n)) as node_id,
            n.name as name,
            n.description as description,
            n.type as type,
            1.0 as score
        LIMIT $top_k
        """
        
        result = self.neo4j.execute_query(fallback_query, params)
        
        return [
            {
                "node_id": record["node_id"],
                "name": record["name"],
                "description": record["description"] or "",
                "type": record["type"],
                "score": record["score"],
            }
            for record in result.records
        ]
    
    async def _expand_graph_context(
        self,
        vector_results: List[Dict[str, Any]],
        scope: Optional[Dict[str, Any]] = None,
        max_hops: int = 2,
    ) -> Dict[str, Any]:
        """
        Expand the graph context around vector search results.
        Follows relationships to build a rich context.
        """
        if not vector_results:
            return {"nodes": [], "relationships": [], "paths": []}
        
        node_ids = [r["node_id"] for r in vector_results[:5]]  # Top 5 for expansion
        
        try:
            expansion_query = """
            UNWIND $node_ids AS nodeId
            MATCH (n)
            WHERE n.id = nodeId OR elementId(n) = nodeId
            OPTIONAL MATCH path = (n)-[r*1..2]-(related)
            RETURN 
                n.name as source_name,
                [rel in relationships(path) | type(rel)] as rel_types,
                [node in nodes(path) | node.name] as path_nodes,
                related.name as related_name,
                related.type as related_type
            LIMIT 50
            """
            
            result = self.neo4j.execute_query(expansion_query, {"node_ids": node_ids})
            
            relationships = []
            nodes = set()
            
            for record in result.records:
                source = record["source_name"]
                if source:
                    nodes.add(source)
                
                related = record["related_name"]
                if related:
                    nodes.add(related)
                
                rel_types = record["rel_types"]
                if rel_types and source and related:
                    relationships.append({
                        "source": source,
                        "target": related,
                        "types": rel_types,
                    })
            
            return {
                "nodes": list(nodes),
                "relationships": relationships,
                "node_count": len(nodes),
                "relationship_count": len(relationships),
            }
            
        except Exception as e:
            logger.error(f"Graph expansion failed: {e}")
            return {"nodes": [], "relationships": [], "paths": []}
    
    async def _generate_answer(
        self,
        question: str,
        vector_context: List[Dict[str, Any]],
        graph_context: Dict[str, Any],
        conversation_context: List[Dict[str, str]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate an answer using the LLM with all available context.
        """
        # Build context prompt
        context_parts = []
        
        # Add vector search results
        if vector_context:
            context_parts.append("Relevant entities found:")
            for i, result in enumerate(vector_context[:5], 1):
                context_parts.append(
                    f"{i}. {result['name']} ({result['type']}): {result['description'][:200]}"
                )
        
        # Add graph relationships
        if graph_context.get("relationships"):
            context_parts.append("\nRelevant relationships:")
            for rel in graph_context["relationships"][:10]:
                rel_str = " -> ".join(rel["types"]) if rel["types"] else "related to"
                context_parts.append(f"- {rel['source']} {rel_str} {rel['target']}")
        
        context_str = "\n".join(context_parts) if context_parts else "No relevant context found."
        
        # Build conversation history
        history_str = ""
        if conversation_context:
            history_parts = []
            for msg in conversation_context[-6:]:  # Last 3 exchanges
                role = "User" if msg["role"] == "user" else "Assistant"
                history_parts.append(f"{role}: {msg['content'][:200]}")
            history_str = "\n".join(history_parts)
        
        # Build the prompt
        system_prompt = """You are a knowledge graph assistant. Answer questions based ONLY on the provided context.
If the context doesn't contain enough information, say so clearly. Do not hallucinate or make up information.
When citing information, mention the source entity name."""

        user_prompt = f"""Context from knowledge graph:
{context_str}

{'Conversation history:' + chr(10) + history_str if history_str else ''}

Question: {question}

Provide a helpful answer based on the context. Be precise and cite your sources."""

        # Call AI service
        try:
            response = await self.ai.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=1000,
            )
            answer = response.get("text", "I couldn't generate a response.")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = "I encountered an error generating the response. Please try again."
        
        # Build citations from vector results
        citations = [
            {
                "node_id": r["node_id"],
                "node_name": r["name"],
                "chunk_text": r["description"][:200] if r["description"] else "",
                "confidence": r["score"],
            }
            for r in vector_context[:5]
        ]
        
        return answer, citations


# Singleton instance
_rag_service: Optional[HybridRAGService] = None


def get_rag_service(neo4j, ai_service) -> HybridRAGService:
    """Get or create the RAG service singleton."""
    global _rag_service
    if _rag_service is None:
        _rag_service = HybridRAGService(neo4j, ai_service)
    return _rag_service
