# Enhanced RAG — 10 Intelligence Features

## Architecture

```
ENHANCED LANGGRAPH PIPELINE:

load_context ─→ smart_scope ─→ clarification_check
                                  │
                     ┌────────────┴──────────────┐
                     ↓ (vague)                   ↓ (clear)
              clarification_response      vector_search
                     ↓                          ↓
                    END                   ml_enrichment
                                               ↓
                                        strategic_scout
                                               ↓
                                        graph_expansion
                                               ↓
                                      prediction_injection
                                               ↓
                                        generate_answer
                                               ↓
                                              END
```

## Files Created / Modified

| File | Action | Purpose |
|------|--------|---------|
| `backend/app/services/rag/__init__.py` | Created | Package init |
| `backend/app/services/rag/enhanced_rag.py` | Created | Main Enhanced RAG Service (~500 lines) |
| `backend/app/routes/query.py` | Modified | Switched to Enhanced RAG, added response fields |
| `frontend/src/store/unifiedAssistantStore.ts` | Modified | Added RAGMetadata interface |
| `frontend/src/components/graph/panels/ReasoningAssistant.tsx` | Modified | Display grounding/ML/prediction badges |

## Feature Map

| # | Feature | LangGraph Node | What It Does |
|---|---------|---------------|-------------|
| 1 | Context Grounding | `generate_answer` | Forces LLM to stick to DB evidence via strict system prompt |
| 2 | Conversational Ask-Back | `clarification_check` → `clarification_response` | Detects vague queries, asks user to be specific with entity suggestions |
| 3 | ML-Powered Retrieval | `ml_enrichment` | Uses GDS Node Similarity to find structurally similar nodes |
| 4 | Link Prediction Context | `prediction_injection` | Checks for PREDICTED_LINK relationships and trained LP models |
| 5 | Node Classification | `prediction_injection` | Identifies unlabeled nodes, suggests NC model usage |
| 6 | Similarity Expansion | `ml_enrichment` + `graph_expansion` | Expands search to structurally equivalent nodes not found by text |
| 7 | Smart Scoping | `smart_scope` | Auto-detects most relevant folder when no scope provided |
| 8 | Citation Reranking | `vector_search` | Reranks by `score * 0.7 + centrality * 0.3` using degree count |
| 9 | Multi-Turn Memory | `load_context` | Compresses old history (>5 pairs) into 2-3 fact summary via LLM |
| 10 | Answer Confidence | `generate_answer` | Returns grounding_score = entities_in_answer / total_entities |

## API Response (New Fields)

```json
{
  "answer": "...",
  "citations": [...],
  "session_id": "...",
  "related_nodes": [...],
  "grounding_score": 0.73,
  "needs_clarification": false,
  "ml_insights_count": 3,
  "predictions_count": 1
}
```

## Frontend Badges

Each assistant response shows:
- **Green badge**: `73% GROUNDED` (high confidence)
- **Amber badge**: `45% GROUNDED` (moderate)
- **Red badge**: `20% GROUNDED` (low — mostly general knowledge)
- **Purple badge**: `3 ML INSIGHTS` (GDS nodes found)
- **Pink badge**: `1 PREDICTION` (LP/NC insights injected)
