"""
Weight Service

Converts user-defined weight formulas (stored as JSON) into:
  - Cypher expressions for GDS projections
  - Human-readable descriptions for RAG/Chat context

Supports formula types:
  - property:      Single numeric property (e.g., "marks")
  - ratio:         numerator / denominator (e.g., marks / time = accuracy)
  - weighted_sum:  c1*p1 + c2*p2 + ... (e.g., 0.6*marks + 0.4*time)
  - expression:    Free-form math expression with named properties
"""
import re
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

# Properties that are used internally and should never be treated as weight properties
RESERVED_PROPERTIES = {
    "id", "name", "type", "description", "folder_id", "file_id", "file_ids",
    "folderId", "fileId", "entity_id", "embedding", "created_at", "updated_at",
    "created_by", "label", "text", "title", "x", "y", "z", "conflicts",
}

# Allowed operators in expressions (safety whitelist)
SAFE_EXPR_PATTERN = re.compile(r'^[\w\s\.\+\-\*\/\(\)]+$')


class WeightService:
    """Converts weight config formulas into Cypher expressions."""

    @staticmethod
    def formula_to_cypher(formula: Dict[str, Any], rel_var: str = "r") -> str:
        """
        Convert a formula JSON to a Cypher expression.
        
        Args:
            formula: The formula dict from weight_configs.formula
            rel_var: The Cypher variable name for the relationship (default "r")
        
        Returns:
            A Cypher expression string like "toFloat(r.marks) / toFloat(r.time)"
        """
        formula_type = formula.get("type", "property")

        if formula_type == "property":
            prop = formula.get("property", "strength")
            return f"coalesce(toFloat({rel_var}.{prop}), 1.0)"

        elif formula_type == "ratio":
            num = formula.get("numerator", "strength")
            den = formula.get("denominator", "1")
            # Prevent division by zero
            return (
                f"CASE WHEN coalesce(toFloat({rel_var}.{den}), 0) = 0 "
                f"THEN 1.0 "
                f"ELSE coalesce(toFloat({rel_var}.{num}), 0.0) / toFloat({rel_var}.{den}) "
                f"END"
            )

        elif formula_type == "weighted_sum":
            terms = formula.get("terms", [])
            if not terms:
                return "1.0"
            cypher_terms = []
            for term in terms:
                prop = term.get("property", "strength")
                coeff = term.get("coefficient", 1.0)
                cypher_terms.append(f"{coeff} * coalesce(toFloat({rel_var}.{prop}), 0.0)")
            return " + ".join(cypher_terms)

        elif formula_type == "expression":
            expr = formula.get("expr", "1.0")
            properties = formula.get("properties", [])
            # Validate expression safety
            if not SAFE_EXPR_PATTERN.match(expr):
                logger.warning(f"[WeightService] Unsafe expression rejected: {expr}")
                return "1.0"
            # Replace property names with Cypher references
            cypher_expr = expr
            for prop in sorted(properties, key=len, reverse=True):
                cypher_expr = cypher_expr.replace(
                    prop, f"coalesce(toFloat({rel_var}.{prop}), 0.0)"
                )
            return cypher_expr

        else:
            logger.warning(f"[WeightService] Unknown formula type: {formula_type}")
            return "1.0"

    @staticmethod
    def formula_to_node_cypher(formula: Dict[str, Any], node_var: str = "n") -> str:
        """
        Convert a formula to a Cypher expression for NODE properties.
        Used when weight properties are on nodes instead of relationships.
        """
        formula_type = formula.get("type", "property")

        if formula_type == "property":
            prop = formula.get("property", "strength")
            return f"coalesce(toFloat({node_var}.{prop}), 0.0)"

        elif formula_type == "ratio":
            num = formula.get("numerator")
            den = formula.get("denominator")
            return (
                f"CASE WHEN coalesce(toFloat({node_var}.{den}), 0) = 0 "
                f"THEN 0.0 "
                f"ELSE coalesce(toFloat({node_var}.{num}), 0.0) / toFloat({node_var}.{den}) "
                f"END"
            )

        elif formula_type == "weighted_sum":
            terms = formula.get("terms", [])
            if not terms:
                return "0.0"
            parts = []
            for t in terms:
                parts.append(f"{t.get('coefficient', 1.0)} * coalesce(toFloat({node_var}.{t['property']}), 0.0)")
            return " + ".join(parts)

        elif formula_type == "expression":
            expr = formula.get("expr", "0.0")
            properties = formula.get("properties", [])
            if not SAFE_EXPR_PATTERN.match(expr):
                return "0.0"
            cypher_expr = expr
            for prop in sorted(properties, key=len, reverse=True):
                cypher_expr = cypher_expr.replace(
                    prop, f"coalesce(toFloat({node_var}.{prop}), 0.0)"
                )
            return cypher_expr

        return "0.0"

    @staticmethod
    def formula_to_description(formula: Dict[str, Any]) -> str:
        """
        Convert a formula to a human-readable description for RAG/Chat context.
        """
        formula_type = formula.get("type", "property")

        if formula_type == "property":
            return f"Weighted by: {formula.get('property', 'strength')}"

        elif formula_type == "ratio":
            label = formula.get("label", "ratio")
            return f"Weighted by: {label} = {formula.get('numerator')} / {formula.get('denominator')}"

        elif formula_type == "weighted_sum":
            terms = formula.get("terms", [])
            parts = [f"{t.get('coefficient', 1.0)}×{t['property']}" for t in terms]
            return f"Weighted by: {' + '.join(parts)}"

        elif formula_type == "expression":
            return f"Weighted by: {formula.get('expr', 'custom formula')}"

        return "No weight applied"

    @staticmethod
    def get_formula_properties(formula: Dict[str, Any]) -> List[str]:
        """Extract all property names referenced in a formula."""
        formula_type = formula.get("type", "property")

        if formula_type == "property":
            return [formula.get("property", "strength")]

        elif formula_type == "ratio":
            return [formula.get("numerator", ""), formula.get("denominator", "")]

        elif formula_type == "weighted_sum":
            return [t.get("property", "") for t in formula.get("terms", [])]

        elif formula_type == "expression":
            return formula.get("properties", [])

        return []


# Singleton
_weight_service = None

def get_weight_service() -> WeightService:
    global _weight_service
    if _weight_service is None:
        _weight_service = WeightService()
    return _weight_service
