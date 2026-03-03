from typing import List, Dict, Any
from app.core.config import settings

def clean_label(label: str) -> str:
    """Strip folder isolation suffixes like _F_uuid."""
    if not label:
        return "Unknown"
    if "_F_" in label:
        return label.split("_F_")[0]
    return label

def get_node_type(labels: List[str], props: Dict[str, Any]) -> str:
    """Extract descriptive type from labels if props['type'] is missing."""
    raw_type = "Unknown"
    
    if props.get("type"):
        raw_type = str(props.get("type"))
    else:
        # Preferred order of labels (most descriptive first)
        system_labels = set(settings.GRAPH_SYSTEM_LABELS)
        domain_labels = [l for l in labels if l not in system_labels and not l.startswith('F_')]
        
        if domain_labels:
            priority = getattr(settings, "GRAPH_LABEL_PRIORITY", [])
            found_priority = False
            for p in priority:
                if p in domain_labels:
                    raw_type = p
                    found_priority = True
                    break
            
            if not found_priority:
                raw_type = max(domain_labels, key=len)
        else:
            other_labels = [l for l in labels if l not in system_labels]
            raw_type = other_labels[0] if other_labels else (labels[0] if labels else "Unknown")
            
    return clean_label(raw_type)

def get_node_name(labels: List[str], props: Dict[str, Any], node_id: str) -> str:
    """Find the best display name, avoiding technical IDs."""
    # 1. Properties priority — configurable display name keys
    priority_keys = settings.NODE_NAME_PRIORITY_KEYS
    for key in priority_keys:
        val = props.get(key)
        if val and isinstance(val, str) and len(val) < 100:
            # Check if it looks like a long UUID/hash
            if not (len(val) > 30 and (val.count('-') >= 4 or val.isalnum())):
                return val
    
    # 2. Fallback to Type (Cleaned)
    ntype = get_node_type(labels, props)
    if ntype not in ['Entity', 'Unknown']:
        return ntype
        
    # 3. Last resort: ID snippet
    if node_id:
        if len(node_id) > 20: # Likely UUID
             return node_id[:8]
        return node_id
        
    return "Unknown"
