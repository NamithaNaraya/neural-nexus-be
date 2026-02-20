"""
ML API Routes

Endpoints for all Neo4j GDS Machine Learning capabilities:
  - Model Catalog (list / delete)
  - Link Prediction (train / predict)
  - Node Classification (train / predict)
  - Node Embeddings (FastRP / Node2Vec)
  - Node Similarity
"""
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
import logging
from app.core.security import get_current_user
from app.db.connections import get_neo4j_driver
from app.services.ml.ml_service import MLService
from app.services.gds_service import get_gds_service, GDSService

router = APIRouter()
logger = logging.getLogger(__name__)


def get_ml_service():
    return MLService(get_neo4j_driver())


# ═══ MODEL CATALOG ═══════════════════════════════════════════

@router.get("/models")
async def list_models(
    current_user: dict = Depends(get_current_user),
    ml: MLService = Depends(get_ml_service),
) -> Dict[str, Any]:
    try:
        models = await ml.list_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(500, detail=str(e))


@router.delete("/model/{model_name}")
async def drop_model(
    model_name: str,
    current_user: dict = Depends(get_current_user),
    ml: MLService = Depends(get_ml_service),
) -> Dict[str, Any]:
    try:
        ok = await ml.drop_model(model_name)
        return {"success": ok, "message": f"Model '{model_name}' deleted."}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ═══ LINK PREDICTION ═════════════════════════════════════════

@router.post("/link-prediction/train")
async def train_link_prediction(
    folder_id: str = Query(...),
    pipeline_name: str = Query("lp_pipeline", min_length=3),
    model_name: str = Query("lp_model", min_length=3),
    current_user: dict = Depends(get_current_user),
    ml: MLService = Depends(get_ml_service),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    try:
        graph = await gds.ensure_projection(folder_id=folder_id, undirected=True)
        result = await ml.train_link_prediction_e2e(graph, pipeline_name, model_name)
        return {
            "status": "success", "model_name": model_name,
            "model_info": result.get("modelInfo"),
            "training_time_ms": result.get("trainMillis"),
        }
    except Exception as e:
        logger.error(f"LP training error: {e}")
        raise HTTPException(500, detail=f"Training failed: {e}")


@router.get("/link-prediction/predict")
async def predict_links(
    folder_id: str = Query(...),
    model_name: str = Query(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    top_n: int = Query(50, le=500),
    current_user: dict = Depends(get_current_user),
    ml: MLService = Depends(get_ml_service),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    try:
        graph = await gds.ensure_projection(folder_id=folder_id, undirected=True)
        preds = await ml.predict_links(graph, model_name, threshold, top_n)
        return {"predictions": preds, "count": len(preds)}
    except Exception as e:
        raise HTTPException(500, detail=f"Prediction failed: {e}")


# ═══ NODE CLASSIFICATION ═════════════════════════════════════

@router.post("/node-classification/train")
async def train_node_classification(
    pipeline_name: str = Query("nc_pipeline", min_length=3),
    model_name: str = Query("nc_model", min_length=3),
    current_user: dict = Depends(get_current_user),
    ml: MLService = Depends(get_ml_service),
) -> Dict[str, Any]:
    """
    E2E Node Classification. Automatically:
      - Encodes string 'type' → numeric for GDS
      - Creates a dedicated graph projection
      - Trains the model
      - Cleans up projection + pipeline
    """
    try:
        result = await ml.train_node_classification_e2e(pipeline_name, model_name)
        return {
            "status": "success", "model_name": model_name,
            "model_info": result.get("modelInfo"),
            "training_time_ms": result.get("trainMillis"),
            "type_map": result.get("type_map"),
        }
    except Exception as e:
        logger.error(f"NC training error: {e}")
        raise HTTPException(500, detail=f"Training failed: {e}")


@router.get("/node-classification/predict")
async def predict_node_classes(
    model_name: str = Query(...),
    top_n: int = Query(100, le=500),
    current_user: dict = Depends(get_current_user),
    ml: MLService = Depends(get_ml_service),
) -> Dict[str, Any]:
    try:
        preds = await ml.predict_node_classes(model_name, top_n)
        return {"predictions": preds, "count": len(preds)}
    except Exception as e:
        raise HTTPException(500, detail=f"Prediction failed: {e}")


# ═══ NODE EMBEDDINGS ═════════════════════════════════════════

@router.get("/embeddings/generate")
async def generate_embeddings(
    folder_id: str = Query(...),
    method: str = Query("fastRP"),
    dim: int = Query(128, ge=16, le=512),
    top_k: int = Query(100, le=500),
    current_user: dict = Depends(get_current_user),
    ml: MLService = Depends(get_ml_service),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    try:
        graph = await gds.ensure_projection(folder_id=folder_id, undirected=True)
        if method == "node2vec":
            results = await ml.generate_node2vec(graph, dim, top_k)
        else:
            results = await ml.generate_fastrp(graph, dim, top_k)
        return {"method": method, "dimension": dim, "embeddings": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(500, detail=f"Embedding generation failed: {e}")


# ═══ NODE SIMILARITY ═════════════════════════════════════════

@router.get("/node-similarity")
async def node_similarity(
    folder_id: str = Query(...),
    top_k: int = Query(10, le=50),
    cutoff: float = Query(0.1, ge=0.0, le=1.0),
    current_user: dict = Depends(get_current_user),
    ml: MLService = Depends(get_ml_service),
    gds: GDSService = Depends(get_gds_service),
) -> Dict[str, Any]:
    try:
        graph = await gds.ensure_projection(folder_id=folder_id, undirected=True)
        results = await ml.run_node_similarity(graph, top_k, cutoff)
        return {"similarities": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(500, detail=f"Similarity failed: {e}")
