"""
Neo4j GDS Machine Learning Service

Full ML lifecycle for:
  - Link Prediction (predict missing connections)
  - Node Classification (predict missing labels)
  - Node Embeddings (FastRP, Node2Vec)
  - Node Similarity (find similar nodes)
"""
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class MLService:
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver

    async def _run(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return records as dicts."""
        params = params or {}
        try:
            async with self.driver.session() as session:
                result = await session.run(query, **params)
                return await result.data()
        except Exception as e:
            logger.error(f"ML Query Error: {e}\nQuery: {query}\nParams: {params}")
            raise

    async def _drop_pipeline_safe(self, pipeline_name: str):
        """Silently drop a pipeline if it exists."""
        try:
            await self._run(
                "CALL gds.beta.pipeline.drop($name) YIELD pipelineName RETURN pipelineName",
                {"name": pipeline_name},
            )
            logger.info(f"Dropped stale pipeline: {pipeline_name}")
        except Exception:
            pass

    async def _drop_graph_safe(self, graph_name: str):
        """Silently drop a GDS graph projection if it exists."""
        try:
            await self._run(
                "CALL gds.graph.drop($name, false) YIELD graphName RETURN graphName",
                {"name": graph_name},
            )
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════
    #  MODEL CATALOG
    # ═══════════════════════════════════════════════════════════

    async def list_models(self) -> List[Dict[str, Any]]:
        """List all trained ML models in the Neo4j GDS catalog."""
        return await self._run(
            "CALL gds.model.list() "
            "YIELD modelName, modelType, trainConfig "
            "RETURN modelName, modelType, trainConfig"
        )

    async def drop_model(self, model_name: str) -> bool:
        """Drop a trained model from the catalog."""
        try:
            await self._run(
                "CALL gds.model.drop($name) YIELD modelName RETURN modelName",
                {"name": model_name},
            )
            return True
        except Exception:
            return False

    # ═══════════════════════════════════════════════════════════
    #  1. LINK PREDICTION  (predict missing connections)
    # ═══════════════════════════════════════════════════════════

    async def train_link_prediction_e2e(
        self, graph_name: str, pipeline_name: str, model_name: str, embedding_dim: int = 256
    ) -> Dict[str, Any]:
        """
        End-to-end Link Prediction pipeline:
          1. FastRP embeddings → 2. Hadamard features → 3. Split config
          4. LogReg + RandomForest candidates → 5. Train → 6. Cleanup
        """
        p = {"pipeline_name": pipeline_name}

        await self._drop_pipeline_safe(pipeline_name)

        await self._run("CALL gds.beta.pipeline.linkPrediction.create($pipeline_name)", p)

        await self._run("""
            CALL gds.beta.pipeline.linkPrediction.addNodeProperty($pipeline_name, 'fastRP', {
                mutateProperty: 'embedding', embeddingDimension: $dim, randomSeed: 42
            })
        """, {**p, "dim": embedding_dim})

        await self._run("""
            CALL gds.beta.pipeline.linkPrediction.addFeature($pipeline_name, 'hadamard', {
                nodeProperties: ['embedding']
            })
        """, p)

        await self._run("""
            CALL gds.beta.pipeline.linkPrediction.configureSplit($pipeline_name, {
                testFraction: 0.25, trainFraction: 0.6, validationFolds: 3
            })
        """, p)

        await self._run("""
            CALL gds.beta.pipeline.linkPrediction.addLogisticRegression($pipeline_name, {
                penalty: 0.0, tolerance: 0.01, maxEpochs: 500
            })
        """, p)

        try:
            await self._run("""
                CALL gds.beta.pipeline.linkPrediction.addRandomForest($pipeline_name, {
                    numberOfDecisionTrees: 10, maxDepth: 5
                })
            """, p)
        except Exception:
            logger.info("RandomForest not available for LP, continuing with LogReg only")

        try:
            rel_types_records = await self._run(
                "CALL gds.graph.list($graph_name) YIELD schemaWithOrientation RETURN keys(schemaWithOrientation.relationships) AS types",
                {"graph_name": graph_name}
            )
            rel_types = rel_types_records[0]["types"] if rel_types_records and "types" in rel_types_records[0] else []
            
            target_rel_type = "ALL_RELATIONSHIPS"
            if "ALL_RELATIONSHIPS" in rel_types:
                target_rel_type = "ALL_RELATIONSHIPS"
            elif "_ALL_" in rel_types:
                target_rel_type = "_ALL_"
            elif rel_types:
                target_rel_type = rel_types[0]

            records = await self._run("""
                CALL gds.beta.pipeline.linkPrediction.train($graph_name, {
                    pipeline: $pipeline_name,
                    modelName: $model_name,
                    randomSeed: 42,
                    targetRelationshipType: $target_rel_type
                })
                YIELD modelInfo, trainMillis
                RETURN modelInfo, trainMillis
            """, {"graph_name": graph_name, "pipeline_name": pipeline_name, "model_name": model_name, "target_rel_type": target_rel_type})
        except Exception as e:
            await self._drop_pipeline_safe(pipeline_name)
            raise e

        await self._drop_pipeline_safe(pipeline_name)
        return records[0] if records else {}

    async def predict_links(
        self, graph_name: str, model_name: str, threshold: float = 0.5, top_n: int = 100
    ) -> List[Dict[str, Any]]:
        """Stream link predictions from a trained model."""
        return await self._run("""
            CALL gds.beta.pipeline.linkPrediction.predict.stream($graph_name, {
                modelName: $model_name, threshold: $threshold, topN: $top_n
            })
            YIELD node1, node2, probability
            WITH gds.util.asNode(node1) AS n1, gds.util.asNode(node2) AS n2, probability
            RETURN n1.id AS source_id, n1.name AS source_name, n1.type AS source_type,
                   n2.id AS target_id, n2.name AS target_name, n2.type AS target_type,
                   probability
            ORDER BY probability DESC
        """, {"graph_name": graph_name, "model_name": model_name,
              "threshold": threshold, "top_n": top_n})

    # ═══════════════════════════════════════════════════════════
    #  2. NODE CLASSIFICATION  (predict missing labels)
    #
    #  Challenge: GDS native projections only support numeric
    #  properties. We must encode the string 'type' → numeric
    #  '_nc_type_id' before projecting.
    # ═══════════════════════════════════════════════════════════

    async def _prepare_nc_graph(self) -> tuple:
        """
        Prepare a dedicated graph projection for Node Classification:
          1. Discover all unique node types
          2. Encode each type as a numeric _nc_type_id on nodes
          3. Create a native undirected projection including _nc_type_id
        Returns (graph_name, type_map) where type_map = {0: 'Herb', ...}
        """
        graph_name = "nc_classification_graph"

        # Always recreate to reflect latest data
        await self._drop_graph_safe(graph_name)

        # Step 1: Discover unique types
        types_data = await self._run(
            "MATCH (n:Entity) WHERE n.type IS NOT NULL "
            "RETURN DISTINCT n.type AS type ORDER BY type"
        )
        type_list = [r["type"] for r in types_data if r["type"]]
        type_map = {i: t for i, t in enumerate(type_list)}  # {0: 'Herb', 1: 'Property', ...}
        logger.info(f"NC type map: {type_map}")

        # Step 2: Encode as numeric property (ONLY for nodes that have a type)
        for type_id, type_name in type_map.items():
            await self._run(
                "MATCH (n:Entity) WHERE n.type = $type_name "
                "SET n._nc_type_id = $type_id",
                {"type_name": type_name, "type_id": type_id}
            )

        # REMOVE _nc_type_id from nodes WITHOUT a type
        # GDS will skip these during training but still predict for them
        await self._run(
            "MATCH (n:Entity) WHERE n.type IS NULL "
            "REMOVE n._nc_type_id"
        )

        # Step 3: Create native projection with the numeric property
        await self._run("""
            CALL gds.graph.project(
                $name,
                {Entity: {properties: ['_nc_type_id']}},
                {_ALL_: {type: '*', orientation: 'UNDIRECTED'}}
            )
        """, {"name": graph_name})

        logger.info(f"Created NC projection: {graph_name} with {len(type_map)} classes")
        return graph_name, type_map

    async def train_node_classification_e2e(
        self, pipeline_name: str, model_name: str, embedding_dim: int = 256
    ) -> Dict[str, Any]:
        """
        End-to-end Node Classification pipeline:
          1. Prepare graph with encoded types
          2. FastRP embeddings → 3. Select features → 4. Split config
          5. LogReg + RandomForest → 6. Train on '_nc_type_id' → 7. Cleanup
        Returns training results + type_map for decoding.
        """
        p = {"pipeline_name": pipeline_name}

        # Prepare graph with numeric type encoding
        graph_name, type_map = await self._prepare_nc_graph()

        await self._drop_pipeline_safe(pipeline_name)

        await self._run("CALL gds.beta.pipeline.nodeClassification.create($pipeline_name)", p)

        await self._run("""
            CALL gds.beta.pipeline.nodeClassification.addNodeProperty($pipeline_name, 'fastRP', {
                mutateProperty: 'nc_embedding', embeddingDimension: $dim, randomSeed: 42
            })
        """, {**p, "dim": embedding_dim})

        await self._run("""
            CALL gds.beta.pipeline.nodeClassification.selectFeatures($pipeline_name, ['nc_embedding'])
        """, p)

        await self._run("""
            CALL gds.beta.pipeline.nodeClassification.configureSplit($pipeline_name, {
                testFraction: 0.2, validationFolds: 5
            })
        """, p)

        await self._run("""
            CALL gds.beta.pipeline.nodeClassification.addLogisticRegression($pipeline_name, {
                penalty: 0.0, maxEpochs: 500
            })
        """, p)

        try:
            await self._run("""
                CALL gds.beta.pipeline.nodeClassification.addRandomForest($pipeline_name, {
                    numberOfDecisionTrees: 10, maxDepth: 6
                })
            """, p)
        except Exception:
            logger.info("RandomForest not available for NC, using LogReg only")

        try:
            records = await self._run("""
                CALL gds.beta.pipeline.nodeClassification.train($graph_name, {
                    pipeline: $pipeline_name,
                    modelName: $model_name,
                    targetProperty: '_nc_type_id',
                    randomSeed: 42,
                    metrics: ['ACCURACY']
                })
                YIELD modelInfo, trainMillis
                RETURN modelInfo, trainMillis
            """, {"graph_name": graph_name, "pipeline_name": pipeline_name,
                  "model_name": model_name})
        except Exception as e:
            await self._drop_pipeline_safe(pipeline_name)
            await self._drop_graph_safe(graph_name)
            raise e

        await self._drop_pipeline_safe(pipeline_name)
        await self._drop_graph_safe(graph_name)

        result = records[0] if records else {}
        result["type_map"] = type_map
        return result

    async def predict_node_classes(
        self, model_name: str, top_n: int = 100
    ) -> List[Dict[str, Any]]:
        """Stream node classification predictions with decoded labels."""
        # Reconstruct graph for prediction
        graph_name, type_map = await self._prepare_nc_graph()
        reverse_map = type_map  # {0: 'Herb', 1: 'Property', ...}

        try:
            results = await self._run("""
                CALL gds.beta.pipeline.nodeClassification.predict.stream($graph_name, {
                    modelName: $model_name
                })
                YIELD nodeId, predictedClass, predictedProbabilities
                WITH gds.util.asNode(nodeId) AS node, predictedClass, predictedProbabilities
                RETURN node.id AS id, node.name AS name, node.type AS current_type,
                       predictedClass AS predicted_class_id,
                       predictedProbabilities AS probabilities
                LIMIT $top_n
            """, {"graph_name": graph_name, "model_name": model_name, "top_n": top_n})
        finally:
            await self._drop_graph_safe(graph_name)

        # Decode numeric class IDs back to readable type names
        for r in results:
            raw_id = r.get("predicted_class_id")
            # GDS may return Long/int64 — cast to Python int for dict lookup
            class_id = int(raw_id) if raw_id is not None else -1
            r["predicted_type"] = reverse_map.get(class_id, f"Class {class_id}")

        return results

    # ═══════════════════════════════════════════════════════════
    #  3. NODE EMBEDDINGS  (standalone mathematical fingerprints)
    # ═══════════════════════════════════════════════════════════

    async def generate_fastrp(
        self, graph_name: str, dim: int = 128, top_k: int = 100
    ) -> List[Dict[str, Any]]:
        """Generate FastRP embeddings for all nodes."""
        return await self._run("""
            CALL gds.fastRP.stream($graph_name, {
                embeddingDimension: $dim, randomSeed: 42
            })
            YIELD nodeId, embedding
            WITH gds.util.asNode(nodeId) AS node, embedding
            RETURN node.id AS id, node.name AS name, node.type AS type, embedding
            LIMIT $top_k
        """, {"graph_name": graph_name, "dim": dim, "top_k": top_k})

    async def generate_node2vec(
        self, graph_name: str, dim: int = 128, top_k: int = 100
    ) -> List[Dict[str, Any]]:
        """Generate Node2Vec embeddings using random walks."""
        return await self._run("""
            CALL gds.node2vec.stream($graph_name, {
                embeddingDimension: $dim,
                walkLength: 80,
                walksPerNode: 10,
                windowSize: 10,
                randomSeed: 42
            })
            YIELD nodeId, embedding
            WITH gds.util.asNode(nodeId) AS node, embedding
            RETURN node.id AS id, node.name AS name, node.type AS type, embedding
            LIMIT $top_k
        """, {"graph_name": graph_name, "dim": dim, "top_k": top_k})

    # ═══════════════════════════════════════════════════════════
    #  4. NODE SIMILARITY  (find structurally similar nodes)
    # ═══════════════════════════════════════════════════════════

    async def run_node_similarity(
        self, graph_name: str, top_k: int = 10, similarity_cutoff: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Compute Jaccard similarity between nodes based on shared neighbors."""
        return await self._run("""
            CALL gds.nodeSimilarity.stream($graph_name, {
                topK: $top_k,
                similarityCutoff: $cutoff
            })
            YIELD node1, node2, similarity
            WITH gds.util.asNode(node1) AS n1, gds.util.asNode(node2) AS n2, similarity
            RETURN n1.id AS source_id, n1.name AS source_name, n1.type AS source_type,
                   n2.id AS target_id, n2.name AS target_name, n2.type AS target_type,
                   similarity
            ORDER BY similarity DESC
            LIMIT 50
        """, {"graph_name": graph_name, "top_k": top_k, "cutoff": similarity_cutoff})
