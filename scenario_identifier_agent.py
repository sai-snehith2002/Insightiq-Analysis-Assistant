import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

SCENARIO_CONTEXT_MIN_SIMILARITY = # Ur threshold Value


def _vector_search(
   client: QdrantClient,
   *,
   collection_name: str,
   query_vector: List[float],
   limit: int,
):
   if hasattr(client, "search"):
       return client.search(
           collection_name=collection_name,
           query_vector=query_vector,
           limit=limit,
           with_payload=True,
           with_vectors=False,
       )
   query_response = client.query_points(
       collection_name=collection_name,
       query=query_vector,
       limit=limit,
       with_payload=True,
       with_vectors=False,
   )
   return query_response.points


@dataclass(frozen=True)
class ScenarioMatch:
   scenario_key: str
   matched_user_input: str
   score: float
   scenario_instruction: str
   scenario_output: Dict[str, Any]
   cardinality: str


class ScenarioIdentifierAgent:
   def __init__(
       self,
       *,
       qdrant_url: str,
       qdrant_api_key: str,
       qdrant_collection: str,
       embedding_model: str,
       scenarios_json_path: str,
   ) -> None:
       self.qdrant_collection = qdrant_collection
       self.scenarios_json_path = Path(scenarios_json_path)
       self.embedder = SentenceTransformer(embedding_model)
       self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
       self.scenarios = self._load_scenarios()

   def _load_scenarios(self) -> List[Dict[str, Any]]:
       with self.scenarios_json_path.open("r", encoding="utf-8") as f:
           raw = json.load(f)
       if not isinstance(raw, list):
           raise ValueError(f"{self.scenarios_json_path} must be a JSON array.")
       return raw

   def _stable_point_id(self, *, scenario_key: str, utterance: str) -> str:
       seed = f"{scenario_key}|{utterance}".encode("utf-8")
       digest = hashlib.sha1(seed).hexdigest()
       return int(digest[:16], 16)

   def ensure_collection(self) -> None:
       existing = {c.name for c in self.qdrant.get_collections().collections}
       vector_size = self.embedder.get_sentence_embedding_dimension()
       if self.qdrant_collection in existing:
           return
       self.qdrant.create_collection(
           collection_name=self.qdrant_collection,
           vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
       )

   def sync_scenarios(self) -> int:
       self.ensure_collection()
       points: List[PointStruct] = []
       total = 0

       for scenario in self.scenarios:
           scenario_key = str(scenario.get("scenario_key", "")).strip()
           scenario_instruction = str(scenario.get("scenario_instruction", "")).strip()
           scenario_output = scenario.get("scenario_output") or {}
           cardinality = str(scenario_output.get("cardinality", "")).strip()
           user_inputs = scenario.get("scenario_user_input") or []

           if not scenario_key or not isinstance(user_inputs, list):
               continue

           for utterance in user_inputs:
               if not isinstance(utterance, str):
                   continue
               text = utterance.strip()
               if not text:
                   continue
               vector = self.embedder.encode(text, normalize_embeddings=True).tolist()
               points.append(
                   PointStruct(
                       id=self._stable_point_id(scenario_key=scenario_key, utterance=text),
                       vector=vector,
                       payload={
                           "scenario_key": scenario_key,
                           "scenario_user_input": text,
                           "scenario_instruction": scenario_instruction,
                           "scenario_output": scenario_output,
                           "cardinality": cardinality,
                       },
                   )
               )
               total += 1

       if points:
           BATCH_SIZE = 32  # smaller = safer for large vectors
           for i in range(0, len(points), BATCH_SIZE):
               self.qdrant.upsert(
                   collection_name=self.qdrant_collection,
                   points=points[i : i + BATCH_SIZE],
               )
       return total

   def identify(self, user_input: str) -> ScenarioMatch:
       query_vector = self.embedder.encode(user_input, normalize_embeddings=True).tolist()
       results = _vector_search(
           self.qdrant,
           collection_name=self.qdrant_collection,
           query_vector=query_vector,
           limit=1,
       )

       if not results:
           raise RuntimeError("No scenario matches found in scenario collection.")

       best = results[0]
       payload = best.payload or {}
       scenario_output = payload.get("scenario_output") or {}
       matched_utterance = str(payload.get("scenario_user_input", ""))
       scenario_key = str(payload.get("scenario_key", ""))
       scenario_instruction = str(payload.get("scenario_instruction", ""))
       similarity_score = float(best.score)

       return ScenarioMatch(
           scenario_key=scenario_key,
           matched_user_input=matched_utterance,
           score=similarity_score,
           scenario_instruction=scenario_instruction,
           scenario_output=scenario_output if isinstance(scenario_output, dict) else {},
           cardinality=str(payload.get("cardinality", "")),
       )

   def augment_query(self, user_input: str, match: ScenarioMatch) -> str:
       expected_columns = match.scenario_output.get("expected_columns", [])
       scenario_type = match.scenario_output.get("type", "")
       columns_text = ", ".join(expected_columns) if isinstance(expected_columns, list) else ""

       return (
           f"{user_input}\n\n"
           f"[SCENARIO_CONTEXT_START]\n"
           f"scenario_key: {match.scenario_key}\n"
           f"closest_scenario_user_input: {match.matched_user_input}\n"
           f"scenario_instruction: {match.scenario_instruction}\n"
           f"scenario_output_type: {scenario_type}\n"
           f"scenario_output_expected_columns: {columns_text}\n"
           f"scenario_output_cardinality: {match.cardinality}\n"
           f"[SCENARIO_CONTEXT_END]"
       )

   def query_for_text_to_sql(self, user_input: str, match: ScenarioMatch) -> str:
       if match.score < SCENARIO_CONTEXT_MIN_SIMILARITY:
           return user_input
       return self.augment_query(user_input, match)