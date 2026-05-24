import json
import os
import sys
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

import httpx
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer


EntityType = Literal["table", "stored_proc", "function", "business_rule"]


@dataclass(frozen=True)
class Entity:
    entity_type: EntityType
    entity_name: str
    description: str
    column_name: Optional[str] = None
    tags: Optional[List[str]] = None
    source: Optional[str] = None  # e.g. "handwritten", "dbt", "ddl", "wiki"


def entity_to_canonical_text(entity: Entity) -> str:
    parts: List[str] = []
    parts.append(f"ENTITY_TYPE: {entity.entity_type}")
    parts.append(f"ENTITY_NAME: {entity.entity_name}")
    if entity.column_name:
        parts.append(f"COLUMN_NAME: {entity.column_name}")
    if entity.tags:
        parts.append(f"TAGS: {', '.join(entity.tags)}")
    if entity.source:
        parts.append(f"SOURCE: {entity.source}")
    parts.append("")
    parts.append("DESCRIPTION:")
    parts.append(entity.description.strip())
    return "\n".join(parts).strip() + "\n"


def chunk_text_preserving_context(
    *,
    header: str,
    body: str,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> List[str]:
    header = header.strip()
    body = body.strip()
    if not body:
        return [header]

    separators = ["\n\n", "\n", ". ", " "]
    chunks: List[str] = []

    def split_by_sep(text: str, sep: str) -> List[str]:
        if sep == " ":
            return text.split(sep)
        return text.split(sep)

    def join_with_sep(parts: List[str], sep: str) -> str:
        if sep == " ":
            return " ".join(parts).strip()
        return sep.join(parts).strip()

    remaining = body
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(f"{header}\n\n{remaining}".strip())
            break

        chosen_sep = None
        chosen_parts: List[str] = []
        chosen_sep_used = ""
        for sep in separators:
            parts = split_by_sep(remaining, sep)
            if len(parts) <= 1:
                continue
            chosen_sep = sep
            chosen_parts = parts
            chosen_sep_used = sep
            break

        if not chosen_sep:
            cut = remaining[:max_chars]
            chunks.append(f"{header}\n\n{cut}".strip())
            remaining = remaining[max_chars - overlap_chars :].strip()
            continue

        packed: List[str] = []
        current = ""
        for i, p in enumerate(chosen_parts):
            candidate_parts = packed + [p]
            candidate = join_with_sep(candidate_parts, chosen_sep_used)
            candidate_full = f"{header}\n\n{candidate}".strip()
            if len(candidate_full) <= max_chars:
                packed.append(p)
                current = candidate
                continue
            if not packed:
                cut = remaining[:max_chars]
                chunks.append(f"{header}\n\n{cut}".strip())
                remaining = remaining[max_chars - overlap_chars :].strip()
                break

            chunks.append(f"{header}\n\n{current}".strip())
            leftover = join_with_sep(chosen_parts[len(packed) :], chosen_sep_used)
            tail = current[-overlap_chars:].strip()
            remaining = (tail + (chosen_sep_used if tail and leftover else "") + leftover).strip()
            break
        else:
            chunks.append(f"{header}\n\n{current}".strip())
            remaining = ""

    return [c for c in chunks if c]


def chunk_entity(entity: Entity, *, max_chars: int = 1200, overlap_chars: int = 150) -> List[str]:
    canonical = entity_to_canonical_text(entity)
    lines = canonical.splitlines()
    header_lines: List[str] = []
    body_lines: List[str] = []
    in_description = False
    for ln in lines:
        if ln.strip() == "DESCRIPTION:":
            in_description = True
            body_lines.append(ln)
            continue
        if not in_description:
            header_lines.append(ln)
        else:
            body_lines.append(ln)

    header = "\n".join([l for l in header_lines if l.strip()]).strip()
    body = "\n".join(body_lines).strip()
    return chunk_text_preserving_context(
        header=header,
        body=body,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )


def get_embedder(model_name: str) -> SentenceTransformer:
    # You can swap to a stronger model later; keep it configurable.
    return SentenceTransformer(model_name)


def ensure_qdrant_collection(
    *,
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def ensure_payload_indexes(*, client: QdrantClient, collection_name: str) -> None:
    client.create_payload_index(
        collection_name=collection_name,
        field_name="kb_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="entity_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )


def load_entities_from_json(path: str) -> List[Entity]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be a JSON array of entities")

    entities: List[Entity] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"{path}[{i}] must be an object")
        try:
            entities.append(
                Entity(
                    entity_type=item["entity_type"],
                    entity_name=item["entity_name"],
                    column_name=item.get("column_name"),
                    description=item["description"],
                    tags=item.get("tags"),
                    source=item.get("source"),
                )
            )
        except KeyError as e:
            raise ValueError(f"Missing required field {e} at {path}[{i}]") from e
    return entities


def _entity_id(entity: Entity) -> str:
    return f"{entity.entity_type}:{entity.entity_name}:{entity.column_name or ''}"


def _stable_point_id(*, kb_id: str, entity_id: str, chunk_index: int) -> str:
    """
    Stable UUID so reruns overwrite instead of duplicating.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"kb:{kb_id}|{entity_id}|chunk:{chunk_index}"))


def delete_removed_entities(
    *,
    client: QdrantClient,
    collection_name: str,
    kb_id: str,
    valid_entity_ids: Set[str],
) -> int:
    flt = Filter(
        must=[
            FieldCondition(
                key="kb_id",
                match=MatchValue(value=kb_id),
            )
        ]
    )

    to_delete: List[str] = []
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=flt,
            with_payload=["entity_id"],
            with_vectors=False,
            limit=256,
            offset=next_offset,
        )
        for p in points:
            payload = p.payload or {}
            eid = payload.get("entity_id")
            if isinstance(eid, str) and eid not in valid_entity_ids:
                to_delete.append(str(p.id))
        if next_offset is None:
            break

    if to_delete:
        client.delete(collection_name=collection_name, points_selector=to_delete)
    return len(to_delete)


def upsert_entities(
    *,
    client: QdrantClient,
    collection_name: str,
    embedder: SentenceTransformer,
    entities: Sequence[Entity],
    kb_id: str,
    max_chars: int = 1200,
    overlap_chars: int = 150,
    batch_size: int = 64,
) -> int:
    points: List[PointStruct] = []

    def flush() -> None:
        nonlocal points
        if not points:
            return
        client.upsert(collection_name=collection_name, points=points)
        points = []

    total = 0
    for entity in entities:
        chunks = chunk_entity(entity, max_chars=max_chars, overlap_chars=overlap_chars)
        vectors = embedder.encode(chunks, normalize_embeddings=True).tolist()
        eid = _entity_id(entity)

        for idx, (text, vec) in enumerate(zip(chunks, vectors)):
            payload: Dict[str, Any] = {
                "entity": asdict(entity),
                "kb_id": kb_id,
                "entity_id": eid,
                "chunk_index": idx,
                "chunk_count": len(chunks),
                "text": text,
            }
            points.append(
                PointStruct(
                    id=_stable_point_id(kb_id=kb_id, entity_id=eid, chunk_index=idx),
                    vector=vec,
                    payload=payload,
                )
            )
            total += 1
            if len(points) >= batch_size:
                flush()

    flush()
    return total


def example_entities() -> List[Entity]:
    return [
        Entity(
            entity_type="table",
            entity_name="orders",
            column_name="order_status",
            tags=["sales", "core"],
            description=(
                "Status of the order lifecycle. Use to filter active vs cancelled orders.\n"
                "Valid values: NEW, PAID, SHIPPED, DELIVERED, CANCELLED.\n"
                "Business constraint: shipped orders cannot be cancelled."
            ),
            source="handwritten",
        ),
        Entity(
            entity_type="business_rule",
            entity_name="revenue_recognition",
            tags=["finance"],
            description=(
                "Revenue is recognized at DELIVERED time (not at PAID time).\n"
                "For monthly revenue reporting, group by delivered_at date.\n"
                "Exclude CANCELLED orders from recognized revenue."
            ),
            source="handwritten",
        ),
    ]


def build_client_from_env() -> Tuple[QdrantClient, str]:
    """
    Env vars:
      - QDRANT_URL: e.g. http://localhost:6333 or https://xxxx.cloud.qdrant.io
      - QDRANT_API_KEY: optional (cloud)
      - QDRANT_COLLECTION: defaults to "sql-grounding-kb"
    """
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")
    collection = os.getenv("QDRANT_COLLECTION", "sql-grounding-kb")

    client = QdrantClient(url=url, api_key=api_key, check_compatibility=False)
    return client, collection


def main() -> None:
    load_dotenv()

    # Configure
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    max_chars = int(os.getenv("CHUNK_MAX_CHARS", "1200"))
    overlap_chars = int(os.getenv("CHUNK_OVERLAP_CHARS", "150"))
    entities_path = os.getenv("ENTITIES_JSON", "db_knowledge_base.json")
    kb_id = os.getenv("KB_ID", "db_knowledge_base")

    # Init
    client, collection = build_client_from_env()
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    embedder = get_embedder(embedding_model)

    # Ensure collection exists (vector size depends on embedding model)
    vector_size = embedder.get_sentence_embedding_dimension()
    try:
        ensure_qdrant_collection(client=client, collection_name=collection, vector_size=vector_size)
        ensure_payload_indexes(client=client, collection_name=collection)

        # Load from JSON (source of truth)
        entities = load_entities_from_json(entities_path)
        valid_entity_ids = {_entity_id(e) for e in entities}
        deleted = delete_removed_entities(
            client=client,
            collection_name=collection,
            kb_id=kb_id,
            valid_entity_ids=valid_entity_ids,
        )
        stored = upsert_entities(
            client=client,
            collection_name=collection,
            embedder=embedder,
            entities=entities,
            kb_id=kb_id,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
    except (ResponseHandlingException, httpx.ConnectError, httpx.ConnectTimeout) as e:
        print(
            f"\nCannot reach Qdrant at {qdrant_url!r} ({type(e).__name__}: {e})\n\n"
            "This script now loads variables from a `.env` file in the project root (if present).\n"
            "Fix one of the following:\n"
            "  • Set QDRANT_URL (and QDRANT_API_KEY for Qdrant Cloud) in `.env`.\n"
            "  • Or run a local Qdrant, e.g.:\n"
            "      docker run -p 6333:6333 qdrant/qdrant\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    print(
        f"Synced KB '{kb_id}' into Qdrant collection '{collection}': "
        f"upserted {stored} chunks, deleted {deleted} stale chunks."
    )


if __name__ == "__main__":
    main()