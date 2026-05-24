import re
from typing import Any, Dict, List
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


class TextToSQLAgent:
    def __init__(
        self,
        *,
        qdrant_url: str,
        qdrant_api_key: str,
        qdrant_collection: str,
        embedding_model: str,
        groq_api_key: str,
        groq_model: str,
        top_k: int = 40,
    ) -> None:
        self.qdrant_collection = qdrant_collection
        self.groq_api_key = groq_api_key
        self.groq_model = groq_model
        self.top_k = top_k

        self.embedder = SentenceTransformer(embedding_model)
        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.groq_client: Groq | None = None

    def run(self, user_query: str) -> Dict[str, Any]:
        context_chunks = self.search_relevant_chunks(user_query=user_query, top_k=self.top_k)
        if not context_chunks:
            raise RuntimeError("No context found in Qdrant. Please ingest entities first.")
        sql = self.generate_sql_from_groq(user_query=user_query, context_chunks=context_chunks)
        return {"sql": sql, "context_chunks": context_chunks}

    def search_relevant_chunks(self, *, user_query: str, top_k: int) -> List[Dict[str, Any]]:
        query_vector = self.embedder.encode(user_query, normalize_embeddings=True).tolist()

        if hasattr(self.qdrant, "search"):
            results = self.qdrant.search(
                collection_name=self.qdrant_collection,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
        else:
            query_response = self.qdrant.query_points(
                collection_name=self.qdrant_collection,
                query=query_vector,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            results = query_response.points

        chunks: List[Dict[str, Any]] = []
        for point in results:
            payload = point.payload or {}
            chunks.append(
                {
                    "score": round(float(point.score), 4),
                    "text": payload.get("text", ""),
                    "entity_id": payload.get("entity_id", ""),
                }
            )
        return chunks

    def build_prompt(self, *, user_query: str, context_chunks: List[Dict[str, Any]]) -> str:
        context_text = "\n\n---\n\n".join(
            [f"[{idx + 1}] {chunk['text']}" for idx, chunk in enumerate(context_chunks)]
        )
        return f"""
You are a senior analytics engineer.
Generate a single SQL query based on the user's requirement and grounding context.

Rules:
1) Use ONLY entities/columns/rules from the provided context.
2) If requirement is ambiguous just create the SQL query abiding to the SQL schema and dont assume any additional columns to be present.
3) Prefer ANSI-compatible SQL. If warehouse-specific syntax is needed, use database-compatible SQL.
4) Return SQL only, comments not needed.
5) By default the year to be considered is 2026 if at all the user didn't mention the year.

User requirement:
{user_query}

Grounding context:
{context_text}
""".strip()

    def extract_pure_sql(self, llm_response: str) -> str:
        text = llm_response.strip()
        fenced = re.findall(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            text = "\n".join(fenced).strip()

        text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
        text = re.sub(r"(?m)^\s*--.*?$", "", text)
        text = re.sub(r"(?m)\s+--.*?$", "", text)

        sql_start = re.search(
            r"\b(select|with|insert|update|delete|create|alter|drop|merge|truncate)\b",
            text,
            flags=re.IGNORECASE,
        )
        if not sql_start:
            return ""

        candidate = text[sql_start.start() :].strip()
        statement = re.search(r"(?s)^(.*?;)", candidate)
        sql = statement.group(1).strip() if statement else candidate
        sql = re.sub(r"\n{3,}", "\n\n", sql)
        return sql.strip()

    def generate_sql_from_groq(self, *, user_query: str, context_chunks: List[Dict[str, Any]]) -> str:
        if not self.groq_api_key:
            raise RuntimeError("GROQ_API_KEY missing. Set it in your .env file.")
        if self.groq_client is None:
            self.groq_client = Groq(api_key=self.groq_api_key)

        prompt = self.build_prompt(user_query=user_query, context_chunks=context_chunks)
        completion = self.groq_client.chat.completions.create(
            model=self.groq_model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": "You generate correct SQL from grounded context."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = completion.choices[0].message.content or ""
        cleaned = self.extract_pure_sql(raw)
        if not cleaned:
            raise RuntimeError("Could not extract SQL from LLM response.")
        return cleaned