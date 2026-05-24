import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request, send_from_directory, url_for

from analysis_session import extend_session, normalize_session, resolve_analysis_query
from frontend_ui import render_index_html
from scenario_identifier_agent import (
    SCENARIO_CONTEXT_MIN_SIMILARITY,
    ScenarioIdentifierAgent,
)
from postgres_connector import PostgresConnector, load_postgres_config_from_env
from text_to_sql_agent import TextToSQLAgent
from output_analyser import OutputAnalyserAgent


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

try:
    import sqlparse
except ImportError:  # pragma: no cover
    sqlparse = None  # type: ignore[misc, assignment]


def format_sql_for_display(sql: str) -> str:
    """Pretty-print SQL for the UI (indent + uppercase keywords). Falls back to raw SQL."""
    if not sql or not str(sql).strip():
        return sql
    if sqlparse is None:
        return sql
    try:
        return sqlparse.format(
            str(sql).strip(),
            reindent=True,
            keyword_case="upper",
            strip_comments=False,
        )
    except Exception:
        return sql


def env(name: str, default: str = "") -> str:
    value = os.getenv(name, default).strip()
    return value


def resolve_entities_json(path_str: str) -> str:
    """Entity JSON paths are relative to this package directory unless absolute."""
    p = Path(path_str)
    if not p.is_absolute():
        p = BASE_DIR / p
    return str(p)


# Existing text_to_sql_agent Qdrant cluster (kept backward compatible).
QDRANT_URL = env("TEXT2SQL_QDRANT_URL", env("QDRANT_URL"))
QDRANT_API_KEY = env("TEXT2SQL_QDRANT_API_KEY", env("QDRANT_API_KEY"))
QDRANT_COLLECTION = env("TEXT2SQL_QDRANT_COLLECTION", env("QDRANT_COLLECTION", "sql-grounding-kb"))
EMBEDDING_MODEL = env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_API_KEY = env("GROQ_API_KEY")
GROQ_MODEL = env("GROQ_MODEL", "openai/gpt-oss-120b")
TOP_K = int(env("TOP_K", "40"))

# Dedicated scenario_identifier Qdrant cluster.
SCENARIO_QDRANT_URL = env("SCENARIO_QDRANT_URL")
SCENARIO_QDRANT_API_KEY = env("SCENARIO_QDRANT_API_KEY")
SCENARIO_QDRANT_COLLECTION = env("SCENARIO_QDRANT_COLLECTION", "rca_agent_classifier")
SCENARIO_EMBEDDING_MODEL = env("SCENARIO_EMBEDDING_MODEL", EMBEDDING_MODEL)
SCENARIO_KNOWLEDGE_BASE_JSON = resolve_entities_json(
    env("SCENARIO_KNOWLEDGE_BASE_JSON", "scenario_knowledge_base.json")
)


app = Flask(
    __name__,
    static_folder=str(BASE_DIR / "static"),
    static_url_path="/static",
)
agent = TextToSQLAgent(
    qdrant_url=QDRANT_URL,
    qdrant_api_key=QDRANT_API_KEY,
    qdrant_collection=QDRANT_COLLECTION,
    embedding_model=EMBEDDING_MODEL,
    groq_api_key=GROQ_API_KEY,
    groq_model=GROQ_MODEL,
    top_k=TOP_K,
)
scenario_identifier = ScenarioIdentifierAgent(
    qdrant_url=SCENARIO_QDRANT_URL,
    qdrant_api_key=SCENARIO_QDRANT_API_KEY,
    qdrant_collection=SCENARIO_QDRANT_COLLECTION,
    embedding_model=SCENARIO_EMBEDDING_MODEL,
    scenarios_json_path=SCENARIO_KNOWLEDGE_BASE_JSON,
)
_ = scenario_identifier.sync_scenarios()

postgres_config = load_postgres_config_from_env()
postgres_connector = PostgresConnector(postgres_config)

# Second-pass LLM agent: generates plain-English narratives from query results.
output_analyser = OutputAnalyserAgent(
    groq_api_key=GROQ_API_KEY,
    groq_model=GROQ_MODEL,
)


@app.route("/insightiq_logo.png")
def insightiq_logo_image() -> Any:
    """Serve Insightiq logo from project root or static/."""
    for folder in (BASE_DIR, BASE_DIR / "static"):
        path = folder / "insightiq_logo.png"
        if path.is_file():
            return send_from_directory(path.parent, path.name, mimetype="image/png")
    abort(404)


@app.route("/", methods=["GET", "POST", "HEAD"])
def index() -> str:
    """
    POST is accepted for backward compatibility (older HTML forms posted to `/`).
    The SPA only needs GET; POST/HEAD return the same page.
    """
    logo_url = url_for("insightiq_logo_image")
    return render_index_html(logo_url=logo_url)


@app.route("/api/query", methods=["POST", "GET", "OPTIONS"])
def api_query() -> Any:
    if request.method == "OPTIONS":
        return ("", 204)
    if request.method == "GET":
        r = jsonify(
            {
                "error": (
                    'Use POST with JSON: {"query": "..."} for a new analysis, or '
                    '{"follow_up": "...", "session": [{"user_input": "...", "sql": "..."}]} '
                    "to continue with prior context."
                ),
                "logs": [],
            }
        )
        r.headers["Allow"] = "POST, OPTIONS"
        return r, 405

    payload = request.get_json(silent=True) or {}
    logs: List[Dict[str, str]] = []

    def add_log(level: str, msg: str) -> None:
        logs.append({"level": level, "msg": msg})

    follow_up_text = str(payload.get("follow_up") or "").strip()
    is_follow_up = bool(follow_up_text)
    prior_session_raw = payload.get("session") if is_follow_up else []

    try:
        user_query = resolve_analysis_query(payload)
    except ValueError as exc:
        add_log("error", str(exc))
        return jsonify({"error": str(exc), "logs": logs}), 400

    if is_follow_up:
        prior_session = normalize_session(prior_session_raw if isinstance(prior_session_raw, list) else [])
        add_log(
            "info",
            f"[RCA_FLOW] Follow-up analysis: {len(prior_session)} prior turn(s) in session, "
            f"latest follow-up len={len(follow_up_text)}, combined prompt len={len(user_query)}.",
        )
        for idx, turn in enumerate(prior_session, start=1):
            add_log(
                "info",
                f"[RCA_FLOW] Session turn {idx}: user_input len={len(turn['user_input'])}, "
                f"sql len={len(turn['sql'])}.",
            )
    else:
        add_log("info", f"[RCA_FLOW] New analysis query (len={len(user_query)}).")

    add_log(
        "info",
        f"[RCA_FLOW] Scenario KB: {scenario_identifier.scenarios_json_path}",
    )
    add_log("info", f"[RCA_FLOW] Resolved input for pipeline:\n{user_query}")

    try:
        scenario_match = scenario_identifier.identify(user_query)
        add_log(
            "info",
            "[RCA_FLOW] Cosine similarity score (top-1 scenario vector): "
            f"{scenario_match.score:.6f}",
        )
        add_log(
            "info",
            "[RCA_FLOW] Vector KB utterance that produced this score: "
            f"{scenario_match.matched_user_input!r}",
        )
        add_log(
            "info",
            "[RCA_FLOW] Chosen scenario_key: "
            f"{scenario_match.scenario_key!r} | scenario_instruction: "
            f"{scenario_match.scenario_instruction!r}",
        )

        # Capture the scenario instruction for downstream use (analyser + response payload).
        scenario_instruction: str = ""
        if scenario_match.score >= SCENARIO_CONTEXT_MIN_SIMILARITY:
            scenario_instruction = scenario_match.scenario_instruction or ""

        text_to_sql_query = scenario_identifier.query_for_text_to_sql(user_query, scenario_match)
        if scenario_match.score < SCENARIO_CONTEXT_MIN_SIMILARITY:
            add_log(
                "info",
                "[RCA_FLOW] Similarity below "
                f"{SCENARIO_CONTEXT_MIN_SIMILARITY:.2f}; sending raw user input to TextToSQL "
                "(no scenario instructions appended).",
            )
        add_log(
            "info",
            "[RCA_FLOW] TextToSQLAgent.run() received query "
            f"(len={len(text_to_sql_query)}):\n{text_to_sql_query}",
        )

        result = agent.run(text_to_sql_query)
        sql = result["sql"]
        context_chunks = result["context_chunks"]

        vectors = [
            {
                "text": str(c.get("text", "")),
                "score": float(c.get("score", 0)),
                "source": str(c.get("entity_id", "")),
            }
            for c in context_chunks
        ]
        add_log("ok", f"Retrieved {len(vectors)} top-K grounding chunks from the entity KB.")
        add_log("ok", "SQL generation completed.")

        sql_display = format_sql_for_display(sql)
        query_result: Dict[str, Any] | None = None

        if postgres_config.enabled:
            add_log("info", "Executing SQL against PostgreSQL…")
            try:
                pg_result = postgres_connector.run_query(sql)
                query_result = pg_result.to_api_dict()
                suffix = " (truncated to max row limit)" if pg_result.truncated else ""
                add_log(
                    "ok",
                    f"PostgreSQL returned {pg_result.row_count} row(s); "
                    f"showing top {min(postgres_config.preview_rows, pg_result.row_count)} in UI{suffix}.",
                )
            except Exception as pg_err:
                add_log("error", f"PostgreSQL execution failed: {pg_err}")
                query_result = {"error": str(pg_err)}
        else:
            add_log("info", "PostgreSQL execution skipped (POSTGRES_ENABLED=false).")

        if is_follow_up:
            updated_session = extend_session(
                prior_session_raw if isinstance(prior_session_raw, list) else [],
                user_input=follow_up_text,
                sql=sql_display,
            )
        else:
            updated_session = extend_session([], user_input=user_query, sql=sql_display)

        add_log("info", f"[RCA_FLOW] Session persisted with {len(updated_session)} turn(s).")

        response_body: Dict[str, Any] = {
            "response": "Generated SQL and query results."
            if query_result and "error" not in query_result
            else "Generated SQL.",
            "sql_query": sql_display,
            # Expose the effective resolved input and scenario instruction so the
            # frontend can pass them verbatim to /api/analyse without a second
            # round-trip through scenario matching.
            "effective_user_input": user_query,
            "scenario_instruction": scenario_instruction,
            "session": updated_session,
            "logs": logs,
            "vectors": vectors,
        }
        if query_result is not None:
            response_body["query_result"] = query_result

        return jsonify(response_body)
    except Exception as e:
        add_log("error", str(e))
        return jsonify({"error": str(e), "logs": logs}), 500


@app.route("/api/analyse", methods=["POST", "OPTIONS"])
def api_analyse() -> Any:
    """Second-pass LLM call: generates a plain-English narrative from SQL results.

    Expected JSON body:
    {
        "sql_query":            "<formatted SQL>",
        "user_input":           "<effective resolved user question>",
        "scenario_instruction": "<scenario instruction string or empty>",
        "csv_data":             "<full CSV string of query results>"
    }

    Returns:
    {
        "narrative": "<prose narrative>",
        "logs":      [{"level": "...", "msg": "..."}]
    }
    """
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    logs: List[Dict[str, str]] = []

    def add_log(level: str, msg: str) -> None:
        logs.append({"level": level, "msg": msg})

    sql_query = str(payload.get("sql_query") or "").strip()
    user_input = str(payload.get("user_input") or "").strip()
    scenario_instruction = str(payload.get("scenario_instruction") or "").strip()
    csv_data = str(payload.get("csv_data") or "").strip()

    if not sql_query:
        add_log("error", "Missing required field: sql_query")
        return jsonify({"error": "sql_query is required", "logs": logs}), 400

    add_log(
        "info",
        f"[ANALYSER] Narrative request received — "
        f"sql_query len={len(sql_query)}, csv_data len={len(csv_data)}, "
        f"scenario_instruction len={len(scenario_instruction)}.",
    )

    try:
        narrative = output_analyser.analyse(
            user_input=user_input,
            sql_query=sql_query,
            scenario_instruction=scenario_instruction,
            csv_data=csv_data,
        )
        add_log("ok", "[ANALYSER] Narrative generation completed successfully.")
        return jsonify({"narrative": narrative, "logs": logs})
    except Exception as exc:
        add_log("error", f"[ANALYSER] Narrative generation failed: {exc}")
        return jsonify({"error": str(exc), "logs": logs}), 500


if __name__ == "__main__":
    host = env("FLASK_HOST", "127.0.0.1")
    port = int(env("FLASK_PORT", "5001"))
    debug = env("FLASK_DEBUG", "true").lower() in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug)