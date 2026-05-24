# Insightiq-Analysis-Assistant

Insightiq-Analysis-Assistant is an AI orchestration project that turns natural language business analysis questions into SQL queries, executes those queries against PostgreSQL, and then converts the results into plain-English insight summaries.

## Objective

The goal of this project is to provide an end-to-end root cause analysis (RCA) assistant for analytics teams. It combines:

- vector-based grounding over database metadata and business rules,
- scenario identification to understand the user intent,
- LLM-driven SQL generation,
- PostgreSQL execution of read-only queries,
- optional narrative summarization of results.

This makes it easier for non-technical users to ask business questions in plain language and get useful analytics output quickly.

## How the system works

The core orchestrator is `orchestrator.py`. When you run it, Flask starts a web UI and exposes a REST API for query processing.

### Agents and orchestration flow

1. `ScenarioIdentifierAgent` loads `scenario_knowledge_base.json` and syncs it into a dedicated Qdrant collection.
   - It converts example user phrases into vectors.
   - It identifies the best matching scenario for each incoming request.
   - If the match is strong enough, it appends scenario-specific instructions to the query before SQL generation.

2. `TextToSQLAgent` uses the entity knowledge base stored in Qdrant (`db_knowledge_base.json`) to ground the LLM.
   - It retrieves the top relevant database/entity chunks for the user request.
   - It sends the user query plus grounding context to the GROQ/LLM model and extracts a single SQL query.

3. `PostgresConnector` executes the generated SQL against PostgreSQL if enabled.
   - Only read-only `SELECT` / `WITH` style queries are allowed.
   - Results are returned as preview rows plus CSV-ready data.

4. `OutputAnalyserAgent` can generate a plain-English narrative explaining the query results.
   - It receives the original user input, query, scenario instruction, and query CSV data.
   - It returns a short business-friendly summary.

### UI behavior

When the Flask app starts, the static UI is served from `/`.

- User inputs are sent to the `/api/query` endpoint.
- The orchestrator resolves the request, identifies the scenario, generates SQL, runs the query, and returns results.
- The UI displays SQL, query results, and narrative output if available.

## Dataset files

The project uses two JSON files as knowledge sources:

### `db_knowledge_base.json`

This file contains metadata about database entities, columns, business rules, and functions. It is used by the entity knowledge base ingestion script and by the `TextToSQLAgent` grounding process.

Each entry should include:

- `entity_type`: one of `table`, `stored_proc`, `function`, or `business_rule`
- `entity_name`: the table, object, or rule name
- `description`: a plain-language explanation of the entity or rule
- `column_name`: optional column name when the entity describes a specific column
- `tags`: optional list of tags such as `["sales", "finance"]`
- `source`: optional source string like `"handwritten"`, `"ddl"`, or `"wiki"`

Example:

```json
{
  "entity_type": "table",
  "entity_name": "orders",
  "description": "Contains every customer order with status, amount, and order date.",
  "tags": ["sales", "orders"],
  "source": "warehouse"
}
```

### `scenario_knowledge_base.json`

This file defines user intent scenarios and scenario-specific guidance for generating SQL.

Each scenario should include:

- `scenario_key`: a unique identifier for the scenario
- `scenario_user_input`: a list of example user questions or phrasing patterns
- `scenario_instruction`: the instructions that should guide SQL construction when this scenario matches
- `scenario_output`: an object describing the expected output shape
  - `type`: output type, for example `aggregated_metrics`
  - `expected_columns`: expected column names or output fields
  - `cardinality`: `single_or_multiple_rows`

Example:

```json
{
  "scenario_key": "revenue_by_region",
  "scenario_user_input": [
    "Show me revenue by region",
    "What is the total sales by sales region?"
  ],
  "scenario_instruction": "Generate SQL that groups revenue by region and filters to completed orders.",
  "scenario_output": {
    "type": "aggregated_metrics",
    "expected_columns": ["region", "total_revenue"],
    "cardinality": "multiple_rows"
  }
}
```

## Setup

1. Create and activate a Python environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install required packages.

```bash
python -m pip install Flask python-dotenv qdrant-client sentence-transformers groq psycopg requests httpx
```

3. Configure environment variables in a `.env` file.

At minimum, set values for:

```text
FLASK_HOST=127.0.0.1
FLASK_PORT=5001
FLASK_DEBUG=true
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=sql-grounding-kb
SCENARIO_QDRANT_URL=http://localhost:6333
SCENARIO_QDRANT_API_KEY=
SCENARIO_QDRANT_COLLECTION=rca_agent_classifier
SCENARIO_KNOWLEDGE_BASE_JSON=scenario_knowledge_base.json
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
POSTGRES_ENABLED=true
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=ecommerce_analytics
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
```

4. Start Qdrant and PostgreSQL.

- For Qdrant locally:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

- For PostgreSQL, use your existing database or start a local instance.

5. Load the entity knowledge base into Qdrant.

```bash
python qdrant_entity_kb.py
```

This script reads `db_knowledge_base.json`, chunks the entity text, and uploads it into the Qdrant collection.

## Running the orchestrator

Run the Flask-based orchestrator with:

```bash
python orchestrator.py
```

Then open the UI in your browser at:

```text
http://127.0.0.1:5001
```

From there, user input drives orchestration across agents:

- the scenario agent classifies the intent,
- the text-to-SQL agent generates a grounded query,
- PostgreSQL executes the query,
- and the optional analyser turns results into a business-friendly summary.

## Tips for filling the dataset

- In `db_knowledge_base.json`, give each entity a clear description and include column-level context when available.
- Use `tags` to group related entities and make grounding easier.
- In `scenario_knowledge_base.json`, provide multiple example user inputs for each scenario so the vector matcher can correctly identify intent.
- Keep `scenario_instruction` focused on how the SQL should be structured and which tables or filters matter.
- Update `scenario_output.expected_columns` when you know the desired output schema.

## Notes

- The orchestrator will automatically sync scenario definitions into the scenario Qdrant collection on startup.
- If PostgreSQL execution is disabled via `POSTGRES_ENABLED=false`, the flow still generates SQL but skips database execution.
- Keep the JSON files valid arrays of objects. The system validates required fields and will raise errors for malformed entries.
