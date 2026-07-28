# SQRL-9B on Modal

Serves Feyn's SQRL-9B text-to-SQL model with vLLM on Modal, plus a local CLI
harness that asks it questions against a Snowflake warehouse.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 1. Deploy the model to Modal

```bash
modal deploy serve.py
```

This prints a URL like `https://<workspace>--sqrl-9b-inference-server.us-east.modal.direct`
once deployed. The first request after a deploy (or after 15 idle minutes) takes
a few minutes to cold-start while vLLM loads the model; see `serve.py` for the
`FAST_BOOT` toggle if you want faster boots at the cost of generation speed.

## 2. Configure environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Required:

| Variable | Description |
|---|---|
| `SQRL_BASE_URL` | The URL printed by `modal deploy` above |
| `SNOWFLAKE_ACCOUNT` | Your Snowflake account identifier |
| `SNOWFLAKE_USER` | Snowflake username |
| `SNOWFLAKE_PRIVATE_KEY_PATH` | Path to your `.p8` key-pair auth private key |
| `SNOWFLAKE_WAREHOUSE` | Warehouse to run queries on |
| `SNOWFLAKE_DATABASE` | Database to introspect and query |
| `SNOWFLAKE_SCHEMA` | Schema to introspect and query |

Optional: `SNOWFLAKE_PRIVATE_KEY_PASSPHRASE`, `SNOWFLAKE_ROLE`, `SNOWFLAKE_TABLE`
(comma-separated allowlist), `SQRL_EVIDENCE` (domain hints), `SQRL_DEBUG=1`
(verbose logging). See the top of `query_harness.py` for details.

**Use a read-only Snowflake role.** The harness generates and executes SQL
from natural language against a real warehouse; a client-side guard only
allows `SELECT`/`WITH` statements, but that's not a substitute for
database-level permissions.

## 3. Run the query harness

```bash
python query_harness.py
```

Type a question, SQRL reasons over your schema (optionally probing Snowflake
first), and the final query's results print as a table.
