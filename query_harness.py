"""
Interactive text-to-SQL harness for SQRL-9B, backed by a Snowflake warehouse.

Type a question -> SQRL reasons over the schema, optionally issues read-only
<sql> probes against Snowflake to inspect the data, then emits a final
<answer> query -> we execute it and print the results as a table.

SQRL was fine-tuned against SQLite (backtick identifier quoting, SQLite
functions). We keep the model's system prompt saying "SQLite" so it stays
in its trained distribution, then transpile whatever it emits into
Snowflake dialect with sqlglot right before execution. Transpilation is not
perfect for every query shape, so if you hit persistent failures on a
specific pattern, print `translated` (below) to see what's actually being
sent to Snowflake.

Required environment variables (see .env):
    SQRL_BASE_URL                     Modal server URL, e.g. https://you--sqrl-9b-inference-server.modal.direct
    SNOWFLAKE_ACCOUNT
    SNOWFLAKE_USER
    SNOWFLAKE_PRIVATE_KEY_PATH         path to a PKCS#8 .p8 private key file (key-pair auth)
    SNOWFLAKE_WAREHOUSE
    SNOWFLAKE_DATABASE
    SNOWFLAKE_SCHEMA
Optional:
    SNOWFLAKE_PRIVATE_KEY_PASSPHRASE   blank/unset if the key is unencrypted
    SNOWFLAKE_ROLE
    SNOWFLAKE_TABLE                    comma-separated table allowlist, bare or fully-qualified
                                        (default: all tables in the schema)
    SQRL_EVIDENCE                      free-text domain hints injected into the prompt
    SQRL_DEBUG                         set to 1 to print translated SQL, timings, and raw
                                        model reasoning as they happen

This harness runs entirely on your machine -- .env and the .p8 key are read
straight off local disk. Nothing here runs in a Modal container; only
serve.py's vLLM server does, and it never sees your Snowflake credentials.

IMPORTANT: SQRL generates SQL from natural language and we execute it
against a real warehouse. Point SNOWFLAKE_ROLE at a role with read-only
grants (SELECT only) and use a small warehouse for testing -- this script
also enforces a client-side SELECT/WITH-only guard as a second layer, but
that is not a substitute for database-level permissions.
"""

import os
import re
import sys
import time

import snowflake.connector
from cryptography.hazmat.primitives import serialization
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.table import Table

load_dotenv()

try:
    import sqlglot
    HAVE_SQLGLOT = True
except ImportError:
    HAVE_SQLGLOT = False

MODEL_NAME = "sqrl-9b"
MAX_STEPS = 5
PROBE_ROW_LIMIT = 50
ANSWER_ROW_LIMIT = 500
DEBUG = os.environ.get("SQRL_DEBUG", "").lower() in {"1", "true", "yes"}

console = Console()


def debug(msg: str) -> None:
    if DEBUG:
        console.print(f"[grey50]  debug:[/grey50] {msg}")


def env(name: str, default: str | None = None, required: bool = True) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        console.print(f"[red]Missing required environment variable: {name}[/red]")
        sys.exit(1)
    return value


def load_private_key(path: str, passphrase: str | None) -> bytes:
    """
    Read a PKCS#8 .p8 file and return DER bytes for snowflake-connector-python.
    """
    if not os.path.exists(path):
        console.print(f"[red]Private key file not found: {path}[/red]")
        sys.exit(1)
    with open(path, "rb") as f:
        pem_data = f.read()
    key = serialization.load_pem_private_key(
        pem_data,
        password=passphrase.encode() if passphrase else None,
    )
    return key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def build_schema_text(cur, database: str, schema: str, tables: list[str] | None) -> str:
    """
    Render Snowflake INFORMATION_SCHEMA as readable CREATE-TABLE listings."""
    query = f"""
        SELECT table_name, column_name, data_type, is_nullable
        FROM {database}.INFORMATION_SCHEMA.COLUMNS
        WHERE table_schema = %s
        ORDER BY table_name, ordinal_position
    """
    cur.execute(query, (schema,))
    rows = cur.fetchall()

    columns_by_table: dict[str, list[str]] = {}
    for table_name, column_name, data_type, is_nullable in rows:
        if tables and table_name.upper() not in tables:
            continue
        columns_by_table.setdefault(table_name, []).append(
            f"  {column_name} {data_type}" + ("" if is_nullable == "YES" else " NOT NULL")
        )

    if not columns_by_table:
        console.print(
            f"[red]No tables found in {database}.{schema} "
            f"(after applying SQRL_TABLES filter, if set).[/red]"
        )
        sys.exit(1)

    ddls = [
        f"CREATE TABLE {table} (\n" + ",\n".join(cols) + "\n);"
        for table, cols in columns_by_table.items()
    ]
    return "\n\n".join(ddls)


def build_system_prompt(schema: str, evidence: str) -> dict:
    content = f"""You are an expert data analyst, fluent in SQL, with a meticulous eye for \
matching a question's intent to the exact tables, columns, and stored value formats of a \
database.

Translate the user's natural-language question into a SQL query that answers it, using the \
database schema in <schema> and any domain hints in <evidence>.

Database engine: SQLite

<schema>
{schema}
</schema>
<evidence>
{evidence}
</evidence>

Think through the problem internally first. Then, in your response, write a BRIEF summary of \
your reasoning -- 2-4 sentences stating which tables and columns are relevant, the joins and \
filters, and any exact value-format detail.

Rules:
- Use only the tables and columns defined in <schema>.
- Quote identifiers containing spaces or special characters with backticks.
- Return exactly the requested columns.
- You have at most {MAX_STEPS} <sql> steps; after that you must give <answer>.
- Emit exactly one action block: <sql>...</sql> for a read-only exploration query, or \
<answer>...</answer> for your final query.
- The action block must be the LAST thing in your message; do not discuss the tags themselves.
"""
    return {"role": "system", "content": content}


READ_ONLY_RE = re.compile(r"^\s*(WITH|SELECT)\b", re.IGNORECASE)


def to_snowflake_sql(sqlite_sql: str) -> str:
    """
    Translate SQRL's SQLite-flavored SQL into Snowflake dialect.
    """
    if HAVE_SQLGLOT:
        try:
            return sqlglot.transpile(sqlite_sql, read="sqlite", write="snowflake")[0]
        except Exception as exc:  # noqa: BLE001 -- sqlglot raises various parser error types; any of them means "fall back"
            console.print(f"[yellow]sqlglot transpile failed ({exc}); falling back to raw SQL with backtick fix.[/yellow]")
    # Fallback: SQLite/MySQL-style backtick quoting -> Snowflake double-quote quoting.
    return re.sub(r"`([^`]*)`", r'"\1"', sqlite_sql)


def enforce_read_only(sql: str) -> None:
    if not READ_ONLY_RE.match(sql):
        raise ValueError("Refusing to execute a non-SELECT statement against Snowflake.")


def run_probe(cur, sql: str) -> str:
    translated = to_snowflake_sql(sql)
    enforce_read_only(translated)
    if "LIMIT" not in translated.upper():
        translated = f"{translated.rstrip(';')} LIMIT {PROBE_ROW_LIMIT}"
    debug(f"executing probe against Snowflake: {translated}")
    start = time.perf_counter()
    try:
        cur.execute(translated)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        debug(f"probe returned {len(rows)} row(s) in {time.perf_counter() - start:.2f}s")
        lines = ["\t".join(cols)]
        lines += ["\t".join(str(v) for v in row) for row in rows]
        return "\n".join(lines) if rows else "(0 rows)"
    except Exception as exc:  # noqa: BLE001 -- any driver error becomes an <observation> the model reads and reacts to
        debug(f"probe failed after {time.perf_counter() - start:.2f}s: {exc}")
        return f"ERROR: {exc}"


def run_answer(cur, sql: str):
    translated = to_snowflake_sql(sql)
    enforce_read_only(translated)
    if "LIMIT" not in translated.upper():
        translated = f"{translated.rstrip(';')} LIMIT {ANSWER_ROW_LIMIT}"
    debug(f"executing final SQL against Snowflake: {translated}")
    start = time.perf_counter()
    cur.execute(translated)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    debug(f"final query returned {len(rows)} row(s) in {time.perf_counter() - start:.2f}s")
    return cols, rows


def extract_last(tag: str, text: str) -> str | None:
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return matches[-1].strip() if matches else None


def ask(client: OpenAI, cur, system_prompt: dict, question: str):
    messages = [
        system_prompt,
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                "Reason about it, then give <sql> to investigate or <answer> to finish."
            ),
        },
    ]

    for step in range(MAX_STEPS + 1):
        label = f"step {step + 1}/{MAX_STEPS + 1}: waiting for SQRL..."
        start = time.perf_counter()
        with console.status(f"[bold blue]{label}[/bold blue]"):
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.7,
                top_p=0.95,
                max_tokens=8192,
                # Cut generation the instant an action block closes instead of
                # relying on the model to end its turn on its own -- without
                # this it can run all the way to max_tokens (minutes at ~44 tok/s).
                stop=["</sql>", "</answer>"],
                extra_body={"include_stop_str_in_output": True},
            )
        elapsed = time.perf_counter() - start
        usage = getattr(resp, "usage", None)
        usage_str = f", {usage.completion_tokens} tokens" if usage else ""
        finish_reason = resp.choices[0].finish_reason
        finish_str = f", finish_reason={finish_reason}" if finish_reason != "stop" else ""
        console.print(
            f"[blue]›[/blue] step {step + 1}/{MAX_STEPS + 1}: got response ({elapsed:.1f}s{usage_str}{finish_str})"
        )
        if finish_reason == "length":
            console.print(
                "[yellow]  hit max_tokens before an action block closed -- "
                "the model may be looping; treating this as a dead end.[/yellow]"
            )

        content = resp.choices[0].message.content
        if "</think>" in content:
            reasoning, content = content.split("</think>", 1)
            content = content.strip()
            debug(f"reasoning: {reasoning.strip()[:500]}")
        messages.append({"role": "assistant", "content": content})

        answer = extract_last("answer", content)
        if answer:
            console.print(f"[dim]  final SQL:[/dim] {answer}")
            return answer

        sql = extract_last("sql", content)
        if not sql:
            console.print("[red]Model produced neither <sql> nor <answer>; stopping.[/red]")
            debug(f"raw content: {content[:1000]}")
            return None

        console.print(f"[dim]  step {step + 1} probe:[/dim] {sql}")
        with console.status("[bold blue]executing probe against Snowflake...[/bold blue]"):
            obs = run_probe(cur, sql)
        console.print(f"[dim]  step {step + 1} observation:[/dim] {obs[:300]}")
        messages.append(
            {
                "role": "user",
                "content": f"<observation>\n{obs}\n</observation>\nContinue with <sql> or <answer>.",
            }
        )

    console.print("[yellow]Hit max probe steps without a final answer.[/yellow]")
    return None


def display_table(cols: list[str], rows: list[tuple]) -> None:
    table = Table(show_lines=False)
    for col in cols:
        table.add_column(col)
    for row in rows:
        table.add_row(*(str(v) if v is not None else "" for v in row))
    console.print(table)
    console.print(f"[dim]{len(rows)} row(s)[/dim]")


def main():
    base_url = env("SQRL_BASE_URL").rstrip("/") + "/v1"
    client = OpenAI(base_url=base_url, api_key="EMPTY")

    with console.status("[bold blue]connecting to Snowflake...[/bold blue]"):
        private_key_der = load_private_key(
            env("SNOWFLAKE_PRIVATE_KEY_PATH"),
            os.environ.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE") or None,
        )
        conn = snowflake.connector.connect(
            account=env("SNOWFLAKE_ACCOUNT"),
            user=env("SNOWFLAKE_USER"),
            private_key=private_key_der,
            warehouse=env("SNOWFLAKE_WAREHOUSE"),
            database=env("SNOWFLAKE_DATABASE"),
            schema=env("SNOWFLAKE_SCHEMA"),
            role=os.environ.get("SNOWFLAKE_ROLE"),
        )
        cur = conn.cursor()
    console.print("[blue]›[/blue] connected to Snowflake")

    # bare or fully-qualified (DB.SCHEMA.TABLE) names both work; we match on the last segment
    tables_env = os.environ.get("SNOWFLAKE_TABLE")
    tables = (
        [t.strip().split(".")[-1].upper() for t in tables_env.split(",")]
        if tables_env
        else None
    )
    evidence = os.environ.get("SQRL_EVIDENCE", "(none provided)")

    with console.status("[bold blue]introspecting Snowflake schema...[/bold blue]"):
        schema_text = build_schema_text(
            cur, env("SNOWFLAKE_DATABASE"), env("SNOWFLAKE_SCHEMA"), tables
        )
        system_prompt = build_system_prompt(schema_text, evidence)
    console.print(f"[blue]›[/blue] loaded schema ({len(schema_text.splitlines())} lines of DDL)")
    debug(f"schema:\n{schema_text}")

    if not HAVE_SQLGLOT:
        console.print(
            "[yellow]sqlglot not installed -- dialect translation will be a naive "
            "backtick-to-quote replace only. `pip install sqlglot` for better results.[/yellow]"
        )

    console.print("\n[bold]SQRL Text-to-SQL harness[/bold] (Ctrl-D or 'quit' to exit)\n")
    while True:
        try:
            question = console.input("[bold cyan]> [/bold cyan]").strip()
        except EOFError:
            break
        if not question or question.lower() in {"quit", "exit"}:
            break

        sql = ask(client, cur, system_prompt, question)
        if not sql:
            continue

        try:
            with console.status("[bold blue]executing final answer against Snowflake...[/bold blue]"):
                cols, rows = run_answer(cur, sql)
            display_table(cols, rows)
        except Exception as exc:  # noqa: BLE001 -- surface any execution failure to the REPL and keep looping
            console.print(f"[red]Execution failed: {exc}[/red]")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()