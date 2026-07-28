import json
import time
from typing import Any

import aiohttp
import modal

MODEL_NAME = "feyninc/sqrl-9b"
SERVED_MODEL_NAME = "sqrl-9b"
FAST_BOOT = False

# context window SQRL was tuned for; also caps KV cache sizing.
MAX_MODEL_LEN = 32768

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",  # faster model transfers
            "VLLM_LOG_STATS_INTERVAL": "1",  # more frequent metrics logging
        }
    )
)

app = modal.App("sqrl-9b-inference")

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000


@app.server(
    image=vllm_image,
    gpu=f"L40S:{N_GPU}",
    scaledown_window=15 * MINUTES,
    startup_timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    port=VLLM_PORT,
    target_concurrency=100,
    unauthenticated=True,  # relies on an unguessable URL; see README before exposing real data
)
class Server:
    @modal.enter()
    def start(self):
        import subprocess

        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--served-model-name",
            SERVED_MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--uvicorn-log-level=info",
            "--gpu-memory-utilization",
            "0.90",
            "--max-model-len",
            str(MAX_MODEL_LEN),
        ]

        # enforce-eager disables both Torch compilation and CUDA graph capture
        cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
        cmd += ["--tensor-parallel-size", str(N_GPU)]

        # IMPORTANT: do not add --reasoning-parser or --enable-auto-tool-choice.
        # SQRL emits its <sql>/<answer> action tags *after* an inline </think>
        # scratchpad in the same message; a reasoning parser strips everything
        # up to and including </think>, which would delete the action tags too.

        print(*cmd)

        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()


@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES):
    """Smoke test: confirm the server boots and answers one text-to-SQL question."""
    import asyncio

    url = await Server.get_url.aio()

    schema = (
        "CREATE TABLE employees (id INTEGER, name TEXT, department TEXT, salary INTEGER);"
    )
    system_prompt = {
        "role": "system",
        "content": (
            "You are an expert data analyst, fluent in SQL. Translate the "
            "user's question into a SQL query using the schema below.\n\n"
            f"<schema>\n{schema}\n</schema>\n<evidence>(none provided)</evidence>\n\n"
            "Respond with a brief reasoning summary, then exactly one action "
            "block: <sql>...</sql> to explore, or <answer>...</answer> to finish. "
            "The action block must be the last thing in your message."
        ),
    }
    messages = [
        system_prompt,
        {
            "role": "user",
            "content": (
                "Question: What is the average salary in the Engineering department?\n"
                "Reason about it, then give <sql> to investigate or <answer> to finish."
            ),
        },
    ]

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Running health check for server at {url}")
        deadline = time.time() + test_timeout - 1 * MINUTES
        while time.time() < deadline:
            async with session.get(
                "/health", timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status == 200:
                    break
                if resp.status == 503:
                    await asyncio.sleep(1)
                    continue
                assert False, f"Failed health check for server at {url}: HTTP {resp.status}"
        else:
            assert False, f"Failed health check for server at {url}"
        print(f"Successful health check for server at {url}")

        payload: dict[str, Any] = {
            "messages": messages,
            "model": SERVED_MODEL_NAME,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 1024,
            "stream": True,
        }
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

        print(f"Sending question to {url}")
        async with session.post(
            "/v1/chat/completions", json=payload, headers=headers
        ) as resp:
            resp.raise_for_status()
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line or line == "data: [DONE]":
                    continue
                line = line.removeprefix("data: ")
                chunk = json.loads(line)
                delta = chunk["choices"][0]["delta"]
                content = delta.get("content")
                if content:
                    print(content, end="")
        print()
