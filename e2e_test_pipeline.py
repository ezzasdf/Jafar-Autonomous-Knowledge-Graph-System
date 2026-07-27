"""
End-to-end test for the full RuleEngine → Qwen inference pipeline.

Starts llama-server.exe, sends a prompt via HTTP (stream:false, max_tokens=50),
then runs a prompt through the jafar_cli pipeline (RuleEngine + TransformerReasoningEngine)
to verify the whole chain works.

Usage:
    python e2e_test_pipeline.py
"""
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("e2e")

BASE = os.path.dirname(os.path.abspath(__file__))
SERVER_PATH = os.path.join(BASE, "llama", "llama-server.exe")
MODEL_PATH = os.path.join(BASE, "ai model", "Qwen2.5-7B-Instruct-Uncensored.Q4_K_M.gguf")
PORT = 8080
BASE_URL = f"http://127.0.0.1:{PORT}"

_FAILURES = 0
def _check(ok: bool, msg: str):
    global _FAILURES
    if ok:
        log.info(f"  PASS: {msg}")
    else:
        log.info(f"  FAIL: {msg}")
        _FAILURES += 1


def _start_server() -> subprocess.Popen | None:
    if not os.path.isfile(SERVER_PATH):
        log.error(f"llama-server.exe not found at {SERVER_PATH}")
        return None
    if not os.path.isfile(MODEL_PATH):
        log.error(f"Model not found at {MODEL_PATH}")
        return None

    proc = subprocess.Popen(
        [SERVER_PATH, "-m", MODEL_PATH, "--host", "127.0.0.1",
         "--port", str(PORT), "-t", "4", "-c", "2048", "-np", "1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )

    log.info("Waiting for llama-server to become ready...")
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            r = urllib.request.urlopen(f"{BASE_URL}/health", timeout=5)
            if r.status == 200:
                log.info(f"  Server ready (port {PORT})")
                return proc
        except Exception:
            pass
        time.sleep(1)

    log.error("Server failed to become ready within timeout")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    return None


def _stop_server(proc: subprocess.Popen | None):
    if proc is None:
        return
    log.info("Stopping llama-server...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
    log.info("  Server stopped")


# ------------------------------------------------------------------
#  Test 1: Direct HTTP API (stream: false, max_tokens=50)
# ------------------------------------------------------------------
def test_direct_http_api():
    log.info("\n=== Test 1: Direct HTTP API (stream:false, n_predict=50) ===")

    payload = json.dumps({
        "prompt": "What is the capital of France? Answer in one word.",
        "n_predict": 50,
        "temperature": 0.1,
        "stream": False,
        "stop": ["</s>", "Q:"],
    }).encode()

    req = urllib.request.Request(
        f"{BASE_URL}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=300)
        data = json.loads(resp.read().decode())
        content = data.get("content", "").strip()
        elapsed = time.time() - t0
        log.info(f"  Response: {content!r}")
        log.info(f"  Time: {elapsed:.1f}s")
        _check(bool(content) and len(content) > 0, "HTTP API returned non-empty response")
        _check("paris" in content.lower(), "Response mentions Paris")
    except Exception as e:
        elapsed = time.time() - t0
        log.error(f"  Request failed after {elapsed:.1f}s: {e}")
        _check(False, "HTTP API request succeeded")


# ------------------------------------------------------------------
#  Test 2: Full pipeline (RuleEngine fallthrough -> Qwen)
# ------------------------------------------------------------------
def test_full_pipeline():
    log.info("\n=== Test 2: Full pipeline (RuleEngine fallthrough -> Qwen) ===")

    sys.path.insert(0, BASE)

    try:
        from rule_engine import RuleEngine
        from transformer_reasoning import TransformerReasoningEngine
        from config import GGUF_CONFIG, TRANSFORMER_REASONING_CONFIG
    except ImportError as e:
        log.error(f"Failed to import pipeline modules: {e}")
        _check(False, "Import pipeline modules")
        return

    raw_llm = TransformerReasoningEngine(
        memory_system=None,
        model_name=GGUF_CONFIG.get("model_path", ""),
        max_new_tokens=50,
        temperature=0.1,
        use_gguf=True,
    )

    if not raw_llm._load():
        log.error("Failed to load TransformerReasoningEngine")
        _check(False, "TransformerReasoningEngine loads")
        return

    _check(raw_llm._loaded, "TransformerReasoningEngine loaded")

    def llm_generator(prompt: str, temperature: float = 0.1, max_tokens: int = 50) -> str:
        old_temp = raw_llm.temperature
        old_tokens = raw_llm.max_new_tokens
        raw_llm.temperature = temperature
        raw_llm.max_new_tokens = max_tokens
        try:
            result = raw_llm._generate(prompt)
            return result or ""
        except Exception as e:
            log.error(f"  LLM generate error: {e}")
            return ""
        finally:
            raw_llm.temperature = old_temp
            raw_llm.max_new_tokens = old_tokens

    engine = RuleEngine(llm_generator=llm_generator, confidence_threshold=0.65)

    prompt = "Explain what a Python decorator is in simple terms."
    t0 = time.time()
    response = engine.decide(prompt, temperature=0.1, max_tokens=50)
    elapsed = time.time() - t0

    log.info(f"  Prompt: {prompt}")
    log.info(f"  Response: {response!r}")
    log.info(f"  Time: {elapsed:.1f}s")

    _check(bool(response) and len(response) > 0, "Pipeline returned non-empty response")
    _check(len(response) > 10, "Pipeline returned substantive response (not a rule match)")

    raw_llm.unload()


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------
def main():
    log.info("=" * 60)
    log.info("Jafar Pipeline E2E Test")
    log.info(f"Server: {SERVER_PATH}")
    log.info(f"Model:  {MODEL_PATH}")
    log.info("=" * 60)

    proc = _start_server()
    if proc is None:
        log.error("Cannot start server — aborting")
        sys.exit(1)

    try:
        test_direct_http_api()
        test_full_pipeline()
    finally:
        _stop_server(proc)

    log.info("")
    log.info(f"Results: {_FAILURES} failure(s)")
    if _FAILURES:
        sys.exit(1)
    log.info("All e2e tests passed.")


if __name__ == "__main__":
    main()
