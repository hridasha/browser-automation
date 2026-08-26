import logging
import os
import sys

_CONFIGURED = False


def configure_logging() -> None:
    """Idempotent: safe to call from every entrypoint (CLI, API, MCP server).

    Logs go to stderr, not stdout — this matters for mcp_server/server.py in particular,
    since its stdio transport uses stdout as the JSON-RPC wire; anything printed to stdout
    there would corrupt the protocol stream.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    _CONFIGURED = True
