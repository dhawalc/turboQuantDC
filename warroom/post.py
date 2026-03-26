#!/usr/bin/env python3
"""Post a message to the TurboQuantDC War Room.

Usage:
    python warroom/post.py <agent> <message> [type]

Types: system, research, finding, consensus, warning, update (default)
"""

import json
import sys
import time
from pathlib import Path

MESSAGES_FILE = Path(__file__).parent / "messages.jsonl"


def post(agent: str, content: str, msg_type: str = "update"):
    """Append a message to the war room log."""
    msg = {
        "agent": agent,
        "content": content,
        "type": msg_type,
        "timestamp": time.time(),
    }
    with open(MESSAGES_FILE, "a") as f:
        f.write(json.dumps(msg) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python warroom/post.py <agent> <message> [type]")
        sys.exit(1)
    post(
        agent=sys.argv[1],
        content=sys.argv[2],
        msg_type=sys.argv[3] if len(sys.argv) > 3 else "update",
    )
