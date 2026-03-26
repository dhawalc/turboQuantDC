#!/usr/bin/env python3
"""TurboQuantDC War Room — real-time agent dashboard server."""

import http.server
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs

PORT = 8811
ROOT = Path(__file__).parent
MESSAGES_FILE = ROOT / "messages.jsonl"


class WarRoomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self.path = "/index.html"
            return super().do_GET()

        elif parsed.path == "/api/messages":
            params = parse_qs(parsed.query)
            after = int(params.get("after", [0])[0])
            messages = self._read_messages()
            result = messages[after:]
            self._json_response({"messages": result, "total": len(messages)})

        else:
            return super().do_GET()

    def do_POST(self):
        if self.path == "/api/message":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            body.setdefault("timestamp", time.time())
            with open(MESSAGES_FILE, "a") as f:
                f.write(json.dumps(body) + "\n")
            self._json_response({"ok": True})

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_messages(self):
        if not MESSAGES_FILE.exists():
            return []
        msgs = []
        with open(MESSAGES_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        msgs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return msgs

    def log_message(self, fmt, *args):
        pass  # suppress request spam


if __name__ == "__main__":
    # Clear old messages on fresh start
    if MESSAGES_FILE.exists():
        MESSAGES_FILE.unlink()

    server = http.server.HTTPServer(("0.0.0.0", PORT), WarRoomHandler)
    print(f"\n  \033[1;33m⚡ TurboQuantDC War Room\033[0m")
    print(f"  \033[90mhttp://localhost:{PORT}\033[0m\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  War Room shut down.")
        server.server_close()
