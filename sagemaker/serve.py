"""SageMaker inference server: GET /ping, POST /invocations on port 8080.

SageMaker starts the container as `docker run <image> serve`; the argument is ignored.
Nothing a person sends is logged or kept.
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import form


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ping":
            self.reply(200, {"status": "ok"})
        else:
            self.reply(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/invocations":
            self.reply(404, {"error": "not found"})
            return
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.reply(200, form.answer(json.loads(body)))
        except (ValueError, json.JSONDecodeError) as error:
            self.reply(400, {"error": str(error)})

    def reply(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # the default logs every request line; we keep no record of visitors


if __name__ == "__main__":
    print("model loaded, serving on :8080", flush=True)
    HTTPServer(("", 8080), Handler).serve_forever()
