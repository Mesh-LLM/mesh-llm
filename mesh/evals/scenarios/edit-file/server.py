"""Simple HTTP API server for a todo list."""
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

todos = []

class TodoHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/todos':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(todos).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/todos':
            length = int(self.headers['Content-Length'])
            body = json.loads(self.rfile.read(length))
            # BUG: no validation, no id assignment, no error handling
            body['id'] = len(todos) + 1
            todos.append(body)
            self.send_response(200)
            self.end_headers()

    def do_DELETE(self):
        # BUG: not implemented but should be
        self.send_response(405)
        self.end_headers()

if __name__ == '__main__':
    server = HTTPServer(('localhost', 8080), TodoHandler)
    print('Serving on port 8080')
    server.serve_forever()
