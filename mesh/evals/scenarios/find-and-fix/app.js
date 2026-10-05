// Express-like mini framework + app with bugs
const http = require('http');
const fs = require('fs');
const path = require('path');

class Router {
  constructor() {
    this.routes = [];
  }

  get(pattern, handler) {
    this.routes.push({ method: 'GET', pattern, handler });
  }

  post(pattern, handler) {
    this.routes.push({ method: 'POST', pattern, handler });
  }

  match(method, url) {
    for (const route of this.routes) {
      if (route.method !== method) continue;
      // BUG 1: pattern matching doesn't handle path params
      if (route.pattern === url) {
        return { handler: route.handler, params: {} };
      }
    }
    return null;
  }
}

const router = new Router();
const users = new Map();
let nextId = 1;

// List users
router.get('/users', (req, res) => {
  const list = Array.from(users.values());
  res.writeHead(200, { 'Content-Type': 'application/json' });
  // BUG 2: circular reference possible if user objects reference each other
  res.end(JSON.stringify(list));
});

// Create user
router.post('/users', async (req, res) => {
  let body = '';
  req.on('data', chunk => body += chunk);
  req.on('end', () => {
    const data = JSON.parse(body); // BUG 3: no try/catch, crashes on invalid JSON
    const user = { id: nextId++, ...data, createdAt: Date.now() };
    users.set(user.id, user);
    // BUG 4: should return 201, not 200
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(user));
  });
});

// Get user by ID
router.get('/users/:id', (req, res, params) => {
  const user = users.get(parseInt(params.id));
  if (!user) {
    // BUG 5: sends 200 instead of 404
    res.writeHead(200);
    res.end('Not found');
    return;
  }
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(user));
});

const server = http.createServer((req, res) => {
  const match = router.match(req.method, req.url);
  if (match) {
    match.handler(req, res, match.params);
  } else {
    res.writeHead(404);
    res.end('Not found');
  }
});

server.listen(3000, () => console.log('Server on :3000'));
