CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS pricing(model TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS requests(
    id TEXT PRIMARY KEY, terms TEXT NOT NULL, state TEXT NOT NULL,
    cap INTEGER NOT NULL CHECK(cap>0), spent INTEGER NOT NULL CHECK(spent>=0)
);
CREATE TABLE IF NOT EXISTS charges(
    hash TEXT PRIMARY KEY, request_id TEXT NOT NULL REFERENCES requests(id),
    segment INTEGER NOT NULL, invoice TEXT NOT NULL, amount INTEGER NOT NULL,
    max_total INTEGER NOT NULL, state TEXT NOT NULL, total INTEGER NOT NULL,
    settled_day INTEGER, UNIQUE(request_id,segment)
);
CREATE TABLE IF NOT EXISTS receivables(
    hash TEXT PRIMARY KEY, request_id TEXT NOT NULL, peer TEXT NOT NULL,
    segment INTEGER NOT NULL, invoice TEXT NOT NULL, tokens INTEGER NOT NULL,
    state TEXT NOT NULL, UNIQUE(request_id,segment)
);

CREATE TABLE IF NOT EXISTS serving_requests(id TEXT PRIMARY KEY,peer TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS serving_accounting(
    id TEXT PRIMARY KEY REFERENCES serving_requests(id), pricing TEXT NOT NULL,
    max_output INTEGER NOT NULL, tokens INTEGER NOT NULL DEFAULT 0,
    finished INTEGER NOT NULL DEFAULT 0, forgiven INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS receivables_peer_state ON receivables(peer,state);
CREATE INDEX IF NOT EXISTS serving_requests_peer ON serving_requests(peer);
CREATE INDEX IF NOT EXISTS charges_state ON charges(state);
