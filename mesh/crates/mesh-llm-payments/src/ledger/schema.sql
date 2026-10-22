CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS pricing(model TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS model_deposits(
    model TEXT PRIMARY KEY REFERENCES pricing(model) ON DELETE CASCADE,
    deposit_msat INTEGER NOT NULL
);
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
CREATE INDEX IF NOT EXISTS charges_state ON charges(state);

CREATE TABLE IF NOT EXISTS metered_requests(
    id TEXT PRIMARY KEY, peer TEXT NOT NULL, model TEXT NOT NULL DEFAULT '',
    pricing TEXT NOT NULL,
    require_deposit INTEGER NOT NULL, deposit_msat INTEGER NOT NULL,
    dispatched INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL, outcome TEXT, bill_msat INTEGER,
    deposit_paid_msat INTEGER NOT NULL DEFAULT 0, remaining_msat INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS metered_invoices(
    hash TEXT PRIMARY KEY, request_id TEXT NOT NULL REFERENCES metered_requests(id),
    peer TEXT NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('deposit','final')),
    invoice TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN ('unpaid','paid','forgiven')),
    UNIQUE(request_id,kind)
);
CREATE INDEX IF NOT EXISTS metered_requests_peer ON metered_requests(peer,state);
CREATE INDEX IF NOT EXISTS metered_invoices_peer_state ON metered_invoices(peer,kind,state);
