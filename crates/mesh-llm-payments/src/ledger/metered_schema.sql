CREATE TABLE IF NOT EXISTS metered_requests(
    id TEXT PRIMARY KEY REFERENCES serving_requests(id), peer TEXT NOT NULL,
    pricing TEXT NOT NULL, estimated_input_tokens INTEGER NOT NULL,
    max_output_tokens INTEGER NOT NULL, require_deposit INTEGER NOT NULL,
    dispatched INTEGER NOT NULL DEFAULT 0, reserved_msat INTEGER NOT NULL, deposit_msat INTEGER NOT NULL,
    state TEXT NOT NULL, outcome TEXT, bill_msat INTEGER,
    deposit_paid_msat INTEGER NOT NULL DEFAULT 0, credit_applied_msat INTEGER NOT NULL DEFAULT 0,
    credit_created_msat INTEGER NOT NULL DEFAULT 0, remaining_msat INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS metered_requests_peer ON metered_requests(peer,state);
CREATE TABLE IF NOT EXISTS peer_credit(peer TEXT PRIMARY KEY, msat INTEGER NOT NULL CHECK(msat>=0));
