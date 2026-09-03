//! Reducer acceptance tests, split by class. Every test is synchronous and
//! sleep-free; the one concurrency test uses real threads plus a `Barrier`.

mod fixtures;
mod lock_separation;
mod ordering;
mod rebuild;
mod terminal;
