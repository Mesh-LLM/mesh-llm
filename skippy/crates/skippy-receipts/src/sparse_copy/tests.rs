use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;

/// A remote file in memory that counts the requests and bytes it serves.
struct Remote {
    bytes: Vec<u8>,
    requests: Arc<AtomicUsize>,
    served: Arc<AtomicUsize>,
}

impl RangeFetch for Remote {
    fn fetch(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        self.requests.fetch_add(1, Ordering::SeqCst);
        self.served.fetch_add(len as usize, Ordering::SeqCst);
        let start = offset as usize;
        Ok(self.bytes[start..start + len as usize].to_vec())
    }
}

struct Fixture {
    _dir: tempfile::TempDir,
    path: PathBuf,
    bytes: Vec<u8>,
    requests: Arc<AtomicUsize>,
    served: Arc<AtomicUsize>,
}

impl Fixture {
    fn new(size: usize) -> Self {
        let dir = tempfile::tempdir().unwrap();
        Self {
            path: dir.path().join("model.gguf"),
            _dir: dir,
            bytes: (0..size).map(|i| (i % 251) as u8).collect(),
            requests: Arc::default(),
            served: Arc::default(),
        }
    }

    fn identity(&self, source: &str) -> RemoteIdentity {
        RemoteIdentity {
            source: source.into(),
            size: self.bytes.len() as u64,
        }
    }

    fn open(&self, source: &str) -> SparseCopy {
        let remote = Remote {
            bytes: self.bytes.clone(),
            requests: self.requests.clone(),
            served: self.served.clone(),
        };
        SparseCopy::open(&self.path, self.identity(source), Box::new(remote)).unwrap()
    }

    fn requests(&self) -> usize {
        self.requests.load(Ordering::SeqCst)
    }

    fn served(&self) -> usize {
        self.served.load(Ordering::SeqCst)
    }
}

#[test]
fn a_copy_fetches_each_byte_once_and_keeps_it() {
    let f = Fixture::new(1 << 20);
    let copy = f.open("repo@abc/model.gguf");
    assert_eq!(copy.read_at(1000, 10).unwrap(), &f.bytes[1000..1010]);
    assert_eq!(copy.read_at(1005, 10).unwrap(), &f.bytes[1005..1015]);
    assert_eq!(f.served(), 15);
    drop(copy);
    // a new process reads what the record says the copy holds
    let copy = f.open("repo@abc/model.gguf");
    assert_eq!(copy.read_at(1000, 15).unwrap(), &f.bytes[1000..1015]);
    assert_eq!(f.served(), 15);
    assert_eq!(copy.held_bytes(), 15);
}

#[test]
fn a_copy_of_another_file_starts_over() {
    let f = Fixture::new(4096);
    let copy = f.open("repo@abc/model.gguf");
    copy.read_at(0, 100).unwrap();
    drop(copy);
    let copy = f.open("repo@def/model.gguf");
    assert_eq!(copy.held_bytes(), 0);
    copy.read_at(0, 100).unwrap();
    assert_eq!(f.served(), 200);
}

#[test]
fn near_ranges_fetch_as_one_and_far_ones_at_once() {
    let f = Fixture::new(64 << 20);
    let copy = f.open("repo@abc/model.gguf");
    let ranges = [(0, 10), (100, 10), (40 << 20, 10), (50 << 20, 10)];
    copy.prefetch(&ranges).unwrap();
    assert_eq!(f.requests(), 3);
    for (offset, len) in ranges {
        let (start, end) = (offset as usize, (offset + len) as usize);
        assert_eq!(copy.read_at(offset, len).unwrap(), &f.bytes[start..end]);
    }
    assert_eq!(f.requests(), 3);
}

#[test]
fn a_large_range_fetches_in_parts() {
    let f = Fixture::new(20 << 20);
    let copy = f.open("repo@abc/model.gguf");
    copy.prefetch(&[(0, 20 << 20)]).unwrap();
    assert_eq!(f.requests(), 3);
    assert_eq!(copy.read_at(0, 20 << 20).unwrap(), f.bytes);
}

#[test]
fn a_range_past_the_end_is_refused() {
    let f = Fixture::new(4096);
    let copy = f.open("repo@abc/model.gguf");
    assert!(copy.read_at(4000, 200).is_err());
    assert_eq!(f.requests(), 0);
}

#[test]
fn the_held_ranges_join() {
    let mut held = BTreeMap::new();
    insert(&mut held, 10, 20);
    insert(&mut held, 30, 40);
    assert_eq!(lacking(&held, 0, 50), vec![(0, 10), (20, 30), (40, 50)]);
    assert_eq!(lacking(&held, 12, 18), vec![]);
    assert_eq!(lacking(&held, 15, 35), vec![(20, 30)]);
    insert(&mut held, 20, 30);
    assert_eq!(held.into_iter().collect::<Vec<_>>(), vec![(10, 40)]);
}
