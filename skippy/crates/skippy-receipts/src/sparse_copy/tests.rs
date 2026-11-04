use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::ThreadId;

use super::*;

/// A remote file in memory that counts the requests and bytes it serves,
/// and notes the thread of each request.
struct Remote {
    bytes: Vec<u8>,
    requests: Arc<AtomicUsize>,
    served: Arc<AtomicUsize>,
    threads: Arc<Mutex<Vec<ThreadId>>>,
}

impl RangeFetch for Remote {
    fn fetch(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        self.requests.fetch_add(1, Ordering::SeqCst);
        self.served.fetch_add(len as usize, Ordering::SeqCst);
        self.threads
            .lock()
            .unwrap()
            .push(std::thread::current().id());
        let start = offset as usize;
        Ok(self.bytes[start..start + len as usize].to_vec())
    }
}

/// A remote whose ranges at or past `from` fail.
struct FailsFrom {
    remote: Remote,
    from: u64,
}

impl RangeFetch for FailsFrom {
    fn fetch(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        if offset >= self.from {
            bail!("the range at {offset} failed");
        }
        self.remote.fetch(offset, len)
    }
}

struct Fixture {
    _dir: tempfile::TempDir,
    path: PathBuf,
    bytes: Vec<u8>,
    requests: Arc<AtomicUsize>,
    served: Arc<AtomicUsize>,
    threads: Arc<Mutex<Vec<ThreadId>>>,
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
            threads: Arc::default(),
        }
    }

    fn identity(&self, source: &str) -> RemoteIdentity {
        RemoteIdentity {
            source: source.into(),
            size: self.bytes.len() as u64,
        }
    }

    fn remote(&self) -> Remote {
        Remote {
            bytes: self.bytes.clone(),
            requests: self.requests.clone(),
            served: self.served.clone(),
            threads: self.threads.clone(),
        }
    }

    fn open(&self, source: &str) -> SparseCopy {
        SparseCopy::open(&self.path, self.identity(source), Box::new(self.remote())).unwrap()
    }

    fn record(&self) -> Record {
        serde_json::from_slice(&std::fs::read(sidecar(&self.path, "ranges")).unwrap()).unwrap()
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

#[test]
fn an_empty_range_is_read_without_a_request() {
    let f = Fixture::new(4096);
    let copy = f.open("repo@abc/model.gguf");
    copy.read_at(100, 50).unwrap();
    // an empty range inside a held one used to ask the map for a range
    // that ends before it starts
    assert_eq!(copy.read_once(120, 0).unwrap(), Vec::<u8>::new());
    assert_eq!(copy.read_at(120, 0).unwrap(), Vec::<u8>::new());
    assert_eq!(copy.missing_bytes(&[(120, 0), (4096, 0)]).unwrap(), 0);
    assert_eq!(f.requests(), 1);
    let mut held = BTreeMap::new();
    insert(&mut held, 10, 20);
    assert_eq!(lacking(&held, 15, 15), vec![]);
    assert_eq!(lacking(&held, 15, 14), vec![]);
}

#[test]
fn offsets_that_wrap_are_past_the_end() {
    let f = Fixture::new(4096);
    let copy = f.open("repo@abc/model.gguf");
    for (offset, len) in [(u64::MAX, 2), (4000, u64::MAX), (4097, 0), (0, 4097)] {
        assert!(copy.read_at(offset, len).is_err(), "{offset}+{len}");
        assert!(copy.read_once(offset, len).is_err(), "{offset}+{len}");
        assert!(copy.prefetch(&[(offset, len)]).is_err(), "{offset}+{len}");
        assert!(copy.missing_bytes(&[(offset, len)]).is_err());
    }
    assert_eq!(f.requests(), 0);
}

#[test]
fn missing_bytes_count_each_byte_once() {
    let f = Fixture::new(4096);
    let copy = f.open("repo@abc/model.gguf");
    copy.read_at(100, 100).unwrap();
    // 50 are held, and the two ranges share 25
    let ranges = [(150, 100), (225, 100)];
    assert_eq!(copy.missing_bytes(&ranges).unwrap(), 125);
    copy.prefetch(&ranges).unwrap();
    assert_eq!(copy.missing_bytes(&ranges).unwrap(), 0);
    assert_eq!(f.served(), 225);
}

#[test]
fn an_opener_never_truncates_a_copy_another_has_open() {
    let f = Fixture::new(4096);
    let first = f.open("repo@abc/model.gguf");
    first.read_at(1000, 100).unwrap();
    // no record, which once had the next opener empty the file under the
    // first, whose ranges then read as zeros
    std::fs::remove_file(sidecar(&f.path, "ranges")).unwrap();
    let same = f.identity("repo@abc/model.gguf");
    assert!(SparseCopy::open(&f.path, same, Box::new(f.remote())).is_err());
    // nor does an opener of another file take the copy over
    let other = f.identity("repo@def/model.gguf");
    let refused = SparseCopy::open(&f.path, other, Box::new(f.remote()))
        .err()
        .unwrap()
        .to_string();
    assert!(refused.contains("open for another file"), "{refused}");
    assert_eq!(first.read_at(1000, 100).unwrap(), &f.bytes[1000..1100]);
    assert_eq!(f.served(), 100);
    drop(first);
    // with no opener left, the copy starts over
    let again = f.open("repo@abc/model.gguf");
    assert_eq!(again.held_bytes(), 0);
    assert_eq!(again.read_at(1000, 100).unwrap(), &f.bytes[1000..1100]);
}

#[test]
fn a_record_without_its_data_file_is_discarded() {
    let f = Fixture::new(4096);
    let copy = f.open("repo@abc/model.gguf");
    copy.read_at(0, 100).unwrap();
    drop(copy);
    std::fs::remove_file(&f.path).unwrap();
    let copy = f.open("repo@abc/model.gguf");
    assert_eq!(copy.held_bytes(), 0);
    assert!(f.record().ranges.is_empty());
    assert_eq!(copy.read_at(0, 100).unwrap(), &f.bytes[..100]);
    assert_eq!(f.served(), 200);
}

#[test]
fn a_record_that_does_not_read_starts_the_copy_over() {
    let f = Fixture::new(4096);
    let identity = serde_json::to_value(f.identity("repo@abc/model.gguf")).unwrap();
    for ranges in [
        serde_json::json!([[0, 5000]]),
        serde_json::json!([[10, 10]]),
        serde_json::json!([[20, 10]]),
        serde_json::json!("all of it"),
    ] {
        f.open("repo@abc/model.gguf").read_at(0, 10).unwrap();
        let record = serde_json::json!({"identity": identity, "ranges": ranges});
        std::fs::write(sidecar(&f.path, "ranges"), record.to_string()).unwrap();
        let copy = f.open("repo@abc/model.gguf");
        assert_eq!(copy.held_bytes(), 0, "{ranges}");
    }
    std::fs::write(sidecar(&f.path, "ranges"), b"not json").unwrap();
    assert_eq!(f.open("repo@abc/model.gguf").held_bytes(), 0);
    // a copy shorter than the file is not a copy of it
    f.open("repo@abc/model.gguf").read_at(0, 10).unwrap();
    std::fs::write(&f.path, b"short").unwrap();
    let copy = f.open("repo@abc/model.gguf");
    assert_eq!(copy.held_bytes(), 0);
    assert_eq!(copy.read_at(0, 10).unwrap(), &f.bytes[..10]);
}

#[test]
fn a_copy_has_a_record_from_the_start() {
    let f = Fixture::new(4096);
    let copy = f.open("repo@abc/model.gguf");
    let record = f.record();
    assert_eq!(record.identity, f.identity("repo@abc/model.gguf"));
    assert!(record.ranges.is_empty());
    copy.read_at(8, 8).unwrap();
    assert_eq!(f.record().ranges, vec![(8, 16)]);
}

#[test]
fn openers_of_one_copy_keep_each_others_ranges() {
    let f = Fixture::new(1 << 20);
    let first = f.open("repo@abc/model.gguf");
    let second = f.open("repo@abc/model.gguf");
    first.read_at(0, 100).unwrap();
    second.read_at(500_000, 100).unwrap();
    // the first learns of the second's range when it next records one
    first.read_at(200, 100).unwrap();
    assert_eq!(first.held_bytes(), 300);
    assert_eq!(
        first.read_at(500_000, 100).unwrap(),
        &f.bytes[500_000..500_100]
    );
    drop((first, second));
    let copy = f.open("repo@abc/model.gguf");
    assert_eq!(copy.held_bytes(), 300);
    assert_eq!(f.served(), 300);
}

#[test]
fn two_copies_in_one_process_record_at_once() {
    let f = Fixture::new(1 << 20);
    let copies = [f.open("repo@abc/model.gguf"), f.open("repo@abc/model.gguf")];
    std::thread::scope(|scope| {
        for (n, copy) in copies.iter().enumerate() {
            scope.spawn(move || {
                for i in 0..50u64 {
                    // far apart, so every read records a range of its own
                    copy.read_at((2 * i + n as u64) * 2048, 8).unwrap();
                }
            });
        }
    });
    drop(copies);
    assert_eq!(f.served(), 100 * 8);
    let left: Vec<_> = std::fs::read_dir(f.path.parent().unwrap())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.ends_with(".tmp"))
        .collect();
    assert!(left.is_empty(), "{left:?}");
}

#[test]
fn one_range_is_fetched_on_the_calling_thread() {
    let f = Fixture::new(64 << 20);
    let copy = f.open("repo@abc/model.gguf");
    copy.read_at(0, 100).unwrap();
    let here = std::thread::current().id();
    assert_eq!(*f.threads.lock().unwrap(), vec![here]);
    f.threads.lock().unwrap().clear();
    copy.prefetch(&[(1 << 20, 10), (40 << 20, 10)]).unwrap();
    let threads = f.threads.lock().unwrap().clone();
    assert_eq!(threads.len(), 2);
    assert!(!threads.contains(&here));
}

#[test]
fn a_failed_range_keeps_the_ranges_that_fetched() {
    let f = Fixture::new(64 << 20);
    let remote = FailsFrom {
        remote: f.remote(),
        from: 30 << 20,
    };
    let identity = f.identity("repo@abc/model.gguf");
    let copy = SparseCopy::open(&f.path, identity, Box::new(remote)).unwrap();
    assert!(copy.prefetch(&[(0, 10), (40 << 20, 10)]).is_err());
    assert_eq!(copy.held_bytes(), 10);
    drop(copy);
    assert_eq!(f.open("repo@abc/model.gguf").held_bytes(), 10);
}
