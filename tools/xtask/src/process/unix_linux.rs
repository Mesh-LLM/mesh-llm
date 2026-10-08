use super::Failure;
use std::fs::{self, File};
use std::io::{self, Read};

pub(super) fn group_active(group: libc::pid_t) -> Result<bool, Failure> {
    let entries = fs::read_dir("/proc").map_err(|error| Failure::io("list processes", error))?;
    for (count, entry) in entries.enumerate() {
        if count >= 131072 {
            return Err(Failure::EnumerationLimit);
        }
        let entry = entry.map_err(|error| Failure::io("list process", error))?;
        if !entry
            .file_name()
            .as_encoded_bytes()
            .iter()
            .all(u8::is_ascii_digit)
        {
            continue;
        }
        let mut stat = Vec::new();
        let result = File::open(entry.path().join("stat"))
            .and_then(|file| file.take(4096).read_to_end(&mut stat));
        match result {
            Ok(_) => (),
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(Failure::io("inspect process", error)),
        }
        let end = stat
            .windows(2)
            .rposition(|bytes| bytes == b") ")
            .ok_or(Failure::EnumerationLimit)?;
        let fields =
            std::str::from_utf8(&stat[end + 2..]).map_err(|_| Failure::EnumerationLimit)?;
        let mut fields = fields.split_whitespace();
        let state = fields.next().ok_or(Failure::EnumerationLimit)?;
        let process_group = fields
            .nth(1)
            .ok_or(Failure::EnumerationLimit)?
            .parse::<i32>()
            .map_err(|_| Failure::EnumerationLimit)?;
        if process_group == group && state != "Z" && state != "X" {
            return Ok(true);
        }
    }
    Ok(false)
}
