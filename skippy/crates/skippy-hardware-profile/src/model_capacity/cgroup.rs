//! Linux cgroup memory limits for the current process and its ancestors.
use std::path::{Path, PathBuf};

fn unescape_mount_path(path: &str) -> String {
    path.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
}

fn location(cgroups: &str, mounts: &str) -> Option<(PathBuf, PathBuf, bool)> {
    for line in cgroups.lines() {
        let mut parts = line.splitn(3, ':');
        let _ = parts.next()?;
        let controllers = parts.next()?;
        let group = PathBuf::from(parts.next()?);
        let v2 = controllers.is_empty();
        if !v2
            && !controllers
                .split(',')
                .any(|controller| controller == "memory")
        {
            continue;
        }
        for mount in mounts.lines() {
            let Some((before, after)) = mount.split_once(" - ") else {
                continue;
            };
            let fields: Vec<_> = before.split_whitespace().collect();
            let details: Vec<_> = after.split_whitespace().collect();
            if fields.len() < 5 || details.len() < 3 {
                continue;
            }
            if (v2 && details[0] != "cgroup2")
                || (!v2
                    && (details[0] != "cgroup" || !details[2].split(',').any(|c| c == "memory")))
            {
                continue;
            }
            let root = PathBuf::from(unescape_mount_path(fields[3]));
            let mountpoint = PathBuf::from(unescape_mount_path(fields[4]));
            if let Ok(relative) = group.strip_prefix(&root) {
                return Some((mountpoint.join(relative), mountpoint, v2));
            }
        }
    }
    None
}

fn read_number(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

pub(super) fn limits(host: (u64, u64)) -> Option<(u64, u64)> {
    let groups = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let mounts = std::fs::read_to_string("/proc/self/mountinfo").ok()?;
    let (current, mount, v2) = location(&groups, &mounts)?;
    limits_at(current, mount, v2, host)
}

fn limits_at(
    mut current: PathBuf,
    mount: PathBuf,
    v2: bool,
    host: (u64, u64),
) -> Option<(u64, u64)> {
    let (limit_file, usage_file) = if v2 {
        ("memory.max", "memory.current")
    } else {
        ("memory.limit_in_bytes", "memory.usage_in_bytes")
    };
    let mut result = host;
    let mut found = false;
    loop {
        if let (Some(limit), Some(used)) = (
            read_number(&current.join(limit_file)),
            read_number(&current.join(usage_file)),
        ) {
            result.0 = result.0.min(limit);
            result.1 = result.1.min(limit.saturating_sub(used));
            found = true;
        }
        if current == mount || !current.pop() {
            break;
        }
    }
    found.then_some(result)
}

#[cfg(test)]
mod tests {
    use super::{limits_at, location};
    use std::path::PathBuf;
    #[test]
    fn resolves_nested_v2_and_v1_mount_roots() {
        let mounts = "42 1 0:28 /tenant /sys/fs/cgroup rw - cgroup2 cgroup rw\n43 1 0:29 / /cg/memory rw - cgroup cgroup rw,memory\n";
        assert_eq!(
            location("0::/tenant/a/b\n", mounts),
            Some((
                PathBuf::from("/sys/fs/cgroup/a/b"),
                PathBuf::from("/sys/fs/cgroup"),
                true
            ))
        );
        assert_eq!(
            location("5:memory:/a/b\n", mounts),
            Some((
                PathBuf::from("/cg/memory/a/b"),
                PathBuf::from("/cg/memory"),
                false
            ))
        );
    }

    #[test]
    fn reads_the_most_restrictive_ancestor() {
        let root = std::env::temp_dir().join(format!("skippy-cgroup-{}", std::process::id()));
        let child = root.join("child");
        std::fs::create_dir_all(&child).unwrap();
        std::fs::write(root.join("memory.max"), "80").unwrap();
        std::fs::write(root.join("memory.current"), "70").unwrap();
        std::fs::write(child.join("memory.max"), "60").unwrap();
        std::fs::write(child.join("memory.current"), "20").unwrap();
        assert_eq!(
            limits_at(child, root.clone(), true, (100, 90)),
            Some((60, 10))
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn accounts_for_limits_above_host_ram_and_zero_limits() {
        let root = std::env::temp_dir().join(format!("skippy-cgroup-edge-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("memory.max"), "150").unwrap();
        std::fs::write(root.join("memory.current"), "145").unwrap();
        assert_eq!(
            limits_at(root.clone(), root.clone(), true, (100, 90)),
            Some((100, 5))
        );
        std::fs::write(root.join("memory.max"), "0").unwrap();
        assert_eq!(
            limits_at(root.clone(), root.clone(), true, (100, 90)),
            Some((0, 0))
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
