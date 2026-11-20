#!/usr/bin/env bash
# Rewrite the UniFFI API checksum guards in the generated Swift bindings with
# the constants the built static library actually returns.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 LIBRARY GENERATED_SWIFT" >&2
  exit 2
fi
library="$1"
swift="$2"
[[ -f "$library" ]] || { echo "static library not found: $library" >&2; exit 1; }
[[ -f "$swift" ]] || { echo "generated Swift not found: $swift" >&2; exit 1; }

values="$(mktemp)"
trap 'rm -f "$values" "$swift.tmp"' EXIT

# Each checksum symbol is a two-instruction function: load an immediate into
# w0/ax/eax, then return. Emit "name value" for each.
LC_ALL=C /usr/bin/otool -tvV "$library" | awk '
  function fail(msg) { print msg > "/dev/stderr"; bad = 1; exit 1 }
  state == 1 {
    line = $0; sub(/^[[:space:]]*[0-9a-fA-F]+[[:space:]]+/, "", line); gsub(/[[:space:]]/, "", line)
    if (line ~ /^movw0,#/) { v = substr(line, 8) }
    else if (line ~ /^movw\$.*,%ax$/ || line ~ /^movl\$.*,%eax$/) { v = line; sub(/^mov[wl]\$/, "", v); sub(/,%e?ax$/, "", v) }
    else fail("checksum symbol " name " must return a constant")
    if (v !~ /^(0x[0-9a-fA-F]+|[0-9]+)$/) fail("invalid checksum constant for " name)
    state = 2; next
  }
  state == 2 {
    line = $0; sub(/^[[:space:]]*[0-9a-fA-F]+[[:space:]]+/, "", line); sub(/[[:space:]]+$/, "", line)
    if (line != "ret" && line != "retq") fail("checksum constant must be followed by return in " name)
    if (name in seen) fail("ambiguous duplicate checksum symbol " name)
    seen[name] = 1; print name, v; state = 0; next
  }
  /^_uniffi_meshllm_ffi_checksum_[A-Za-z0-9_]+:$/ { name = substr($0, 2, length($0) - 2); state = 1 }
  END { if (!bad && state != 0) fail("truncated checksum disassembly") }
' > "$values"

[[ -s "$values" ]] || { echo "no UniFFI checksum symbols in $library" >&2; exit 1; }

VALUES="$values" perl -e '
  open my $fh, "<", $ENV{VALUES} or die; my %v;
  for (<$fh>) { chomp; my ($n, $x) = split / /; $x = hex($x) if $x =~ /^0x/; die "checksum out of range for $n\n" if $x > 65535; $v{$n} = $x }
  my ($count, %seen) = (0);
  while (my $line = <STDIN>) {
    if ($line =~ /(uniffi_meshllm_ffi_checksum_\w+)\(\) != (\d+)\)/) {
      my $name = $1;
      die "native checksum missing for $name\n" unless exists $v{$name};
      die "duplicate generated checksum guard: $name\n" if $seen{$name}++;
      $line =~ s/\Q$name\E\(\) != \d+\)/$name() != $v{$name})/;
      $count++;
    } elsif ($line =~ /uniffi_meshllm_ffi_checksum_/ && $line =~ /!=/) {
      die "malformed generated checksum guard: $line";
    }
    print $line;
  }
  die "Swift bindings contain no UniFFI API checksum guards\n" unless $count;
  print STDERR "Synchronized $count UniFFI API checksum guards\n";
' < "$swift" > "$swift.tmp"
mv "$swift.tmp" "$swift"
trap 'rm -f "$values"' EXIT
