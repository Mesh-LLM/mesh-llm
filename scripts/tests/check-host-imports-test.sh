#!/usr/bin/env bash
# scripts/check-host-imports.sh refuses backend libraries and nothing else.
set -euo pipefail
check="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/check-host-imports.sh"
fail=0

expect() { # <accepted|refused> <import name>
    local got=accepted
    printf '%s\n' "$2" | "$check" --names >/dev/null 2>&1 || got=refused
    [[ "$got" == "$1" ]] || { echo "FAIL: '$2' was $got, want $1" >&2; fail=1; }
}

for name in /usr/lib/libcuda.so.1 libcuda 'C:\Windows\NVCUDA.DLL' cudart64_12.dll \
    cublasLt64_12.dll cublas64_13.dll libcudart.so.12 libcublas.so.12 libnccl.so.2 \
    libamdhip64.so.6 libhipblas.so libhsa-runtime64.so.1 amdhip64.dll rocblas.dll \
    libvulkan.so.1 vulkan-1.dll @rpath/libggml-base.dylib ggml.dll libllama.so llama.dll \
    /System/Library/Frameworks/Metal.framework/Versions/A/Metal; do
    expect refused "$name"
done

for name in libcudax.so cudart64_.dll 'cudart64_1/2.dll' xllama.so /usr/lib/libSystem.B.dylib \
    libc.so.6 libssl.so.3 KERNEL32.dll \
    /System/Library/Frameworks/Security.framework/Versions/A/Security; do
    expect accepted "$name"
done

# One backend library among ordinary ones is enough.
printf 'libc.so.6\nlibcudart.so.12\n' | "$check" --names >/dev/null 2>&1 &&
    { echo "FAIL: a list holding libcudart.so.12 was accepted" >&2; fail=1; }
# An empty listing is a failed listing.
printf '' | "$check" --names >/dev/null 2>&1 &&
    { echo "FAIL: an empty import list was accepted" >&2; fail=1; }

# A grep that fails must not read as "no backend library found".
stub="$(mktemp -d)"
trap 'rm -rf "$stub"' EXIT
printf '#!/bin/sh\nexit 2\n' >"$stub/grep"
chmod +x "$stub/grep"
out="$(printf 'libc.so.6\n' | PATH="$stub:$PATH" "$check" --names 2>&1)" &&
    { echo "FAIL: a failing grep was accepted" >&2; fail=1; }
[[ "$out" == *"grep failed with status 2"* ]] ||
    { echo "FAIL: a failing grep was not reported: $out" >&2; fail=1; }

# A passing host gets a report that names its SHA-256. The copy has its own
# floor file so the test does not depend on this machine's glibc.
cp "$check" "$stub/check-host-imports.sh"
echo 999 >"$stub/linux-glibc-floor.txt"
cp "$(command -v bash)" "$stub/host"
sha="$( (sha256sum "$stub/host" 2>/dev/null || shasum -a 256 "$stub/host") | awk '{print $1}')"
"$stub/check-host-imports.sh" "$stub/host" --report "$stub/report.json" >/dev/null ||
    { echo "FAIL: this machine's bash was refused" >&2; fail=1; }
for want in '"binary": "host"' "\"binary_sha256\": \"$sha\"" '"policy": "mesh-llm-dynamic-host-v2"' \
    '"rejected_imports": []' '"imports": [' '"glibc_floor": ' '"format": "'; do
    grep -qF -- "$want" "$stub/report.json" || { echo "FAIL: the report lacks $want" >&2; fail=1; }
done
# A failed check removes the report that was there and writes none.
PATH="$stub:$PATH" "$stub/check-host-imports.sh" "$stub/host" --report "$stub/report.json" >/dev/null 2>&1 &&
    { echo "FAIL: a failing grep was accepted for a host" >&2; fail=1; }
[[ -z "$(find "$stub" -name 'report.json*')" ]] ||
    { echo "FAIL: a failed check left a report behind" >&2; fail=1; }

# A tool that prints plausible output and then fails must fail the check, and
# take a report that was already there with it. The stubs stand in for Linux.
linux="$stub/linux"
mkdir "$linux"
printf '#!/bin/sh\necho Linux\n' >"$linux/uname"
refuses_when() { # <what fails> <objdump -p status> <objdump -T status>
    cat >"$linux/objdump" <<STUB
#!/bin/sh
if [ "\$1" = -p ]; then echo "  NEEDED               libc.so.6"; exit $2; fi
echo "0000 DF *UND* 0000 (GLIBC_2.17) malloc"; exit $3
STUB
    chmod +x "$linux/uname" "$linux/objdump"
    echo '{}' >"$stub/report.json"
    if PATH="$linux:$PATH" "$stub/check-host-imports.sh" "$stub/host" --report "$stub/report.json" >/dev/null 2>&1; then
        [[ "$1" == nothing ]] || { echo "FAIL: accepted a host when $1 failed" >&2; fail=1; }
    else
        [[ "$1" != nothing ]] || { echo "FAIL: refused the stub Linux host" >&2; fail=1; }
        [[ -z "$(find "$stub" -name 'report.json*')" ]] ||
            { echo "FAIL: a report survived when $1 failed" >&2; fail=1; }
    fi
}
refuses_when nothing 0 0
grep -qF '"glibc_floor": "2.17"' "$stub/report.json" || { echo "FAIL: the stub Linux report lacks its glibc" >&2; fail=1; }
refuses_when "objdump -T" 0 42
refuses_when "objdump -p" 42 0
echo 2.16 >"$stub/linux-glibc-floor.txt"
refuses_when "the glibc floor" 0 0
: >"$stub/linux-glibc-floor.txt"
refuses_when "reading the floor file" 0 0
printf '2.39\n2.17\n' >"$stub/linux-glibc-floor.txt"
refuses_when "a floor file with two versions" 0 0
printf '2.39 2.17\n' >"$stub/linux-glibc-floor.txt"
refuses_when "a floor line with two versions" 0 0
printf '# why\n\n  2.39  \n' >"$stub/linux-glibc-floor.txt"
refuses_when nothing 0 0

((fail == 0)) && echo "check-host-imports-test: ok"
exit "$fail"
