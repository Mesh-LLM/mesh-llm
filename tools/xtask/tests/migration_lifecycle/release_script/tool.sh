#!/bin/bash
set -euo pipefail
tool="${0##*/}"
printf '%s\0' "$tool" "$@" >> "$FIXTURE_ROOT/calls"
printf '\0' >> "$FIXTURE_ROOT/calls"
case "$tool" in
    git)
        case "$*" in
            'status --porcelain') exit 0 ;;
            'branch --show-current') printf 'main\n' ;;
            'fetch origin main') exit 0 ;;
            'rev-parse HEAD'|'rev-parse origin/main') printf '%s\n' "$FIXTURE_HEAD" ;;
            'rev-parse -q --verify refs/tags/'*) exit 1 ;;
            'ls-remote --exit-code --tags origin refs/tags/'*) exit 2 ;;
            'show origin/main:Cargo.toml') printf '[workspace.package]\nversion = "%s"\n' "$FIXTURE_VERSION" ;;
            *) echo 'unowned Git operation denied' >&2; exit 91 ;;
        esac ;;
    cargo)
        [[ "$#" == 5 && "$1" == xtool && "$2" == release && "$3" == version-at-least && "$4" == 0.1.0 && "$5" == "$FIXTURE_VERSION" ]] || exit 92
        exit 0 ;;
    date)
        [[ "$#" == 2 && "$1" == -u && "$2" == '+%Y-%m-%dT%H:%M:%SZ' ]] || exit 93
        printf '%s\n' "$FIXTURE_START" ;;
    sleep)
        [[ "$#" == 1 && ( "$1" == 2 || "$1" == 5 ) ]] || exit 94
        exit 0 ;;
    gh)
        case "$1 $2" in
            'auth status'|'workflow view') exit 0 ;;
            'workflow run')
                [[ "$FAIL_AT" != dispatch ]] || exit 17
                if [[ "$FIXTURE_ROUTE" == url ]]; then printf 'https://github.com/fixture/repo/actions/runs/101\n'
                else printf 'workflow dispatched without a URL\n'; fi ;;
            'release view')
                [[ -f "$FIXTURE_ROOT/watch.done" ]] || exit 1 ;;
            'run watch')
                [[ "$#" == 5 && "$3" =~ ^[0-9]+$ && "$4" == --compact && "$5" == --exit-status ]] || exit 95
                [[ "$FAIL_AT" != watch ]] || exit 23
                printf 'done' > "$FIXTURE_ROOT/watch.done" ;;
            'run list')
                [[ "$FAIL_AT" != lookup ]] || exit 17
                shift 2
                query=''
                while (( $# )); do
                    case "$1" in
                        --workflow|--branch|--commit|--event|--created|--json) shift 2 ;;
                        --jq) query="$2"; shift 2 ;;
                        *) echo 'unowned gh query argument' >&2; exit 96 ;;
                    esac
                done
                [[ -n "$query" ]] || exit 97
                jq -r "$query" "$FIXTURE_ROOT/runs.json" ;;
            *) echo 'unowned GitHub operation denied' >&2; exit 98 ;;
        esac ;;
    *) echo 'native/network/build command denied in finite fixture' >&2; exit 99 ;;
esac
