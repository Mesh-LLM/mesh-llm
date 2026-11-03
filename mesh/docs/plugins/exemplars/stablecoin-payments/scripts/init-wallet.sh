#!/usr/bin/env bash
# Explicit operator-only key creation. Never invoked by build, tests or packaging.
set -euo pipefail
umask 077
if [[ $# != 2 || $1 != /* || $2 != /* ]]; then
  echo 'Usage: init-wallet.sh /absolute/new-private-directory /absolute/wallet-ousd-tempo' >&2
  exit 2
fi
profile_root=$1
wallet_binary=$2
if [[ -e "$profile_root" ]]; then
  echo 'Refusing an existing directory; use the existing wallet instead of replacing it.' >&2
  exit 1
fi
mkdir -m 700 -- "$profile_root"
openssl rand -base64 48 > "$profile_root/password"
chmod 600 "$profile_root/password"
exec "$wallet_binary" --state-dir "$profile_root/wallet" --password-file "$profile_root/password" init
