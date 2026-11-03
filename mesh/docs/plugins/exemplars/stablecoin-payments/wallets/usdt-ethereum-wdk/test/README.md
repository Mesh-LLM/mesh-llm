# Offline Ethereum WDK tests

Run from this worker directory with Node 22.13 or newer:

```sh
npm test
```

Tests use published deterministic seeds, temporary encrypted profiles and strict
JSON-RPC servers bound to `127.0.0.1`. Never fund fixture addresses. They make no
mainnet calls and create no wallets outside their temporary directories. An
unexpected fake RPC method fails the test rather than returning permissive data.

Actual pinned WDK code derives and signs mainnet type2 Ethereum transactions;
ethers independently decodes their signatures and hashes. The suite covers exact
100-byte USDT transfer calldata with an obligation memo, empty successful USDT
return data, ETH fees at 18 decimals versus USDT amounts at six decimals, storage
permissions, canonical payment IDs, finalized receipt evidence, and class/worker
process restart with unchanged raw bytes and nonce. Missing receipts and expired
local authorization metadata never establish native Ethereum expiry.

The test-side fsynced file exercises the prepare-before-broadcast interface.
The Rust wrapper owns the production journal, nonce reservation and one-time
receipt consumption and requires its own tests. This suite neither replaces
those checks nor establishes live Ethereum settlement.
