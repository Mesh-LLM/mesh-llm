// Entry-row id display (design §3A:
// "ids: `exch 8225…94ef · rec 28e6…3f6a`, click-to-copy, full string in the
// expansion only"). A real sealed capsule id / exchange key is 64 lower-hex
// (`capsuleIdIsDigestShaped`'s DIGEST_LEN) -- far too long to sit in a
// one-line summary. Short, human-scale identifiers (harness fixtures, a
// self-minted `capsule-chatcmpl-…` marker) are left as-is: truncating an
// already-short string to the same first4…last4 shape buys nothing and
// would make two distinct short ids collide on screen.
const SHORTEN_ABOVE_LENGTH = 16

export function shortId(id: string): string {
  if (id.length <= SHORTEN_ABOVE_LENGTH) return id
  return `${id.slice(0, 4)}…${id.slice(-4)}`
}
