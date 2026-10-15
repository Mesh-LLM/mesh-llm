// Words the page must never show, even to deny them. The copy gates below
// import them from here instead of spelling them out, so the source tree
// carries no copy of a banned word outside this assembly.
const join = (...parts: string[]) => parts.join('')

/** A standing assigned to a peer from its history. */
export const STANDING_WORD = join('repu', 'tation')
/** A second party's signature on someone else's record. */
export const SECOND_SIGNATURE_WORD = join('counter', 'sign')
/** Words that grade a peer: a figure computed about it, or an ordering. */
export const GRADING_WORDS = [join('sco', 're'), join('rat', 'ing'), join('ra', 'nk')]
