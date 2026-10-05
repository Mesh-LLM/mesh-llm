const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i

class LogsIdentifierError extends Error {
  constructor(kind: string) {
    super(`${kind} is invalid`)
    this.name = 'LogsIdentifierError'
  }
}

function hasControlCharacter(value: string): boolean {
  return [...value].some((character) => {
    const codePoint = character.charCodeAt(0)
    return codePoint <= 31 || codePoint === 127
  })
}

function tryParseUuid(value: string) {
  return UUID_PATTERN.test(value) ? value.toLowerCase() : undefined
}

function parseUuid(value: string, kind: string) {
  const parsed = tryParseUuid(value)
  if (!parsed) throw new LogsIdentifierError(kind)
  return parsed
}

/** A request identifier that has crossed the logs API boundary. */
export class LogRequestId {
  readonly #value: string

  private constructor(value: string) {
    this.#value = value
  }

  static parse(value: string) {
    return new LogRequestId(parseUuid(value, 'request ID'))
  }

  static tryParse(value: string) {
    const parsed = tryParseUuid(value)
    return parsed ? new LogRequestId(parsed) : undefined
  }

  toString() {
    return this.#value
  }

  toJSON() {
    return this.#value
  }
}

/** An event identifier that has crossed the logs API boundary. */
export class LogEventId {
  readonly #value: string

  private constructor(value: string) {
    this.#value = value
  }

  static parse(value: string) {
    return new LogEventId(parseUuid(value, 'event ID'))
  }

  toString() {
    return this.#value
  }

  toJSON() {
    return this.#value
  }
}

/** An artifact identifier that is safe to interpolate into a logs API path. */
export class LogArtifactId {
  readonly #value: string

  private constructor(value: string) {
    this.#value = value
  }

  static parse(value: string) {
    return new LogArtifactId(parseUuid(value, 'artifact ID'))
  }

  toString() {
    return this.#value
  }

  toJSON() {
    return this.#value
  }
}

/** An idempotency key for a previewed logs maintenance operation. */
export class LogOperationId {
  readonly #value: string

  private constructor(value: string) {
    this.#value = value
  }

  static parse(value: string) {
    return new LogOperationId(parseUuid(value, 'operation ID'))
  }

  static create() {
    return LogOperationId.parse(crypto.randomUUID())
  }

  toString() {
    return this.#value
  }

  toJSON() {
    return this.#value
  }
}

/** An audit-entry identifier returned by a completed logs maintenance operation. */
export class LogAuditId {
  readonly #value: string

  private constructor(value: string) {
    this.#value = value
  }

  static parse(value: string) {
    return new LogAuditId(parseUuid(value, 'audit ID'))
  }

  toString() {
    return this.#value
  }

  toJSON() {
    return this.#value
  }
}

/** An opaque REST page cursor. Its internals deliberately remain client-opaque. */
export class LogPageCursor {
  readonly #value: string

  private constructor(value: string) {
    this.#value = value
  }

  static parse(value: string) {
    if (value.length === 0 || hasControlCharacter(value)) {
      throw new LogsIdentifierError('page cursor')
    }
    return new LogPageCursor(value)
  }

  toString() {
    return this.#value
  }
}

/** A versioned cursor used by the dedicated logs SSE protocol. */
export class LogReplayCursor {
  readonly #requests: bigint
  readonly #operations: bigint
  readonly #system: bigint

  private constructor(requests: bigint, operations: bigint, system: bigint) {
    this.#requests = requests
    this.#operations = operations
    this.#system = system
  }

  static parse(value: string) {
    const match = /^v1:(\d+)\.(\d+)\.(\d+)$/.exec(value)
    if (!match) throw new LogsIdentifierError('replay cursor')

    try {
      return new LogReplayCursor(BigInt(match[1]), BigInt(match[2]), BigInt(match[3]))
    } catch {
      throw new LogsIdentifierError('replay cursor')
    }
  }

  sequence(channel: LogReplayChannel) {
    switch (channel) {
      case 'requests':
        return this.#requests
      case 'operations':
        return this.#operations
      case 'system':
        return this.#system
    }
  }

  toString() {
    return `v1:${this.#requests}.${this.#operations}.${this.#system}`
  }
}

export class LogAuditCursor {
  readonly #sequence: bigint

  private constructor(sequence: bigint) {
    this.#sequence = sequence
  }

  static parse(value: string) {
    const match = /^a1:(\d+)$/.exec(value)
    if (!match) throw new LogsIdentifierError('audit cursor')

    try {
      return new LogAuditCursor(BigInt(match[1]))
    } catch {
      throw new LogsIdentifierError('audit cursor')
    }
  }

  sequence() {
    return this.#sequence
  }

  toString() {
    return `a1:${this.#sequence}`
  }
}

export type LogReplayChannel = 'requests' | 'operations' | 'system'
export type LogAuditChannel = 'audit'
