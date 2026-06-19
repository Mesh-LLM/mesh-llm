import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { HighlightedTomlLines } from '@/features/configuration/components/toml-highlight'

describe('HighlightedTomlLines', () => {
  it('highlights comments, tables, dotted keys, quoted keys, and scalar values', () => {
    const { container } = render(
      <HighlightedTomlLines
        toml={[
          '# Mesh LLM generated config preview',
          '[runtime]',
          'debug = true',
          '[defaults.request_defaults]',
          'temperature = 0.55',
          'top_p = 1',
          'headers = { }',
          '"nested.key" = "literal"',
          'plugin.enabled = false',
          '[[models]]',
          'model = "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL"'
        ].join('\n')}
      />
    )

    const sections = Array.from(container.querySelectorAll('[data-toml-token="section"]')).map(
      (node) => node.textContent
    )
    expect(sections).toContain('[runtime]')
    expect(sections).toContain('[defaults.request_defaults]')
    expect(sections).toContain('[[models]]')

    const keys = Array.from(container.querySelectorAll('[data-toml-token="key"]')).map((node) => node.textContent)
    expect(keys).toContain('temperature')
    expect(keys).toContain('"nested.key"')
    expect(keys).toContain('plugin.enabled')
    expect(keys).toContain('headers')

    const values = Array.from(container.querySelectorAll('[data-toml-token="value"]')).map((node) => node.textContent)
    expect(values).toEqual(
      expect.arrayContaining(['true', '0.55', '1', '{ }', '"literal"', 'false', '"unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL"'])
    )
  })
})
