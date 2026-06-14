import { describe, expect, it } from 'vitest'
import { splitAssistantThinking } from '@/features/chat/components/thinking-segments'

describe('splitAssistantThinking', () => {
  it('returns plain assistant output as a response segment', () => {
    expect(splitAssistantThinking('The capital of France is Paris.')).toEqual([
      { kind: 'response', text: 'The capital of France is Paris.' }
    ])
  })

  it('splits explicit think tags from final response text', () => {
    expect(splitAssistantThinking('<think>Check facts.</think> Paris.')).toEqual([
      { kind: 'thinking', text: 'Check facts.', open: false },
      { kind: 'response', text: ' Paris.' }
    ])
  })

  it('supports streamed output where the opening think tag is missing', () => {
    expect(splitAssistantThinking('Check facts.</think> Paris.')).toEqual([
      { kind: 'thinking', text: 'Check facts.', open: false },
      { kind: 'response', text: ' Paris.' }
    ])
  })

  it('splits Gemma channel thinking from final response text', () => {
    expect(
      splitAssistantThinking('<|channel|>thoughtCheck whether the prompt is a test.<channel|>Test received!')
    ).toEqual([
      { kind: 'thinking', text: 'Check whether the prompt is a test.', open: false },
      { kind: 'response', text: 'Test received!' }
    ])
  })

  it('splits Gemma channel thinking when the thought marker is missing the leading pipe', () => {
    expect(splitAssistantThinking('<channel|>thoughtCheck facts.<channel|>Final answer.')).toEqual([
      { kind: 'thinking', text: 'Check facts.', open: false },
      { kind: 'response', text: 'Final answer.' }
    ])
  })

  it('keeps an unclosed think segment open for live streams', () => {
    expect(splitAssistantThinking('<think>Checking facts')).toEqual([
      { kind: 'thinking', text: 'Checking facts', open: true }
    ])
  })

  it('keeps untagged streamed text inside an open thinking segment until the close tag arrives', () => {
    expect(splitAssistantThinking('Checking facts as tokens stream', { streaming: true })).toEqual([
      { kind: 'thinking', text: 'Checking facts as tokens stream', open: true }
    ])
  })

  it('opens the thinking segment as soon as the streamed think tag appears', () => {
    expect(splitAssistantThinking('<think>')).toEqual([{ kind: 'thinking', text: '', open: true }])
  })

  it('finds mixed-case tags without shifting indices after unicode text', () => {
    expect(splitAssistantThinking('İstanbul <THINK>Check facts.</THINK> Paris.')).toEqual([
      { kind: 'response', text: 'İstanbul ' },
      { kind: 'thinking', text: 'Check facts.', open: false },
      { kind: 'response', text: ' Paris.' }
    ])
  })
})
