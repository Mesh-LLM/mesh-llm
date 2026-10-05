import { act, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { DisabledControlFrame } from '@/features/configuration/components/settings/DisabledControlFrame'

const DETAIL_TEXT = 'Requires speculation-mode = draft'
const VISIBLE_HELPER_TEXT = 'Path can point to a local model artifact.'

type DisabledControlFrameTestOptions = {
  readonly details?: readonly string[]
  readonly disabled?: boolean
  readonly disabledDetails?: readonly string[]
}

function renderDisabledControlFrame({
  details = [],
  disabled = false,
  disabledDetails = []
}: DisabledControlFrameTestOptions = {}) {
  return render(
    <DisabledControlFrame
      details={details}
      detailsId="draft-mode-details"
      disabled={disabled}
      disabledDetails={disabledDetails}
    >
      <div data-testid="draft-mode-control">Draft mode control</div>
    </DisabledControlFrame>
  )
}

describe('DisabledControlFrame', () => {
  it('renders children without helper chrome when there are no details', () => {
    renderDisabledControlFrame()

    expect(screen.getByTestId('draft-mode-control')).toBeInTheDocument()
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
    expect(screen.queryByText(DETAIL_TEXT)).not.toBeInTheDocument()
  })

  it('renders a helper info trigger instead of a visible detail card', () => {
    renderDisabledControlFrame({ details: [DETAIL_TEXT], disabled: true })

    expect(screen.getByTestId('draft-mode-control')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /setting information/i })).toBeInTheDocument()
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
    expect(screen.queryByText(DETAIL_TEXT)).not.toBeInTheDocument()
  })

  it('reveals helper tooltip content on hover', async () => {
    const user = userEvent.setup()
    renderDisabledControlFrame({ details: [DETAIL_TEXT], disabled: true })

    await user.hover(screen.getByRole('button', { name: /setting information/i }))

    expect(await screen.findByRole('tooltip')).toHaveTextContent(DETAIL_TEXT)
  })

  it('reveals helper tooltip content on focus', async () => {
    renderDisabledControlFrame({ details: [DETAIL_TEXT], disabled: true })

    await act(async () => {
      screen.getByRole('button', { name: /setting information/i }).focus()
    })

    expect(await screen.findByRole('tooltip')).toHaveTextContent(DETAIL_TEXT)
  })

  it('keeps helper and disabled details behind distinct icon triggers', async () => {
    const user = userEvent.setup()
    const { container } = renderDisabledControlFrame({
      details: [VISIBLE_HELPER_TEXT],
      disabled: true,
      disabledDetails: [DETAIL_TEXT]
    })

    expect(screen.queryByText(VISIBLE_HELPER_TEXT)).not.toBeInTheDocument()
    expect(container.querySelector('[data-disabled-control-frame="true"]')).not.toBeInTheDocument()
    expect(screen.queryByText(DETAIL_TEXT)).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /setting information/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /why unavailable/i })).toBeInTheDocument()

    await user.hover(screen.getByRole('button', { name: /setting information/i }))
    expect(await screen.findByRole('tooltip')).toHaveTextContent(VISIBLE_HELPER_TEXT)
    await user.unhover(screen.getByRole('button', { name: /setting information/i }))

    await act(async () => {
      screen.getByRole('button', { name: /why unavailable/i }).focus()
    })

    expect(await screen.findByText(DETAIL_TEXT, { selector: 'div' })).toBeInTheDocument()
  })
})
