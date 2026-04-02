import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('shiki', () => ({
  codeToHtml: vi.fn().mockResolvedValue(
    '<pre class="shiki github-dark"><code><span class="line"><span style="color:#79C0FF">version</span></span></code></pre>',
  ),
}));

import type { MeshConfig } from '../../../types/config';
import { TomlEditor } from '../TomlEditor';

const emptyConfig: MeshConfig = { version: 1, nodes: [] };
const configWithNode: MeshConfig = {
  version: 1,
  nodes: [{ node_id: 'abc123', models: [{ name: 'TestModel' }] }],
};

describe('TomlEditor', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('renders initially with textarea, header, and valid status indicator', async () => {
    render(<TomlEditor config={emptyConfig} onConfigChange={vi.fn()} />);

    await act(async () => {
      await vi.runAllTimersAsync();
    });

    expect(screen.getByTestId('toml-textarea')).toBeInTheDocument();
    expect(screen.getByTestId('toml-editor-panel')).toHaveClass('resize-y');
    expect(screen.getByTestId('toml-valid')).toBeInTheDocument();
    expect(screen.queryByTestId('toml-error')).not.toBeInTheDocument();
    expect(screen.getByText('Configuration TOML')).toBeVisible();
    expect(screen.getByText(/\d+ lines/)).toBeVisible();
  });

  it('syncs visual → TOML when the config prop changes', async () => {
    const onConfigChange = vi.fn();
    const { rerender } = render(<TomlEditor config={emptyConfig} onConfigChange={onConfigChange} />);

    await act(async () => {
      await vi.runAllTimersAsync();
    });

    rerender(<TomlEditor config={configWithNode} onConfigChange={onConfigChange} />);

    await act(async () => {
      await vi.runAllTimersAsync();
    });

    const textarea = screen.getByTestId('toml-textarea') as HTMLTextAreaElement;
    expect(textarea.value).toContain('abc123');
  });

  it('syncs TOML → visual after debounce when valid TOML is typed', async () => {
    const onConfigChange = vi.fn();
    render(<TomlEditor config={emptyConfig} onConfigChange={onConfigChange} />);

    await act(async () => {
      await vi.runAllTimersAsync();
    });

    const textarea = screen.getByTestId('toml-textarea') as HTMLTextAreaElement;
    const validToml = `version = 1\n\n[[nodes]]\nnode_id = "xyz"\nmodels = []\n`;

    fireEvent.change(textarea, { target: { value: validToml } });

    await act(async () => {
      vi.advanceTimersByTime(200);
    });

    expect(onConfigChange).toHaveBeenCalledWith(
      expect.objectContaining({
        nodes: expect.arrayContaining([
          expect.objectContaining({ node_id: 'xyz' }),
        ]),
      }),
    );
    expect(screen.getByTestId('toml-valid')).toBeInTheDocument();
    expect(screen.queryByTestId('toml-error')).not.toBeInTheDocument();
  });

  it('shows toml-error and does not call onConfigChange for invalid TOML', async () => {
    const onConfigChange = vi.fn();
    render(<TomlEditor config={emptyConfig} onConfigChange={onConfigChange} />);

    await act(async () => {
      await vi.runAllTimersAsync();
    });

    const textarea = screen.getByTestId('toml-textarea') as HTMLTextAreaElement;

    fireEvent.change(textarea, { target: { value: '[[[ this is not valid toml' } });

    await act(async () => {
      vi.advanceTimersByTime(200);
    });

    expect(screen.getByTestId('toml-error')).toBeInTheDocument();
    expect(screen.queryByTestId('toml-valid')).not.toBeInTheDocument();
    expect(onConfigChange).not.toHaveBeenCalled();
  });

  it('reports parse validity changes to the parent', async () => {
    const onParseErrorChange = vi.fn();
    render(<TomlEditor config={emptyConfig} onConfigChange={vi.fn()} onParseErrorChange={onParseErrorChange} />);

    await act(async () => {
      await vi.runAllTimersAsync();
    });

    const textarea = screen.getByTestId('toml-textarea') as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: '[[[ this is not valid toml' } });

    await act(async () => {
      vi.advanceTimersByTime(200);
    });

    expect(onParseErrorChange).toHaveBeenLastCalledWith('Invalid TOML');

    fireEvent.change(textarea, { target: { value: 'version = 1\n\nnodes = []\n' } });

    await act(async () => {
      vi.advanceTimersByTime(200);
    });

    expect(onParseErrorChange).toHaveBeenLastCalledWith(null);
  });

  it('applies shiki syntax highlighting to the highlight div after 200ms debounce', async () => {
    render(<TomlEditor config={configWithNode} onConfigChange={vi.fn()} />);

    await act(async () => {
      await vi.runAllTimersAsync();
    });

    const highlightDiv = screen.getByTestId('toml-highlight');
    expect(highlightDiv.innerHTML).toContain('<span');
  });
});
