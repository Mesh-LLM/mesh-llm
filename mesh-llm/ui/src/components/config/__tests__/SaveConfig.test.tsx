import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../../../lib/api', () => ({
  broadcastConfig: vi.fn(),
}));

import { broadcastConfig } from '../../../lib/api';
import type { MeshConfig } from '../../../types/config';
import { SaveConfig } from '../SaveConfig';

const config: MeshConfig = { version: 1, nodes: [] };

describe('SaveConfig', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('save-config-btn is disabled when config is not dirty', () => {
    render(<SaveConfig config={config} isDirty={false} onSaveSuccess={vi.fn()} />);
    expect(screen.getByTestId('save-config-btn')).toBeDisabled();
  });

  it('save-config-btn is enabled when config is dirty', () => {
    render(<SaveConfig config={config} isDirty={true} isConfigValid={true} onSaveSuccess={vi.fn()} />);
    expect(screen.getByTestId('save-config-btn')).not.toBeDisabled();
  });

  it('disables save-config-btn when config is invalid', () => {
    render(<SaveConfig config={config} isDirty={true} isConfigValid={false} invalidReason="Invalid TOML" onSaveSuccess={vi.fn()} />);
    expect(screen.getByTestId('save-config-btn')).toBeDisabled();
    expect(screen.getByTestId('save-config-btn')).toHaveAttribute('title', 'Invalid TOML');
  });

  it('clicking Save opens the diff dialog', async () => {
    const user = userEvent.setup();

    render(<SaveConfig config={config} isDirty={true} onSaveSuccess={vi.fn()} />);

    await user.click(screen.getByTestId('save-config-btn'));

    expect(screen.getByTestId('config-diff-dialog')).toBeInTheDocument();
  });

  it('clicking Confirm & Save calls broadcastConfig', async () => {
    const user = userEvent.setup();
    vi.mocked(broadcastConfig).mockResolvedValue({ ok: true, saved: 1, total: 1, failed: [] });

    render(<SaveConfig config={config} isDirty={true} onSaveSuccess={vi.fn()} />);

    await user.click(screen.getByTestId('save-config-btn'));
    await user.click(screen.getByTestId('confirm-save-button'));

    await waitFor(() => {
      expect(broadcastConfig).toHaveBeenCalledTimes(1);
    });
  });

  it('clicking Cancel does not call broadcastConfig', async () => {
    const user = userEvent.setup();

    render(<SaveConfig config={config} isDirty={true} onSaveSuccess={vi.fn()} />);

    await user.click(screen.getByTestId('save-config-btn'));
    await user.click(screen.getByTestId('cancel-save-button'));

    expect(broadcastConfig).not.toHaveBeenCalled();
  });

  it('calls broadcastConfig on save click', async () => {
    const user = userEvent.setup();
    vi.mocked(broadcastConfig).mockResolvedValue({ ok: true, saved: 2, total: 2, failed: [] });

    render(<SaveConfig config={config} isDirty={true} onSaveSuccess={vi.fn()} />);

    await user.click(screen.getByTestId('save-config-btn'));
    await user.click(screen.getByTestId('confirm-save-button'));

    await waitFor(() => {
      expect(broadcastConfig).toHaveBeenCalledTimes(1);
    });
    expect(broadcastConfig).toHaveBeenCalledWith(expect.any(String));
  });

  it('shows success toast after broadcast succeeds', async () => {
    const user = userEvent.setup();
    vi.mocked(broadcastConfig).mockResolvedValue({ ok: true, saved: 2, total: 2, failed: [] });

    render(<SaveConfig config={config} isDirty={true} onSaveSuccess={vi.fn()} />);

    await user.click(screen.getByTestId('save-config-btn'));
    await user.click(screen.getByTestId('confirm-save-button'));

    await waitFor(() => {
      expect(screen.getByTestId('save-toast')).toBeInTheDocument();
    });
    expect(screen.getByTestId('save-toast')).toHaveTextContent('Configuration saved to 2/2 nodes');
  });

  it('calls onSaveSuccess with the saved config snapshot after success', async () => {
    const user = userEvent.setup();
    const onSaveSuccess = vi.fn();
    vi.mocked(broadcastConfig).mockResolvedValue({ ok: true, saved: 2, total: 2, failed: [] });

    render(<SaveConfig config={config} isDirty={true} onSaveSuccess={onSaveSuccess} />);

    await user.click(screen.getByTestId('save-config-btn'));
    await user.click(screen.getByTestId('confirm-save-button'));

    await waitFor(() => {
      expect(onSaveSuccess).toHaveBeenCalledWith(config);
    });
  });

  it('shows error toast when broadcast fails entirely', async () => {
    const user = userEvent.setup();
    vi.mocked(broadcastConfig).mockResolvedValue({ ok: false, error: 'Failed to save config locally: disk full' });

    render(<SaveConfig config={config} isDirty={true} onSaveSuccess={vi.fn()} />);

    await user.click(screen.getByTestId('save-config-btn'));
    await user.click(screen.getByTestId('confirm-save-button'));

    await waitFor(() => {
      expect(screen.getByTestId('save-toast')).toBeInTheDocument();
    });
    expect(screen.getByTestId('save-toast')).toHaveTextContent('Failed to save config locally: disk full');
  });

  it('shows partial failure toast with saved count and failed hostnames', async () => {
    const user = userEvent.setup();
    vi.mocked(broadcastConfig).mockResolvedValue({
      ok: false,
      saved: 1,
      total: 2,
      failed: ['beta.local'],
    });

    render(<SaveConfig config={config} isDirty={true} onSaveSuccess={vi.fn()} />);

    await user.click(screen.getByTestId('save-config-btn'));
    await user.click(screen.getByTestId('confirm-save-button'));

    await waitFor(() => {
      expect(screen.getByTestId('save-toast')).toBeInTheDocument();
    });

    const toast = screen.getByTestId('save-toast');
    expect(toast).toHaveTextContent('Saved: 1/2');
    expect(toast).toHaveTextContent('Failed: beta.local');
  });

  it('calls onSaveSuccess on partial failure (local node saved)', async () => {
    const user = userEvent.setup();
    const onSaveSuccess = vi.fn();
    vi.mocked(broadcastConfig).mockResolvedValue({
      ok: false,
      saved: 1,
      total: 2,
      failed: ['beta.local'],
    });

    render(<SaveConfig config={config} isDirty={true} onSaveSuccess={onSaveSuccess} />);

    await user.click(screen.getByTestId('save-config-btn'));
    await user.click(screen.getByTestId('confirm-save-button'));

    await waitFor(() => {
      expect(onSaveSuccess).toHaveBeenCalledWith(config);
    });
  });

  it('does not call onSaveSuccess when broadcast fails entirely', async () => {
    const user = userEvent.setup();
    const onSaveSuccess = vi.fn();
    vi.mocked(broadcastConfig).mockResolvedValue({ ok: false, error: 'Network error: connection refused' });

    render(<SaveConfig config={config} isDirty={true} onSaveSuccess={onSaveSuccess} />);

    await user.click(screen.getByTestId('save-config-btn'));
    await user.click(screen.getByTestId('confirm-save-button'));

    await waitFor(() => {
      expect(screen.getByTestId('save-toast')).toBeInTheDocument();
    });

    expect(onSaveSuccess).not.toHaveBeenCalled();
  });

  it('revert button is disabled when config is not dirty', () => {
    const onRevert = vi.fn();
    render(<SaveConfig config={config} isDirty={false} onSaveSuccess={vi.fn()} onRevert={onRevert} />);
    expect(screen.getByTestId('revert-config-btn')).toBeDisabled();
  });

  it('revert button calls onRevert when clicked', async () => {
    const user = userEvent.setup();
    const onRevert = vi.fn();
    render(<SaveConfig config={config} isDirty={true} onSaveSuccess={vi.fn()} onRevert={onRevert} />);

    await user.click(screen.getByTestId('revert-config-btn'));
    expect(onRevert).toHaveBeenCalledTimes(1);
  });

  it('does not render revert button when onRevert is not provided', () => {
    render(<SaveConfig config={config} isDirty={true} onSaveSuccess={vi.fn()} />);
    expect(screen.queryByTestId('revert-config-btn')).not.toBeInTheDocument();
  });

  it('toast container has bottom-16 class for proper positioning', async () => {
    const user = userEvent.setup();
    vi.mocked(broadcastConfig).mockResolvedValue({ ok: true, saved: 1, total: 1, failed: [] });

    render(<SaveConfig config={config} isDirty={true} onSaveSuccess={vi.fn()} />);

    await user.click(screen.getByTestId('save-config-btn'));
    await user.click(screen.getByTestId('confirm-save-button'));

    await waitFor(() => {
      const toastContainer = screen.getByTestId('save-toast').parentElement;
      expect(toastContainer).toHaveClass('bottom-16');
      expect(toastContainer).not.toHaveClass('bottom-4');
    });
  });

  it('broadcasts a config with placement_mode separate and gpu_index in the serialized TOML', async () => {
    const user = userEvent.setup();
    vi.mocked(broadcastConfig).mockResolvedValue({ ok: true, saved: 1, total: 1, failed: [] });

    const specialConfig: MeshConfig = {
      version: 1,
      nodes: [
        {
          node_id: 'node-a',
          placement_mode: 'separate',
          models: [{ name: 'Qwen3', gpu_index: 1 }],
        },
      ],
    };

    render(<SaveConfig config={specialConfig} isDirty={true} onSaveSuccess={vi.fn()} />);

    await user.click(screen.getByTestId('save-config-btn'));
    await user.click(screen.getByTestId('confirm-save-button'));

    await waitFor(() => {
      expect(broadcastConfig).toHaveBeenCalledTimes(1);
    });

    const broadcastArg = vi.mocked(broadcastConfig).mock.calls[0]?.[0] ?? '';
    expect(broadcastArg).toContain('placement_mode');
    expect(broadcastArg).toContain('separate');
    expect(broadcastArg).toContain('gpu_index');
  });

  it('onSaveSuccess receives a config snapshot with placement_mode and gpu_index preserved after broadcast', async () => {
    const user = userEvent.setup();
    const onSaveSuccess = vi.fn();
    vi.mocked(broadcastConfig).mockResolvedValue({ ok: true, saved: 1, total: 1, failed: [] });

    const specialConfig: MeshConfig = {
      version: 1,
      nodes: [
        {
          node_id: 'node-a',
          placement_mode: 'separate',
          models: [{ name: 'Qwen3', gpu_index: 1 }],
        },
      ],
    };

    render(<SaveConfig config={specialConfig} isDirty={true} onSaveSuccess={onSaveSuccess} />);

    await user.click(screen.getByTestId('save-config-btn'));
    await user.click(screen.getByTestId('confirm-save-button'));

    await waitFor(() => {
      expect(onSaveSuccess).toHaveBeenCalledTimes(1);
    });

    const savedConfig = onSaveSuccess.mock.calls[0]?.[0] as MeshConfig;
    expect(savedConfig.version).toBe(1);
    const node = savedConfig.nodes[0];
    expect(node?.placement_mode).toBe('separate');
    expect(node?.models[0]?.gpu_index).toBe(1);
  });

  it('diff dialog shows added and removed models per node', async () => {
    const user = userEvent.setup();

    const currentConfig: MeshConfig = {
      version: 1,
      nodes: [{ node_id: 'node-a', models: [{ name: 'Qwen3' }, { name: 'NewModel' }] }],
    };
    const baseSavedConfig: MeshConfig = {
      version: 1,
      nodes: [{ node_id: 'node-a', models: [{ name: 'Qwen3' }, { name: 'OldModel' }] }],
    };

    render(
      <SaveConfig
        config={currentConfig}
        savedConfig={baseSavedConfig}
        isDirty={true}
        onSaveSuccess={vi.fn()}
      />,
    );

    await user.click(screen.getByTestId('save-config-btn'));

    const dialog = screen.getByTestId('config-diff-dialog');
    expect(dialog).toBeInTheDocument();
    expect(dialog).toHaveTextContent('node-a');
    expect(dialog).toHaveTextContent('+ NewModel added');
    expect(dialog).toHaveTextContent('− OldModel removed');
  });
});
