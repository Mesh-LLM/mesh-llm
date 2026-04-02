import { useState } from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../../lib/api', () => ({
  broadcastConfig: vi.fn(),
  fetchAuthoredConfig: vi.fn(),
}));

import { broadcastConfig } from '../../lib/api';
import type { ConfigValidationError } from '../../lib/api';
import type { MeshConfig } from '../../types/config';
import { configHistoryReducer, type ConfigHistoryState } from '../ConfigPage';
import { SaveConfig } from '../../components/config/SaveConfig';

function makeConfig(nodeId: string): MeshConfig {
  return { version: 1, nodes: [{ node_id: nodeId, models: [] }] };
}

function emptyHistory(config: MeshConfig): ConfigHistoryState {
  return { config, past: [], future: [] };
}

describe('configHistoryReducer undo/redo', () => {
  it('3 dispatches then UNDO lands on state 2, second UNDO lands on state 1, REDO returns to state 2', () => {
    const c1 = makeConfig('node-1');
    const c2 = makeConfig('node-2');
    const c3 = makeConfig('node-3');

    let state = emptyHistory(c1);
    state = configHistoryReducer(state, { type: 'SET_CONFIG', config: c2 });
    state = configHistoryReducer(state, { type: 'SET_CONFIG', config: c3 });

    expect(state.config).toBe(c3);
    expect(state.past).toHaveLength(2);

    state = configHistoryReducer(state, { type: 'UNDO' });
    expect(state.config).toBe(c2);
    expect(state.past).toHaveLength(1);
    expect(state.future).toHaveLength(1);

    state = configHistoryReducer(state, { type: 'UNDO' });
    expect(state.config).toBe(c1);
    expect(state.past).toHaveLength(0);
    expect(state.future).toHaveLength(2);

    state = configHistoryReducer(state, { type: 'REDO' });
    expect(state.config).toBe(c2);
    expect(state.past).toHaveLength(1);
    expect(state.future).toHaveLength(1);
  });

  it('31 dispatches keep history length at 30 (oldest entry dropped)', () => {
    let state = emptyHistory(makeConfig('node-0'));

    for (let i = 1; i <= 31; i++) {
      state = configHistoryReducer(state, { type: 'SET_CONFIG', config: makeConfig(`node-${i}`) });
    }

    expect(state.past).toHaveLength(30);
    expect(state.config).toEqual(makeConfig('node-31'));
    expect(state.past[0]).toEqual(makeConfig('node-1'));
  });

  it('UNDO with empty history returns state unchanged without crashing', () => {
    const c = makeConfig('node-x');
    const state = emptyHistory(c);
    const next = configHistoryReducer(state, { type: 'UNDO' });
    expect(next).toBe(state);
    expect(next.config).toBe(c);
  });

  it('REDO with empty future returns state unchanged without crashing', () => {
    const c = makeConfig('node-y');
    const state = emptyHistory(c);
    const next = configHistoryReducer(state, { type: 'REDO' });
    expect(next).toBe(state);
    expect(next.config).toBe(c);
  });

  it('new dispatch after UNDO clears the redo stack', () => {
    const c1 = makeConfig('node-1');
    const c2 = makeConfig('node-2');
    const c3 = makeConfig('node-3');

    let state = emptyHistory(c1);
    state = configHistoryReducer(state, { type: 'SET_CONFIG', config: c2 });
    state = configHistoryReducer(state, { type: 'UNDO' });
    expect(state.future).toHaveLength(1);

    state = configHistoryReducer(state, { type: 'SET_CONFIG', config: c3 });
    expect(state.future).toHaveLength(0);
    expect(state.config).toBe(c3);
  });

  it('action that produces no config change does not push to history', () => {
    const c = makeConfig('node-stable');
    let state = emptyHistory(c);

    const initialPastLength = state.past.length;
    state = configHistoryReducer(state, {
      type: 'UNASSIGN_MODEL',
      nodeId: 'nonexistent',
      modelName: 'nonexistent',
    });

    expect(state.past).toHaveLength(initialPastLength);
  });
});

function BackendErrorWrapper({ config }: { config: MeshConfig }) {
  const [errors, setErrors] = useState<ConfigValidationError[]>([]);
  return (
    <>
      <SaveConfig
        config={config}
        isDirty={true}
        onSaveSuccess={vi.fn()}
        onBackendErrors={setErrors}
      />
      {errors.map((err) => (
        <p key={`${err.code}::${err.path}`} data-testid="backend-validation-error" className="text-destructive text-xs mt-1">
          {err.message}
        </p>
      ))}
    </>
  );
}

describe('backend validation errors', () => {
  const config: MeshConfig = { version: 1, nodes: [] };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows validation error inline after broadcastConfig returns errors array', async () => {
    const user = userEvent.setup();
    vi.mocked(broadcastConfig).mockResolvedValue({
      ok: false,
      errors: [{ code: 'split_gap', path: 'nodes[0].models[0].split', message: 'Gap between layers 5 and 10' }],
    });

    render(<BackendErrorWrapper config={config} />);
    await user.click(screen.getByTestId('save-config-btn'));
    await user.click(screen.getByTestId('confirm-save-button'));

    await waitFor(() => {
      expect(screen.getByText('Gap between layers 5 and 10')).toBeInTheDocument();
    });
  });

  it('clears validation errors after next successful broadcastConfig', async () => {
    const user = userEvent.setup();

    vi.mocked(broadcastConfig).mockResolvedValueOnce({
      ok: false,
      errors: [{ code: 'split_gap', path: 'nodes[0].models[0].split', message: 'Gap between layers 5 and 10' }],
    });

    render(<BackendErrorWrapper config={config} />);
    await user.click(screen.getByTestId('save-config-btn'));
    await user.click(screen.getByTestId('confirm-save-button'));

    await waitFor(() => {
      expect(screen.getByText('Gap between layers 5 and 10')).toBeInTheDocument();
    });

    vi.mocked(broadcastConfig).mockResolvedValueOnce({ ok: true, saved: 1, total: 1, failed: [] });
    await user.click(screen.getByTestId('save-config-btn'));
    await user.click(screen.getByTestId('confirm-save-button'));

    await waitFor(() => {
      expect(screen.queryByText('Gap between layers 5 and 10')).not.toBeInTheDocument();
    });
  });
});
