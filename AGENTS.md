# AGENTS.md: where to edit

## Simulator (`redspark/sim.py`)
- New block: `Block` enum, `DIODES`, `CONDUCTORS`, `_emitted`, `neighbor_changed`, `_tile_tick`, `place`, `to_dict`/`from_dict`
- Support/placement rules: `SUPPORT_BLOCKS`, `SUPPORTS_*`, `SUPPORTS_BELOW`, `FLOOR_IS_SUPPORT`, `can_place`, `_drop_unsupported`
- Tick order / priorities / delays: `step`, `schedule`, `*_DELAY`, `EXTREMELY_HIGH..NORMAL`, `UPDATE_ORDER`, `VALUES_ORDER`
- Power rules: `signal`, `direct_signal_to`, `has_neighbor_signal(_qc)`
- Dust: `_connects`, `dust_shape`, `_dust_target`, `_update_dust`
- Repeater/comparator: `_repeater_*`, `_comparator_*`, `_diode_prioritized`, `_update_front`, `comparator_update`
- Module docstring = rules summary; keep in sync

## Tasks / training
- Gates, truth-table scoring: `redspark/tasks.py` (`GATES`, `evaluate_gate`)
- PPO env, action mask, obs: `redspark/env.py` (`PLACEABLE`, `COMPARATOR_CHOICES`, `_valid`, `action_masks`)
- PPO script, CLI flags: `redspark/train.py`
- GA trainer (Train tab): `redspark/designer.py` (`PLACEABLE`, `Layout.supported`, `Builder.mask`, `Trainer`)
- Example circuits: `circuits/make_examples.py` then run it to rewrite `circuits/*.json`

## Server / app
- WebSocket messages, Train tab backend: `redspark/server.py` (`Session.handle`, `Training`)
- Launcher: `redspark/app.py`, `RedSpark.bat`
- Deps: `pyproject.toml`

## Viewer (`web/src/`)
- Block meshes, dust shape, clicks, panes/layers: `render.ts` (`buildBlock`, `mats`, `geo`, `CircuitView`)
- Pixel textures: `textures.ts` (seeded, 16x16, nearest filter)
- Train tab UI: `train.ts`, `chart.ts`, `web/index.html`
- View tab, tabs, socket: `main.ts`, `net.ts`
- Message/block types: `types.ts` (keep in sync with `server.py` / `to_dict`)

## Build / test
- After any `web/` change: `cd web && npm run build` writes `redspark/viewer/` (served by the app; commit it)
- Tests: `python -m pytest` (`tests/test_sim.py`, `test_comparator.py`, `test_env.py`, `test_designer.py`, `test_server.py`)
- Docs: `README.md`

## Gotchas
- No FastAPI/pydantic or other compiled Python deps: Smart App Control blocks their DLLs. Server is Starlette.
- Use `python -m pip/pytest/uvicorn`: `.exe` launchers are blocked.
- `web/node_modules` was installed on Windows; on Linux/macOS build from a fresh `npm install` elsewhere, don't reuse it.
- Windows folder names ignore case: `redspark/` == `RedSpark/`.
- Art must be original (no Minecraft assets).
