# RedSpark

Train a neural network to design redstone-style logic circuits (NOT, AND, OR, XOR...).

RedSpark has three parts that share one simulator:

| Part | Tech | Where |
|---|---|---|
| Simulator | Python + NumPy, tick by tick | `redspark/sim.py` |
| Training | Gymnasium env + PyTorch + Stable-Baselines3 (MaskablePPO) | `redspark/env.py`, `redspark/train.py` |
| Viewer | Starlette WebSocket backend + three.js / TypeScript / Vite frontend | `redspark/server.py`, `web/` |

Everything (rules, visuals, circuits) is original. It is inspired by redstone logic but uses no game assets and is not a copy of any game's exact rules.

## Open the app (Windows, one click)

Double-click **`RedSpark.bat`**. The first run creates a Python environment and installs the few packages the viewer needs (about a minute). After that it opens straight away, in its own window. Close the window to quit.

You only need Python 3.10+ installed. Node is not needed, because a built copy of the viewer ships in `redspark/viewer/`. On macOS or Linux, run `python -m redspark.app` instead.

## Quick start (development)

Needs Python 3.10+ and Node 20+.

```bash
# 1. Python side (simulator, training, backend)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[train,server,dev]"
python -m pytest                   # 28 tests

# 2. Train an agent to build a NOT gate (about a minute on a laptop CPU)
python -m redspark.train --gate NOT --timesteps 40000
#    -> prints the circuit it built and saves circuits/trained_not.json

# 3. Viewer with live reload while you edit web/ (otherwise just use RedSpark.bat)
python -m uvicorn redspark.server:app --port 8000
cd web && npm install && npm run dev
#    -> open http://localhost:5173
```

After changing anything in `web/`, run `npm run build` there to refresh the copy in `redspark/viewer/` that the app uses.

In the viewer, pick a circuit from the list, click a lever to flip it, and use Play / Step to run ticks. Any JSON file you drop into `circuits/` shows up in the list, including the ones training writes.

## Truth-table training in the viewer

Open the **Train** tab in the app to design your own task and watch a network learn it, no command line needed.

1. Set the grid size (width, depth, up to 3 layers high) and pick the layer to place on.
2. With **Input** selected, click cells to add input levers (A to D). With **Output**, add output lamps (OUT1 to OUT4). **Erase** removes one.
3. Fill the truth table on the right. Each output cell cycles 0, 1, or – (don't care). The preset menu fills OUT1 with NOT, AND, OR, XOR and so on.
4. Pick a **Method** and press **Run**. The default, **Evolve layouts**, evolves circuits directly. The older **Evolve a network** method evolves small neural networks that build circuits block by block. Both are genetic algorithms in `redspark/designer.py` and use only NumPy, so they run in the app without PyTorch. How they search and how they score is described in the next section.

Controls: **Speed** sets generations per second (or as fast as possible). **Show** picks which circuits get drawn: successes only (every row passes), new bests, or every generation. **Draw at most N circuits/s** caps redrawing, so "every generation" at full speed draws the latest generation a few times a second while the chart still records every one. The shown circuit steps through its truth-table rows, and the table marks each row passed or failed.

The chart shows rows passed per generation for the best design and the population mean. Training never resets on its own, so a stuck run keeps its progress; press **Restart** if you want a fresh population. Single-input tasks like NOT solve within a few dozen generations. Two-input gates need room: on a one-layer 7 x 5 grid AND takes seconds to a minute (14 blocks), and XOR needs a bigger block budget (about 24). **Save best** writes the best circuit to `circuits/<name>.json`, which then appears in the View tab.

## How the Train tab searches and scores

### Evolve layouts (default)

There is no neural network. Each individual in the population *is* a circuit layout, and every move follows the simulator's placement and support rules (see `LayoutTrainer` in `redspark/designer.py`).

- **Population:** 48 layouts. The 8 best are kept unchanged each generation (elites); the other 40 slots are children.
- **Choosing parents:** tournament selection. Three layouts are picked at random and the best of the three becomes a parent.
- **Mutation:** a child is a copy of a parent with one edit most of the time, sometimes several (the count is geometric, mean about 1.8). Each edit is one of:
  - **Building block (25%):** add a whole small piece in one move, such as a wire of 1 to 4 dust, a block with a torch on its far side, side or top (an inverter), a repeater into a block, or a lone repeater. A random single block rarely lines up into something useful, so these pieces are what make progress quick.
  - **Add (25%):** place one legal block, 80% of the time next to an existing block so that signal paths grow.
  - **Remove (22%):** delete a random block. Anything that rested on it breaks off, as in the simulator.
  - **Change (28%):** swap a block for a different legal block, facing or comparator mode in the same cell.
- **Crossover:** a quarter of children take the left part (or front part) of one parent and the rest from another, rebuilt so that the result is legal. Half of those are also mutated.
- **Only new designs:** children that are copies of an elite or of each other are re-rolled, and every layout's score is cached.
- **No resets:** the population is never thrown away. The best design so far always stays on screen.
- **Comparators:** both compare and subtract mode are in the parts list.

### Evolve a network (older)

A one-hidden-layer network (32 tanh neurons) looks at the grid and picks the next block to place, or stops, until the block budget is used. The population is 32 networks (6 elites, Gaussian weight noise with sigma 0.08, children re-rolled if they build the same circuit as before). The network is much slower than the layout method for the same result, so it is kept mainly for comparison. It also does not reset.

This setup is recommended for only basic logic gates, as large layouts are too complex to lay out a network.

### How a circuit is scored

Both methods rank circuits with the same function, `score_circuit`. The ranking key is compared one number at a time; the first difference decides:

1. **Rows passed.** Each truth-table row is run by flipping the levers, letting the circuit settle, and checking every output lamp (don't-care cells accept anything). A row also fails if the circuit is still changing after the settle limit. More rows always beats fewer.
2. **Fully correct circuits** (every row passes) are then ranked by, in order: fewer blocks, a smaller bounding box (volume), and a faster activation time (the game tick on which the slowest row's outputs stop changing). A correct circuit therefore always beats a wrong one, and among correct ones the smallest wins.
3. **Not yet correct circuits** are ranked by a hint score between rows passed and block count, so that partial progress counts:
   `hint = 0.4 x bits_right + 0.2 x reach + 0.4 x match - 0.5 if unstable`
   - **bits_right:** the fraction of required output bits that are right, over all rows.
   - **reach:** how close a cell that changes with the inputs gets to an output lamp (1 when it touches it, 0 when it is a full grid span away).
   - **match:** how close the *right* signal gets to the lamp. For each cell, its on/off pattern across the rows is compared to the output column. The cell scores its closeness to the lamp times the squared fraction of rows it matches, and a cell with the exactly *opposite* pattern still counts at 85% because a single torch flips it (for AND, a NAND signal next to the lamp is one inverter away). The best cell counts in full and the best cell with a different signal counts at half, so building a second ingredient (for example both NOT A and NOT B for a NOR) is also progress.
   - **unstable:** circuits that never settle lose 0.5.
   Remaining ties go to fewer blocks.

This is why "always off" does not trap AND any more: it passes 3 of 4 rows but has no signal near the lamp, so a design with 3 rows plus a useful signal near the output outranks it.

## The simulation model

The simulator follows how Java Edition processes redstone. It is written from public technical write-ups, not game code, and the full notes are at the top of `redspark/sim.py`.

- **20 game ticks per second.** One redstone tick is 2 game ticks. A torch reacts in 2 gt, a repeater in 2/4/6/8 gt, and a lamp turns on instantly but off 4 gt later.
- **Each game tick** runs the scheduled tile ticks that are due, ordered by due time, then priority, then scheduling order. Repeaters use priority -3 when facing into another diode, -2 when turning off and -1 when turning on. Everything else uses 0.
- **Event-driven.** Nothing is recomputed globally. A changed block sends *shape updates* to its 6 neighbours, plus *block updates* in Java's order (west, east, down, up, north, south). A block only re-checks its power when it is updated. That is what makes budding and quasi-connectivity possible, and it leaves room for observers and comparators.
- **Strong vs weak power.** Levers strongly power the block they are attached to, torches the block above them, and repeaters the block in front. Strongly powered blocks feed dust. Dust only *weakly* powers the block under it and the blocks it points into, which switches torches, repeaters and lamps but not other dust.
- **Dust** takes 15 from a source and loses 1 per block. It connects sideways and one step up or down. A line points along itself, a lone dot acts as a cross, and a dust with one connection points straight through, so a lamp beside a dust line stays dark. When its level changes it updates 7 positions (42 block updates). Java does this in hash order; RedSpark uses a fixed order.
- **Repeaters** extend short pulses to their full delay and lock when a powered diode faces into their side. **Torches** burn out after 8 toggles in 60 gt and recover 160 gt later.
- **Comparators** read the block behind them (dust by its level) and their strongest side input. Only dust and the direct output of levers, torches, repeaters and comparators count as side inputs. Compare mode passes the rear strength unless a side is stronger, and subtract mode outputs rear minus side. They react after 2 gt, with priority -1 when facing into another diode, else 0. In the viewer, click a comparator to switch its mode. Training only places comparators when you pass `--comparators`.
- **Quasi-connectivity:** `Circuit.has_neighbor_signal_qc()` is ready for pistons, droppers and dispensers.
- `Circuit.settle()` runs until no ticks are pending. A circuit that never settles (a clock) fails its truth table.

Support: dust, repeaters, comparators and standing torches need a block beneath them, and wall torches need a block beside them, as in Java Edition. For now solid blocks and lamps count, plus the grid floor under the bottom layer. The rule is data (`SUPPORTS_BELOW` and `SUPPORTS_WALL_TORCH` in `redspark/sim.py`), so new support blocks only need adding there. Unsupported placements are refused, and a part whose support is removed or replaced breaks off, which can cascade up a tower. The training action mask and the Train tab both follow the same rule.

Simplifications: there are no observers, pistons or containers yet.

Circuits are stored as JSON:

```json
{"size": [5, 1, 3], "blocks": [
  {"pos": [0, 0, 1], "type": "lever", "facing": "+x", "label": "A"},
  {"pos": [1, 0, 1], "type": "solid"},
  {"pos": [2, 0, 1], "type": "torch", "facing": "-x"},
  {"pos": [3, 0, 1], "type": "dust"},
  {"pos": [4, 0, 1], "type": "lamp", "label": "OUT"}
]}
```

For a torch or lever, `facing` points at the block it is attached to (`-y` = the floor). For a repeater, it is the output direction. Hand-built examples (NOT, vertical NOT, OR, AND, repeater line, torch clock) come from `circuits/make_examples.py`.

## The training setup

`RedstoneGateEnv(gate="NOT")` starts with input levers `A` (and `B`) on the left edge and lamp `OUT` on the right.

- **Action:** place one block type (solid, dust, torch in 5 attachments, repeater in 4 facings, or clear) in one cell. `action_masks()` rules out illegal moves, such as a torch with nothing to attach to or overwriting the inputs.
- **Observation:** one-hot block type and facing per cell, plus a mask of the fixed input/output cells.
- **Reward:** the change in truth-table score after each placement, then a bonus of +1 for a full solve (which ends the episode) minus 0.01 per block used.
- **Gates:** `NOT`, `BUFFER`, `AND`, `OR`, `NAND`, `NOR`, `XOR` (see `redspark/tasks.py`).

Current results: NOT is solved reliably in about 40k steps (under a minute). The two-input gates are not solved yet with the default settings (OR reached 3/4 rows after 60k steps). That is the next thing to work on.

## Ideas for next steps

1. **Curriculum:** NOT, then BUFFER, then OR, then AND, then XOR, reusing the trained weights.
2. **A CNN policy** over the 3D grid instead of the flat MLP, so it learns "torch next to block" once instead of per cell.
3. **More training throughput:** `SubprocVecEnv`, or port `sim.py` to Rust (PyO3 for training, WASM so the browser can run it too).
4. **Watch training live:** have the server stream circuits from the current policy.
5. **Editing in the viewer:** place and remove blocks with the mouse, then save the circuit.
6. **Desktop app:** wrap the viewer with Tauri.

## Layout

```
redspark/
  sim.py        simulator
  tasks.py      truth tables and gate scoring
  env.py        Gymnasium environment
  train.py      MaskablePPO training script
  designer.py   truth-table tasks, scoring, and the in-app genetic trainers (layout search and network; NumPy only)
  server.py     Starlette + WebSocket backend (also serves the viewer)
  app.py        one-click launcher used by RedSpark.bat
  viewer/       built viewer (output of `npm run build` in web/)
RedSpark.bat    double-click to open the app on Windows
circuits/       example circuits (JSON) and the script that builds them
tests/          pytest suite for the simulator, env and server
web/            three.js viewer (Vite + TypeScript)
```
