# Rubiks Env Port + Validation — Design

- **Date:** 2026-05-23
- **Branch:** `rubiks` (off `4.0` @ `6edc6c90`), worktree `PufferLib-rubiks`
- **Status:** Approved design, pending spec review

## Goal

Bring the Rubik's cube environment from PufferAI/PufferLib **PR #350** ("Rubiks" by
TBBristol, based on `3.0`) into the **`4.0`** branch as a native ocean env, and prove
it is correct with local validation tests before any training work.

## Scope

**This cycle (sub-project 1):**
1. Port the cube env to 4.0 conventions as a pure-C ocean env.
2. Emit a memory-optimal integer observation (the raw cube).
3. Validate correctness with **local, uncommitted** tests.

**Out of scope (deferred to cycle 2):** training runs, the embedding policy
(value + positional embeddings à la `2048-remix`), the custom embedding CUDA kernel,
and reward/score shaping. A forward-look is captured at the end.

## Background: why this is a port, not a cherry-pick

PR #350 targets the **3.0** structure; 4.0 differs materially:

| Concern | PR #350 (3.0) | 4.0 target |
|---|---|---|
| Env C location | `pufferlib/ocean/rubiks/` | `ocean/rubiks/` (top-level) |
| Binding API | `env_binding.h`, `PyObject* kwargs`, `unpack()` | `src/vecenv.h`, `Dict* kwargs`, `dict_get()` |
| Obs sizing | runtime `size` kwarg, Python allocates | compile-time `OBS_SIZE` (`vec->obs_size = OBS_SIZE`) |
| Python env class | `rubiks.py` (`Cube(PufferEnv)`) | none — envs are pure C, driven by `config/*.ini` |
| Config | `pufferlib/config/ocean/rubiks.ini` (3.0 keys) | `config/rubiks.ini` (4.0 keys) |
| `terminals` type | `unsigned char*` | `float*` |
| Observation | one-hot `Box(6,N,N,6)` float32 | **rejected** — see below |

The cleanest 4.0 template is **`ocean/g2048/`**: it already emits an integer grid as a
`ByteTensor` via `memcpy(observations, grid, …)`, which is exactly the shape of the
rubiks obs we want.

## Design

### 1. Observation & action spaces
- **Observation:** the raw cube — `6·N·N` `uint8` color indices (0–5), one byte per
  sticker. `OBS_TENSOR_T = ByteTensor`. No one-hot. This is the memory-optimal
  representation and the form the embedding policy will index directly.
- **Action:** `Discrete(12)` — 6 faces × {CW, CCW}.

### 2. Cube size N
4.0's vecenv requires obs size at compile time (`benchmark/binding.c`:
`// Current API forces you to edit this per obs size`). N is therefore a single
compile-time knob, with logic written fully general in N (idiomatic — cf.
`lightsout`: `OBS_SIZE (GRID_SIZE*GRID_SIZE)`, `hex`: `2*TOTAL_CELLS`):

```c
#define N 3
#define OBS_SIZE (6*N*N)
```

Changing N is a one-line edit + rebuild. (Other approaches — multiple compiled
variants, runtime max-bound padding, core vecenv changes — were considered and
rejected for this cycle.)

### 3. Env C module — `ocean/rubiks/` (modeled on `ocean/g2048/`)
- **`rubiks.h`** — `Cube` struct with 4.0 required fields:
  `Log log; unsigned char* observations; float* actions; float* rewards;
  float* terminals; int num_agents;` plus `unsigned char stickers[6*N*N]`, `int score`,
  `int tick`, `int max_episode_steps`, `int shuffles`, `unsigned int rng`.
  Port PR #350's cube logic: the 12 face moves, `shuffle(env, shuffles)`, `is_solved`,
  `score` (distance-from-solved by center color), `c_reset`, `c_step`. **Drop** the
  one-hot `OBS` macro.
- **Obs write:** `update_observations(env)` = `memcpy(observations, stickers, 6*N*N)`.
- **Reward/terminal (unchanged from PR for now):** `-1` per step; on solve, terminal +
  `+1`; truncate at `max_episode_steps`. `terminals`/`rewards` are `float*`.
- **`binding.c`** — 4.0 `src/vecenv.h` API:
  ```c
  #include "rubiks.h"
  #define OBS_SIZE (6*N*N)
  #define NUM_ATNS 1
  #define ACT_SIZES {12}
  #define OBS_TENSOR_T ByteTensor
  #define Env Cube
  #include "vecenv.h"
  void my_init(Env* env, Dict* kwargs) { /* shuffles, max_episode_steps, num_agents=1; init(env) */ }
  void my_log(Log* log, Dict* out) { /* perf, score, episode_return, episode_length */ }
  ```

### 4. Config — `config/rubiks.ini` (4.0 style)
Port PR #350's `[train]` hyperparameters into 4.0 sections. Baseline policy via
`[torch] encoder=DefaultEncoder, network=MinGRU, decoder=DefaultDecoder`. Start
`shuffles=1` (the only regime the PR author got to train).

### 5. Build & registration
Compile `ocean/rubiks/binding.c` → `pufferlib/ocean/rubiks/binding.*.so` via the
existing ocean build; confirm the env loads by name through the generic wrapper.
*Open item:* confirm the exact 4.0 build command and the Python entry point used to
instantiate an ocean env by name (needed by the tests).

### 6. Validation tests (local, uncommitted)
Per repo norm (maintainer dislikes committed tests), these stay local and are not
committed. A pytest module that:
- Asserts obs shape `(6,N,N)` `uint8`, action space `Discrete(12)`, and basic
  reset/step/terminal/reward behavior.
- **Cross-checks vs `pycuber`:** fix a mapping from our 12 actions and face/color
  layout to pycuber, drive both with identical random action sequences, assert the
  sticker state matches at every step.
- **Invariants:** `move⁴ = identity`; a scramble followed by the inverse sequence
  returns to solved; `is_solved` correctness; each color appears exactly `N·N` times;
  obs equals the internal sticker array.

## Risks / open items
- Exact 4.0 ocean build command and Python env-construction API (resolve at start of
  implementation).
- pycuber action/face/color mapping must be pinned with a couple of hand-checked moves
  before trusting the cross-check.
- `float*` terminals (vs the PR's `unsigned char*`) — adjust ported code accordingly.
- The PR committed build artifacts (`rubiks` binary, `.dSYM/`, `.png`s) — **not** ported.

## Forward-look (cycle 2 — separate spec)
- **Embedding policy:** `RubiksEmbed` encoder in `pufferlib/models.py`,
  `value_embed = Embedding(6)` (color), `pos_embed = Embedding(6·N·N)` (sticker
  position), summed → MLP (à la remix `G2048`); selected via `[torch]`.
- **Custom embedding kernel:** on PufferLib's C inference path, scatter the value-embed
  table to every warp and keep the pos-embed table resident before the eval phase to
  cut memory movement; iterate until it works.
- **Score iteration:** revisit the sparse `-1`/`+1` reward (shaping toward solved) once
  the env is validated, to make `shuffles>1` learnable.

## Decisions log
- Approach A (pure-C 4.0 env modeled on g2048). *(rejected: lift-and-shift w/ one-hot +
  rubiks.py; full rewrite)*
- Integer `6×N×N` `uint8` obs — the raw cube — not one-hot. *(user directive)*
- N = single compile-time `#define`, logic general in N. *(user directive, given the
  compile-time OBS_SIZE constraint)*
- Reward unchanged this cycle; iterate on score after validation. *(user directive)*
- Tests cross-check against pycuber; kept local/uncommitted. *(user directive)*
