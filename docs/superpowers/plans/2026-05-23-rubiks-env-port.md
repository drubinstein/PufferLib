# Rubiks Env Port + Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port PufferAI/PufferLib PR #350's Rubik's cube env into the `4.0` branch as a native pure-C ocean env that outputs the raw cube as a `6·N·N` uint8 observation, and prove it correct with local (uncommitted) tests.

**Architecture:** A header-only C env (`ocean/rubiks/rubiks.h`) modeled on `ocean/g2048/`, exposed to Python by statically linking into `pufferlib/_C` via `ocean/rubiks/binding.c` (the `src/vecenv.h` API) and `config/rubiks.ini`. Cube size `N` is a single compile-time `#define` (4.0's vecenv requires compile-time `OBS_SIZE`); logic is written general in `N`. Validation drives the built env through `_C.create_vec` and checks strong cube invariants plus a pycuber cross-check.

**Tech Stack:** C (clang), PufferLib 4.0 vecenv/build (`./build.sh`), Python (pytest, numpy, torch, pycuber).

**Reference:** PR #350's original 3.0 source is stashed (untracked) at `.port_ref/rubiks.h` and `.port_ref/rubiks.c` in this worktree. Cite line numbers below refer to `.port_ref/rubiks.h`.

**Environment note:** This is macOS (no CUDA), so the env is built with `./build.sh rubiks --cpu`. On a CUDA box use `./build.sh rubiks` (and `--float` for float32) instead. Tests are **local and uncommitted** (the maintainer dislikes committed tests) — see "Test hygiene" below.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `ocean/rubiks/rubiks.h` | Create | Header-only cube env: struct, moves, reset/step/reward/solve, obs, (optional) raylib render. |
| `ocean/rubiks/binding.c` | Create | 4.0 `vecenv.h` binding: obs/action sizes, `my_init`, `my_log`. |
| `ocean/rubiks/rubiks.c` | Create | Standalone `main()` for `--local`/`--fast` visual/smoke runs. |
| `config/rubiks.ini` | Create | 4.0-style config: `[base]/[vec]/[env]/[policy]/[train]`. |
| `.port_ref/rubiks_test.py` | Create (local, uncommitted) | pytest: build-driven invariants + pycuber cross-check. |
| `.port_ref/conftest.py` | Create (local, uncommitted) | pytest helper to build/create/step the env via `_C`. |

### Cube conventions being ported (from `.port_ref/rubiks.h`)
- Faces `enum { U=0, D=1, R=2, L=3, F=4, B=5 }`; sticker layout `STICKER(env,f,r,c) = stickers[f*N*N + r*N + c]`.
- Solved state: `STICKER(f,r,c) = f` for all r,c (so the obs of a solved cube is `[0]*9 + [1]*9 + ... + [5]*9` for N=3).
- Action `a ∈ [0,12)`: `face = a/2`, `turns = +1 if a even else -1` (CW / CCW). `decode_action` in `.port_ref/rubiks.h:278`.
- Centers never move; face color == its center == its index.

---

## Task 1: Create `ocean/rubiks/rubiks.h` (ported + adapted env)

**Files:**
- Create: `ocean/rubiks/rubiks.h`
- Reference: `.port_ref/rubiks.h`

This is the core deliverable. Build it by copying the PR header and applying the exact adaptations below. The adaptations exist because 4.0 differs from 3.0: `unsigned char*` observations (ByteTensor), `float*` actions/terminals, compile-time `N`, fixed-size arrays (no malloc), per-env reentrant RNG, and an integer (non-one-hot) observation.

- [ ] **Step 1: Write the header preamble + `#define N`**

```c
// Rubik's cube env for PufferLib 4.0. Ported from PR #350 (TBBristol).
// Logic inspired by https://github.com/Princeton-RL/CRTR (gym_rubik).
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "raylib.h"
#include "rlgl.h"

#define N 3   // Cube is NxNxN. Compile-time (4.0 vecenv requires compile-time OBS_SIZE). Logic is general in N.

typedef struct {
    float perf;            // 0-1: solved or not, per episode
    float score;           // distance-from-solved heuristic
    float episode_return;  // sum of rewards over episode
    float episode_length;  // steps per episode
    float n;               // REQUIRED last field (aggregation count)
} Log;

typedef struct { int face, row, col, dr, dc; } strip_t;

enum { U=0, D=1, R=2, L=3, F=4, B=5 };
```

- [ ] **Step 2: Write the `Cube` struct (4.0 required fields + fixed arrays + rng)**

```c
typedef struct {
    Log log;                       // REQUIRED first field
    unsigned char* observations;   // REQUIRED. ByteTensor: 6*N*N color indices 0..5
    float* actions;                // REQUIRED. float on 4.0; cast to int in c_step
    float* rewards;                // REQUIRED
    float* terminals;              // REQUIRED (float on 4.0)
    int num_agents;                // REQUIRED for vecenv
    unsigned int rng;              // per-env seed (vecenv sets env->rng = env index before my_init)
    int N;                         // == macro N; lets ported logic keep using env->N
    int size;                      // 6*N*N
    int shuffles;                  // scramble moves at reset
    int max_episode_steps;
    int tick;
    float score;
    float episode_return;
    float anim_time;               // render only
    int render;                    // render only (0 in training)
    int user_mode, highlight_layer, highlight_axis;  // render only
    strip_t strips[6][4];          // precomputed adjacency strips
    unsigned char stickers[6*N*N]; // the cube state (color indices)
    unsigned char tmp[N];          // rotation scratch
    unsigned char r_tmp[N*N];      // face-rotation scratch
} Cube;
```

- [ ] **Step 3: Write logging + state macros**

```c
void add_log(Cube* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->score;
    env->log.episode_length += env->tick;
    env->log.episode_return += env->episode_return;
    env->log.n++;
}

#define STICKER(env,f,r,c) ((env)->stickers[(f)*(env)->N*(env)->N + (r)*(env)->N + (c)])
#define R_TMP(i,j) (env)->r_tmp[(i)*(env)->N + (j)]
```

> Note vs PR: the one-hot `OBS(...)` macro and `set_color()` are **dropped**. The obs is the raw sticker array.

- [ ] **Step 4: Copy `precompute_strips`, `rotate_strips`, `rotate_strips_ccw`, `rotate_face_ccw`, `rotate_face`, `move`, `decode_action`, `score`, `is_solved`, `reset_stickers` VERBATIM from `.port_ref/rubiks.h`**

Copy these functions exactly as-is from `.port_ref/rubiks.h`:
- `precompute_strips` (lines 111–151)
- `reset_stickers` (lines 167–176)
- `rotate_strips` (197–212), `rotate_strips_ccw` (215–230)
- `rotate_face_ccw` (233–241), `rotate_face` (244–252)
- `move` (255–267)
- `decode_action` (278–281)
- `score` (285–299), `is_solved` (302–314)

They are unchanged: the `STICKER`/`R_TMP` macros now index `unsigned char` arrays, which is type-compatible (values are 0..5).

- [ ] **Step 5: Write the adapted `shuffle` (per-env reentrant RNG)**

```c
void shuffle(Cube* env, int shuffles) {
    for (int i = 0; i < shuffles; i++) {
        int face = rand_r(&env->rng) % 6;
        int turns = (rand_r(&env->rng) % 3) + 1;  // 1,2,3 quarter-turns
        move(env, face, turns);
    }
}
```

> Change vs PR (line 270): `rand()` → `rand_r(&env->rng)`. The PR's global `rand()` is not thread-safe under 4.0's OpenMP-parallel vecenv and would make every env scramble identically.

- [ ] **Step 6: Write the adapted `compute_observations` (memcpy, not one-hot)**

```c
void compute_observations(Cube* env) {
    memcpy(env->observations, env->stickers, 6 * env->N * env->N);
}
```

- [ ] **Step 7: Write the adapted `init` (no malloc; fixed arrays)**

```c
void init(Cube* env) {
    env->N = N;
    env->size = 6 * N * N;
    if (env->anim_time == 0) env->anim_time = 0.5f;
    env->render = 0;
    env->user_mode = 0;
    env->highlight_axis = 0;
    env->highlight_layer = 0;
    precompute_strips(env);
    // NOTE: do NOT touch env->rng here — vecenv seeded it before calling my_init.
}
```

> Change vs PR (lines 154–165): no `malloc` for stickers/tmp/r_tmp (now fixed arrays in the struct), and `env->N`/`env->size` are set from the compile-time `N`.

- [ ] **Step 8: Write the adapted `c_reset`**

```c
void c_reset(Cube* env) {
    reset_stickers(env);
    shuffle(env, env->shuffles);
    env->tick = 0;
    env->score = 0;
    env->episode_return = 0;
    compute_observations(env);
}
```

> Change vs PR (line 318): drop `memset(observations, 0, sizeof(float)*size)` — obs is fully written by `compute_observations`.

- [ ] **Step 9: Copy the render functions VERBATIM, then write the adapted `c_step` and `c_close`**

Copy VERBATIM from `.port_ref/rubiks.h` (render path; only runs in the standalone build when `env->render==1`):
- `init_sticker_colors` + `static Color sticker_colors[6]` (lines 70–79)
- `MoveState`/`anim` (83–92)
- `axis_vector`, `in_layer`, `DrawQuad`, `DrawCubelet`, `c_render` (368–656)
- (optional) `print_stickers`, `print_stickers_file`, `print_strips` (329–360) — handy for debugging.

Then write `c_step` adapted from PR (lines 659–714) with the action cast:

```c
void c_step(Cube* env) {
    env->rewards[0] = 0;
    env->terminals[0] = 0;
    env->tick += 1;

    int face, turns;
    decode_action((int)env->actions[0], &face, &turns);   // CHANGE vs PR: cast float action to int

    static const int FACE_AXIS[6]  = {1, 1, 0, 0, 2, 2};
    static const int FACE_LAYER[6] = {1, 0, 1, 0, 1, 0};
    static const int FACE_SIGN[6]  = {-1,-1,-1,+1,-1,+1};
    int dir = (turns > 0) ? +1 : -1;

    if (env->render) {
        anim.rotating = 1;
        anim.axis     = FACE_AXIS[face];
        anim.layer    = FACE_LAYER[face] ? env->N - 1 : 0;
        anim.dir      = FACE_SIGN[face] * dir;
        anim.elapsed  = 0.0f;
        anim.duration = env->anim_time;
        while (anim.elapsed < anim.duration) {
            if (WindowShouldClose()) break;
            anim.elapsed += GetFrameTime();
            c_render(env);
        }
        move(env, face, turns);
        anim.rotating = 0;
    } else {
        move(env, face, turns);
    }

    env->score = score(env);
    env->rewards[0] -= 1.0f;

    if (is_solved(env)) {
        env->terminals[0] = 1;
        env->rewards[0] = 1.0f;
        env->episode_return += env->rewards[0];
        add_log(env);
        c_reset(env);
        return;
    }
    if (env->tick >= env->max_episode_steps) {
        env->terminals[0] = 1;
        env->episode_return += env->rewards[0];
        add_log(env);
        c_reset(env);
        return;
    }
    env->episode_return += env->rewards[0];
    compute_observations(env);
}

void c_close(Cube* env) {
    if (IsWindowReady()) CloseWindow();
}
```

> Changes vs PR: `(int)env->actions[0]`; `c_close` no longer frees malloc'd arrays (they're fixed). The self-reset on terminal (`add_log` + `c_reset`) matches 4.0's vecenv convention (`cpu_vec_step` just calls `c_step`; the env resets itself).

- [ ] **Step 10: Commit**

```bash
git add ocean/rubiks/rubiks.h
git commit -m "feat(rubiks): port cube env header to 4.0 (integer obs, compile-time N)"
```

---

## Task 2: Create `ocean/rubiks/binding.c` (4.0 vecenv binding)

**Files:**
- Create: `ocean/rubiks/binding.c`

- [ ] **Step 1: Write the binding**

```c
#include "rubiks.h"

#define OBS_SIZE (6*N*N)
#define NUM_ATNS 1
#define ACT_SIZES {12}
#define OBS_TENSOR_T ByteTensor

#define Env Cube
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->shuffles = (int) dict_get(kwargs, "shuffles")->value;
    env->max_episode_steps = (int) dict_get(kwargs, "max_episode_steps")->value;
    env->anim_time = (float) dict_get(kwargs, "anim_time")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
```

> Mirrors `ocean/g2048/binding.c`. `dict_get` keys must match the `[env]` section of `config/rubiks.ini` (Task 4).

- [ ] **Step 2: Commit**

```bash
git add ocean/rubiks/binding.c
git commit -m "feat(rubiks): add 4.0 vecenv binding"
```

---

## Task 3: Create `ocean/rubiks/rubiks.c` (standalone main)

**Files:**
- Create: `ocean/rubiks/rubiks.c`

Needed only by `./build.sh rubiks --local|--fast|--web`. Modeled on `ocean/g2048/g2048.c` but with random actions (no trained net).

- [ ] **Step 1: Write the standalone main**

```c
#include "rubiks.h"

int main(void) {
    Cube env = {0};
    env.rng = 42;
    env.shuffles = 5;
    env.max_episode_steps = 300;
    env.anim_time = 0.3f;
    init(&env);

    unsigned char observations[6*N*N] = {0};
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    env.observations = observations;
    env.actions = actions;
    env.rewards = rewards;
    env.terminals = terminals;

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {            // user mode: press 0..9/a/b not wired; auto-play otherwise
            env.user_mode = 1;
        }
        env.actions[0] = (float)(rand_r(&env.rng) % 12);
        c_step(&env);
        c_render(&env);
    }
    c_close(&env);
    return 0;
}
```

- [ ] **Step 2: Build the standalone binary**

Run: `./build.sh rubiks --local`
Expected: `Built: ./rubiks` (compiles `ocean/rubiks/rubiks.c` with sanitizers). A window opens showing a scrambling cube; ESC/close to exit. This is a quick human smoke test that moves/render work.

- [ ] **Step 3: Commit**

```bash
git add ocean/rubiks/rubiks.c
git commit -m "feat(rubiks): add standalone visual/smoke main"
```

---

## Task 4: Create `config/rubiks.ini` and build the training backend

**Files:**
- Create: `config/rubiks.ini`

- [ ] **Step 1: Write the config (4.0 style, modeled on `config/g2048.ini`)**

```ini
[base]
env_name = rubiks

[vec]
total_agents = 4096
num_buffers = 2
num_threads = 0
seed = 42

[env]
shuffles = 1
max_episode_steps = 300
anim_time = 0.5

[policy]
hidden_size = 256
num_layers = 2
expansion_factor = 1

[train]
gpus = 1
seed = 42
total_timesteps = 100000000
learning_rate = 0.002
anneal_lr = 1
min_lr_ratio = 0
gamma = 0.99
gae_lambda = 0.7
replay_ratio = 2
clip_coef = 0.01
vf_coef = 0.1
vf_clip_coef = 0.1
max_grad_norm = 1.0
ent_coef = 0.02
beta1 = 0.9
beta2 = 0.99
eps = 1e-6
minibatch_size = 32768
horizon = 64
vtrace_rho_clip = 3.0
vtrace_c_clip = 3.0
prio_alpha = 1
prio_beta0 = 0.6
```

> `[env]` keys (`shuffles`, `max_episode_steps`, `anim_time`) must match the `dict_get` calls in `binding.c`. `[torch]` is inherited from `config/default.ini` (`encoder=DefaultEncoder, network=MinGRU, decoder=DefaultDecoder`) — baseline policy, no override this cycle. Any `[train]` keys not set here fall back to `config/default.ini`; if the build/run reports a missing key, copy it from `config/g2048.ini`.

- [ ] **Step 2: Build the CPU training backend with rubiks linked**

Run: `./build.sh rubiks --cpu`
Expected: `Built: pufferlib/_C.<ext>.so` with no compiler errors. (On a CUDA machine: `./build.sh rubiks` instead.)

- [ ] **Step 3: Verify the env loads and obs metadata is correct**

Run:
```bash
python -c "
import sys; sys.argv=['t']
import pufferlib.pufferl as pl
from pufferlib import _C
print('compiled env:', getattr(_C,'env_name',None))
args = pl.load_config('rubiks')
args['vec']['total_agents']=2; args['vec']['num_buffers']=1
vec = _C.create_vec(args, False)
print('obs_size', vec.obs_size, 'num_atns', vec.num_atns, 'obs_dtype', vec.obs_dtype, 'agents', vec.total_agents)
assert vec.obs_size == 6*3*3, vec.obs_size
"
```
Expected: `compiled env: rubiks`, `obs_size 54`, `num_atns 1`, an unsigned/byte `obs_dtype`, `agents 2`.

- [ ] **Step 4: Commit**

```bash
git add config/rubiks.ini
git commit -m "feat(rubiks): add 4.0 config; env builds and loads"
```

---

## Task 5: Local test harness (`conftest.py`)

**Files:**
- Create (local, uncommitted): `.port_ref/conftest.py`

**Test hygiene:** test files live under `.port_ref/` (already in git exclude) so they are never committed. Run them with `pytest .port_ref/ -v` (or `python -m pytest .port_ref/rubiks_test.py -v`). Do **not** `git add` anything under `.port_ref/`.

- [ ] **Step 1: Write the harness**

```python
# .port_ref/conftest.py  (LOCAL ONLY — never commit)
import sys
import numpy as np
import torch
import pytest

N = 3  # must match #define N in rubiks.h

def _make_vec(total_agents=1, env_overrides=None):
    sys.argv = ['pytest']  # load_config calls parse_args(); keep pytest argv out of it
    import pufferlib.pufferl as pl
    from pufferlib import _C
    assert getattr(_C, 'env_name', None) == 'rubiks', \
        "Run ./build.sh rubiks --cpu first (._C compiled for a different env)"
    args = pl.load_config('rubiks')
    args['vec']['total_agents'] = total_agents
    args['vec']['num_buffers'] = 1
    for k, v in (env_overrides or {}).items():
        args['env'][k] = v
    vec = _C.create_vec(args, False)
    return vec, _C

def _obs(vec):
    from pufferlib.torch_pufferl import _cpu_tensor
    t = _cpu_tensor(vec.obs_ptr, (vec.total_agents, vec.obs_size), torch.uint8)
    return np.asarray(t).reshape(vec.total_agents, 6, N, N).copy()

class Env:
    """Thin deterministic driver over one or more vec agents."""
    def __init__(self, total_agents=1, **env_overrides):
        self.vec, self._C = _make_vec(total_agents, env_overrides)
        self.n = total_agents
        self._act = torch.zeros((self.n, self.vec.num_atns), dtype=torch.float32)
        self.vec.reset()
    def obs(self):
        return _obs(self.vec)
    def rewards(self):
        from pufferlib.torch_pufferl import _cpu_tensor
        return np.asarray(_cpu_tensor(self.vec.rewards_ptr, (self.n,), torch.float32)).copy()
    def terminals(self):
        from pufferlib.torch_pufferl import _cpu_tensor
        return np.asarray(_cpu_tensor(self.vec.terminals_ptr, (self.n,), torch.float32)).copy()
    def step(self, actions):
        a = np.atleast_1d(np.asarray(actions, dtype=np.float32))
        self._act[:, 0] = torch.from_numpy(a)
        self._act = self._act.contiguous()
        self.vec.cpu_step(self._act.data_ptr())

@pytest.fixture
def solved_env():
    # shuffles=0 -> reset() yields a solved cube; huge max_steps so sequences don't truncate
    return Env(total_agents=1, shuffles=0, max_episode_steps=10_000)

# Inverse of action a: same face, opposite direction. a even=CW (a+1 is its CCW), a odd=CCW.
def inverse_action(a):
    return a + 1 if (a % 2 == 0) else a - 1
```

- [ ] **Step 2: Commit**

This file is uncommitted by design — **no commit**. Verify it's excluded:
Run: `git status --short` — Expected: `.port_ref/` does NOT appear.

---

## Task 6: Invariant tests (unconditional correctness)

**Files:**
- Create (local, uncommitted): `.port_ref/rubiks_test.py`

These hold for any correct cube regardless of color/orientation conventions, so they validate the port immediately without needing the pycuber mapping. Together they are very strong: color-counts catch sticker loss/duplication; "single move ≠ solved" rules out no-op moves; `move⁴=identity` and CW∘CCW=identity check rotation correctness; **scramble∘inverse=solved** exercises the whole move set and their inverses end-to-end.

- [ ] **Step 1: Write the failing tests**

```python
# .port_ref/rubiks_test.py  (LOCAL ONLY — never commit)
import numpy as np
import random
from conftest import Env, inverse_action, N

SOLVED = np.repeat(np.arange(6, dtype=np.uint8), N*N).reshape(6, N, N)

def test_obs_shape_and_solved_reset(solved_env):
    o = solved_env.obs()[0]
    assert o.shape == (6, N, N) and o.dtype == np.uint8
    assert np.array_equal(o, SOLVED), o

def test_color_counts_preserved_under_random_moves(solved_env):
    rng = random.Random(0)
    for _ in range(200):
        solved_env.step(rng.randrange(12))
        counts = np.bincount(solved_env.obs()[0].ravel(), minlength=6)
        assert (counts == N*N).all(), counts

def test_single_move_is_not_identity(solved_env):
    # A quarter-turn from solved must change the cube (rules out no-op moves).
    for a in range(12):
        e = Env(total_agents=1, shuffles=0, max_episode_steps=10_000)
        e.step(a)
        assert not np.array_equal(e.obs()[0], SOLVED), f"action {a} was a no-op"

def test_move_to_the_fourth_is_identity(solved_env):
    for a in range(0, 12, 2):  # CW moves
        e = Env(total_agents=1, shuffles=0, max_episode_steps=10_000)
        before = e.obs()[0].copy()
        for _ in range(4):
            e.step(a)
        assert np.array_equal(e.obs()[0], before), f"action {a}^4 != identity"

def test_cw_then_ccw_is_identity(solved_env):
    rng = random.Random(1)
    for _ in range(50):
        a = rng.randrange(0, 12, 2)
        before = solved_env.obs()[0].copy()
        solved_env.step(a)
        solved_env.step(inverse_action(a))
        assert np.array_equal(solved_env.obs()[0], before)

def test_scramble_then_inverse_returns_to_solved(solved_env):
    rng = random.Random(2)
    for _ in range(20):
        seq = [rng.randrange(12) for _ in range(30)]
        for a in seq:
            solved_env.step(a)
        for a in reversed(seq):
            solved_env.step(inverse_action(a))
        assert np.array_equal(solved_env.obs()[0], SOLVED)

def test_reward_and_terminal_on_solve(solved_env):
    # One move then its inverse solves -> that 2nd step should give reward +1 and terminal.
    a = 0
    solved_env.step(a)
    solved_env.step(inverse_action(a))
    assert solved_env.terminals()[0] == 1.0
    assert solved_env.rewards()[0] == 1.0

def test_step_cost_is_negative_one(solved_env):
    solved_env.step(0)  # from solved, one move is not solved
    assert solved_env.rewards()[0] == -1.0
    assert solved_env.terminals()[0] == 0.0
```

- [ ] **Step 2: Run and watch them drive out port bugs**

Run: `pytest .port_ref/rubiks_test.py -v`
Expected: all pass. If `test_scramble_then_inverse...` or `test_move_to_the_fourth...` fail, the move/strip logic was mis-ported (check Task 1 Step 4 copies and the `inverse_action`/`decode_action` parity). Fix `ocean/rubiks/rubiks.h`, rebuild (`./build.sh rubiks --cpu`), re-run. Iterate until green.

- [ ] **Step 3: Commit the env fixes (NOT the tests)**

If fixes to `rubiks.h` were needed:
```bash
git add ocean/rubiks/rubiks.h
git commit -m "fix(rubiks): correct cube mechanics found by local invariant tests"
```

---

## Task 7: pycuber cross-check (external ground truth)

**Files:**
- Modify (local, uncommitted): `.port_ref/rubiks_test.py`

Confirms our move *semantics* match a real 3×3 cube, not just self-consistency. pycuber is 3×3 only, so this task assumes `N==3`. The mapping (our action → pycuber move, our face/(r,c) → pycuber face/cell, our color index → pycuber colour) is **calibrated empirically**: pin it with single-move tests, then assert equality over random sequences.

- [ ] **Step 1: Install pycuber locally**

Run: `python -m pip install pycuber`
Expected: pycuber installed (local dev dependency; not added to the project).

- [ ] **Step 2: Write a calibration helper that learns the position+color mapping**

Add to `.port_ref/rubiks_test.py`. Strategy: build a bijection from our 54 obs cells to pycuber's 54 squares using only states both agree are "solved", plus the requirement that it is consistent across all 12 single moves. Concretely, learn it by labeling: drive our env and a pycuber cube through the SAME action sequence and require a single fixed cell-permutation `P` and color-map `C` such that `C(our_obs[P]) == pycuber_state` after every move.

```python
import pycuber as pc
import numpy as np, random, itertools
from conftest import Env, N

PYCUBER_FACES = ['U', 'D', 'R', 'L', 'F', 'B']  # tried first; calibration confirms/reorders

def pycuber_state(cube):
    # Flatten to a (6,3,3) array of colour strings in face order PYCUBER_FACES.
    faces = []
    for f in PYCUBER_FACES:
        face = cube.get_face(f)  # 3x3 of Square
        faces.append([[str(face[r][c].colour) for c in range(3)] for r in range(3)])
    return np.array(faces, dtype=object)

# Our action -> pycuber move string. CALIBRATE these in Step 3 (start from this guess).
ACTION_TO_PYCUBER = {
    0: "U", 1: "U'", 2: "D", 3: "D'", 4: "R", 5: "R'",
    6: "L", 7: "L'", 8: "F", 9: "F'", 10: "B", 11: "B'",
}
# NOTE: our enum is U,D,R,L,F,B with action a -> face a//2; this guess assumes that order.
```

- [ ] **Step 3: Write single-move calibration tests (pin the mapping)**

```python
def _our_after(seq):
    e = Env(total_agents=1, shuffles=0, max_episode_steps=10_000)
    for a in seq:
        e.step(a)
    return e.obs()[0]  # (6,3,3) uint8 color indices

def _pyc_after(seq):
    cube = pc.Cube()
    for a in seq:
        cube(ACTION_TO_PYCUBER[a])
    return pycuber_state(cube)

def test_calibrate_cellperm_and_colormap_exist():
    """A single fixed cell-permutation P and color bijection C must reproduce pycuber
    from our obs across the solved state AND all 12 single moves. Its existence proves
    our cube is isomorphic to pycuber's."""
    samples = [[]] + [[a] for a in range(12)] + [[a, b] for a in range(12) for b in range(12)][:24]
    ours = [_our_after(s).reshape(54) for s in samples]
    pycs = [_pyc_after(s).reshape(54) for s in samples]

    # For each pycuber cell j, find the set of our cells i that always share a consistent
    # color relationship; a valid P maps j->i uniquely and a global color bijection holds.
    # Build candidate i for each j: i is consistent if (ours[k][i]) has a 1:1 colour map to (pycs[k][j]) for all k.
    def consistent(i, j):
        m = {}
        for k in range(len(samples)):
            cv, pv = int(ours[k][i]), pycs[k][j]
            if cv in m and m[cv] != pv:
                return False
            m[cv] = pv
        return True

    P = [None]*54
    for j in range(54):
        cands = [i for i in range(54) if consistent(i, j)]
        assert cands, f"no consistent source cell for pycuber cell {j} — mapping/action guess wrong"
        P[j] = cands
    # Resolve to a unique bijection (each i used once). If this fails, ACTION_TO_PYCUBER is wrong.
    assert _resolve_bijection(P) is not None, "no consistent cell bijection; recalibrate ACTION_TO_PYCUBER"

def _resolve_bijection(P):
    # simple backtracking assignment j->i with each i unique
    used = [False]*54; assign=[None]*54
    order = sorted(range(54), key=lambda j: len(P[j]))
    def bt(t):
        if t == 54: return True
        j = order[t]
        for i in P[j]:
            if not used[i]:
                used[i]=True; assign[j]=i
                if bt(t+1): return True
                used[i]=False; assign[j]=None
        return False
    return assign if bt(0) else None
```

- [ ] **Step 4: Run calibration; fix `ACTION_TO_PYCUBER`/`PYCUBER_FACES` until it passes**

Run: `pytest .port_ref/rubiks_test.py::test_calibrate_cellperm_and_colormap_exist -v`
Expected: PASS. If it fails, our action→move guess or face order is off (e.g. our CW is pycuber's CCW for some face, or D/U orientation differs). Adjust `ACTION_TO_PYCUBER` (swap primes) and/or `PYCUBER_FACES`, re-run. A passing test means a single fixed isomorphism reproduces pycuber across solved + all single + sampled double moves — i.e. **our env is a real cube with the same move semantics**.

- [ ] **Step 5: Add a random-sequence cross-check using the resolved mapping**

```python
def test_random_sequences_match_pycuber():
    # Reuse the resolved bijection from the calibration set, then assert over fresh random seqs.
    base = [[]] + [[a] for a in range(12)] + [[a, b] for a in range(12) for b in range(12)][:24]
    ours = [_our_after(s).reshape(54) for s in base]
    pycs = [_pyc_after(s).reshape(54) for s in base]
    def consistent(i, j):
        m = {}
        for k in range(len(base)):
            cv, pv = int(ours[k][i]), pycs[k][j]
            if cv in m and m[cv] != pv: return False
            m[cv] = pv
        return True
    P = [[i for i in range(54) if consistent(i, j)] for j in range(54)]
    assign = _resolve_bijection(P)
    assert assign is not None
    # color map from any sample
    colmap = {}
    for j in range(54):
        colmap[int(ours[0][assign[j]])] = pycs[0][j]

    rng = random.Random(7)
    for _ in range(50):
        seq = [rng.randrange(12) for _ in range(25)]
        o = _our_after(seq).reshape(54)
        p = _pyc_after(seq).reshape(54)
        mapped = np.array([colmap[int(o[assign[j]])] for j in range(54)], dtype=object)
        assert np.array_equal(mapped, p), f"diverged from pycuber on seq {seq}"
```

- [ ] **Step 6: Run the full suite**

Run: `pytest .port_ref/ -v`
Expected: all invariant + pycuber tests pass. Fix `ocean/rubiks/rubiks.h` for any divergence (rebuild between fixes), keeping the tests uncommitted.

- [ ] **Step 7: Commit any env fixes (NOT the tests)**

```bash
git add ocean/rubiks/rubiks.h
git commit -m "fix(rubiks): align cube semantics with pycuber cross-check"
```

---

## Task 8: Final verification + summary

- [ ] **Step 1: Clean rebuild + full run**

Run:
```bash
./build.sh rubiks --cpu && pytest .port_ref/ -v
```
Expected: build succeeds; every test passes. Capture the output.

- [ ] **Step 2: Confirm git cleanliness (no test/artifact leakage)**

Run: `git status --short`
Expected: only `ocean/rubiks/{rubiks.h,binding.c,rubiks.c}`, `config/rubiks.ini`, and `docs/superpowers/**` are tracked changes; `.port_ref/`, the `rubiks` binary, and `pufferlib/_C*.so` do NOT appear as staged. (Add `rubiks` standalone binary and `_C` artifacts to git exclude if they show up.)

- [ ] **Step 3: Report results**

Summarize: env builds, obs is `(6,3,3)` uint8, all invariants hold, pycuber cross-check passes. The env is validated and ready for cycle 2 (training → embedding policy → custom kernel → reward/score iteration).

---

## Notes for cycle 2 (not in this plan)
- Embedding policy `RubiksEmbed` in `pufferlib/models.py`: `value_embed=Embedding(6)`, `pos_embed=Embedding(6*N*N)`, summed → MLP (à la `2048-remix` `G2048`), selected via `[torch] encoder=RubiksEmbed`.
- Custom embedding kernel on the C inference path (scatter value-embed to warps, keep pos-embed resident before eval).
- Reward/score iteration to make `shuffles>1` learnable.
