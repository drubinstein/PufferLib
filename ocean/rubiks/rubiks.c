#include "rubiks.h"

int main(void) {
    Cube env = {0};
    env.rng = 42;
    env.shuffles = 5;
    env.max_episode_steps = 300;
    env.anim_time = 0.3f;
    init(&env);

#if OBS_ONEHOT
    unsigned char observations[6*N*N*6] = {0};   // one-hot: compute_observations writes 6*N*N*6 bytes
#else
    unsigned char observations[6*N*N] = {0};      // integer colour indices
#endif
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
        env.actions[0] = (float)(rand_r(&env.rng) % NUM_ACTIONS);
        c_step(&env);
        c_render(&env);
    }
    c_close(&env);
    return 0;
}
