#include "rubiks.h"

#if OBS_ONEHOT
#define OBS_SIZE (6*N*N*6)
#else
#define OBS_SIZE (6*N*N)
#endif
#define NUM_ATNS 1
#define ACT_SIZES {NUM_ACTIONS}  // 18: HTM action space (see rubiks.h)
#define OBS_TENSOR_T ByteTensor

#define Env Cube
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->shuffles = (int) dict_get(kwargs, "shuffles")->value;
    env->max_episode_steps = (int) dict_get(kwargs, "max_episode_steps")->value;
    env->depth_exponent = (float) dict_get(kwargs, "depth_exponent")->value;
    env->advance_threshold = (float) dict_get(kwargs, "advance_threshold")->value;
    env->advance_window = (int) dict_get(kwargs, "advance_window")->value;
    env->reward_shaping = (float) dict_get(kwargs, "reward_shaping")->value;
    env->anim_time = (float) dict_get(kwargs, "anim_time")->value;
    env->fixed_depth = (int) dict_get(kwargs, "fixed_depth")->value;
    env->level_mode = (int) dict_get(kwargs, "level_mode")->value;
    env->level_linear = (int) dict_get(kwargs, "level_linear")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "shuffle_depth", log->shuffle_depth);
    // Per-depth histogram: depth_0..depth_35 / solved_0..solved_35. dict_set stores the key
    // POINTER (no copy), so the key strings must persist -> build static buffers once.
    static char depth_keys[LOG_DEPTHS][12];
    static char solved_keys[LOG_DEPTHS][12];
    static int keys_init = 0;
    if (!keys_init) {
        for (int d = 0; d < LOG_DEPTHS; d++) {
            snprintf(depth_keys[d], sizeof(depth_keys[d]), "depth_%d", d);
            snprintf(solved_keys[d], sizeof(solved_keys[d]), "solved_%d", d);
        }
        keys_init = 1;
    }
    for (int d = 0; d < LOG_DEPTHS; d++) {
        dict_set(out, depth_keys[d], log->depth_hist[d]);
        dict_set(out, solved_keys[d], log->solved_hist[d]);
    }
    dict_set(out, "max_shuffles", log->max_shuffles);
    dict_set(out, "max_shuffles_qtm", log->max_shuffles_qtm);
}
