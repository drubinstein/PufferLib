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
