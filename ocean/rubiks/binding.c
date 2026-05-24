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
    dict_set(out, "shuffle_depth", log->shuffle_depth);
    // scramble-depth distribution (fraction of episodes per bucket)
    dict_set(out, "depth_1_7", log->depth_1_7);
    dict_set(out, "depth_8_14", log->depth_8_14);
    dict_set(out, "depth_15_21", log->depth_15_21);
    dict_set(out, "depth_22_28", log->depth_22_28);
    dict_set(out, "depth_29_35", log->depth_29_35);
    // solved episodes per bucket (per-bucket solve rate = solved_X / depth_X)
    dict_set(out, "solved_1_7", log->solved_1_7);
    dict_set(out, "solved_8_14", log->solved_8_14);
    dict_set(out, "solved_15_21", log->solved_15_21);
    dict_set(out, "solved_22_28", log->solved_22_28);
    dict_set(out, "solved_29_35", log->solved_29_35);
    dict_set(out, "max_shuffles", log->max_shuffles);
}
