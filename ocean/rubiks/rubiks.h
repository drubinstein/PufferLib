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
    float score;           // mean fraction of stickers matching their face centre (1.0=solved)
    float episode_return;  // sum of rewards over episode
    float episode_length;  // steps per episode
    float shuffle_depth;   // mean scramble depth (quarter-turns) per episode
    // scramble-depth distribution: fraction of episodes whose depth fell in each bucket
    float depth_1_7, depth_8_14, depth_15_21, depth_22_28, depth_29_35;
    // solved episodes per bucket; per-bucket solve rate = solved_X / depth_X
    float solved_1_7, solved_8_14, solved_15_21, solved_22_28, solved_29_35;
    float max_shuffles;    // mean per-env adaptive curriculum frontier (current max scramble depth)
    float n;               // REQUIRED last field (aggregation count)
} Log;

typedef struct { int face, row, col, dr, dc; } strip_t;

enum { U=0, D=1, R=2, L=3, F=4, B=5 };

typedef struct {
    Log log;                       // REQUIRED first field
    unsigned char* observations;   // REQUIRED. ByteTensor: 6*N*N color indices 0..5
    float* actions;                // REQUIRED. float on 4.0; cast to int in c_step
    float* rewards;                // REQUIRED
    float* terminals;              // REQUIRED (float on 4.0)
    int num_agents;                // REQUIRED for vecenv
    unsigned int rng;              // per-env seed (vecenv sets env->rng = env index before my_init)
    int cube_n;                    // == macro N; lets ported logic keep using env->cube_n
    int size;                      // 6*N*N
    int shuffles;                  // curriculum cap: target max scramble depth (e.g. 35)
    int curriculum_max;            // per-env adaptive frontier; depth ~ randint(1, curriculum_max)
    int scramble_depth;            // this episode's scramble depth (for logging)
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

int is_solved(Cube *env);  // forward decl (defined below); perf tracks solves, not reward sign

void add_log(Cube* env) {
    int solved = is_solved(env);   // solve rate (dense reward is ~always >0, so don't use reward sign)
    int d = env->scramble_depth;
    env->log.perf += solved ? 1 : 0;
    env->log.score += env->score;
    env->log.episode_length += env->tick;
    env->log.episode_return += env->episode_return;
    env->log.shuffle_depth += d;
    // scramble-depth distribution (fraction of episodes per bucket after aggregation)
    env->log.depth_1_7   += (d >= 1  && d <= 7 ) ? 1 : 0;
    env->log.depth_8_14  += (d >= 8  && d <= 14) ? 1 : 0;
    env->log.depth_15_21 += (d >= 15 && d <= 21) ? 1 : 0;
    env->log.depth_22_28 += (d >= 22 && d <= 28) ? 1 : 0;
    env->log.depth_29_35 += (d >= 29) ? 1 : 0;
    // solved episodes per bucket (per-bucket solve rate = solved_X / depth_X)
    env->log.solved_1_7   += (solved && d >= 1  && d <= 7 ) ? 1 : 0;
    env->log.solved_8_14  += (solved && d >= 8  && d <= 14) ? 1 : 0;
    env->log.solved_15_21 += (solved && d >= 15 && d <= 21) ? 1 : 0;
    env->log.solved_22_28 += (solved && d >= 22 && d <= 28) ? 1 : 0;
    env->log.solved_29_35 += (solved && d >= 29) ? 1 : 0;
    env->log.max_shuffles += env->curriculum_max;
    env->log.n++;
}

#define STICKER(env,f,r,c) ((env)->stickers[(f)*(env)->cube_n*(env)->cube_n + (r)*(env)->cube_n + (c)])
#define R_TMP(i,j) (env)->r_tmp[(i)*(env)->cube_n + (j)]

// Precompute strips that surround each face
void precompute_strips(Cube *env) {
    // N is the compile-time macro (#define N 3)
    // For each face looking at it moving clockwise. Strips on other faces that rotate
    // with the face. Order is for clockwise rotation. Describes how to start traversing the strip
    // {Face, starting row, starting col, direction row, direction col}
    // NB inconsistent use of directions here possibly better to use consistent schema but it works
    // for now so I don't want to break it!
    // FRONT (F):
    env->strips[F][0] = (strip_t){U, N-1, 0,   0,  1};
    env->strips[F][1] = (strip_t){R, 0,   0,   1,  0};
    env->strips[F][2] = (strip_t){D, 0,   N-1, 0, -1};
    env->strips[F][3] = (strip_t){L, N-1, N-1,-1,  0};
    // BACK (B):
    env->strips[B][0] = (strip_t){U, 0,    N-1, 0, -1};
    env->strips[B][1] = (strip_t){L, 0,    0,   1,  0};
    env->strips[B][2] = (strip_t){D, N-1,  0,   0,  1};
    env->strips[B][3] = (strip_t){R, N-1,  N-1, -1, 0};
    // UP face
    env->strips[U][0] = (strip_t){F, 0, 0,   0, +1};
    env->strips[U][1] = (strip_t){L, 0, 0,   0, +1};
    env->strips[U][2] = (strip_t){B, 0, 0,   0, +1};
    env->strips[U][3] = (strip_t){R, 0, 0,   0, +1};

    // DOWN (D):
    env->strips[D][0] = (strip_t){F, N-1, 0, 0, 1};
    env->strips[D][1] = (strip_t){L, N-1, 0, 0, 1};
    env->strips[D][2] = (strip_t){B, N-1, 0, 0, 1};
    env->strips[D][3] = (strip_t){R, N-1, 0, 0, 1};

    // RIGHT (R):
    env->strips[R][0] = (strip_t){U, 0,   N-1, 1, 0};
    env->strips[R][1] = (strip_t){B, N-1, 0,  -1,0};
    env->strips[R][2] = (strip_t){D, 0,   N-1, 1, 0};
    env->strips[R][3] = (strip_t){F, 0,   N-1, 1, 0};

    // LEFT (L):
    env->strips[L][0] = (strip_t){U, 0,   0, 1, 0};
    env->strips[L][1] = (strip_t){F, 0,   0, 1, 0};
    env->strips[L][2] = (strip_t){D, 0,   0, 1, 0};
    env->strips[L][3] = (strip_t){B, N-1, N-1,-1,0};
}

void reset_stickers(Cube* env) {
    for(int i = 0; i < 6; i++) {
        int col = i;
            for(int j = 0; j < env->cube_n; j++) {
               for(int k = 0; k < env->cube_n; k++) {
                   STICKER(env, i,j,k) = col;
               }
            }
    }
}

//Just rotates the strips CLOCKWISE, not the face itself
static void rotate_strips(Cube *env, strip_t s[4]) {
    //Copy last strip
    for (int k=0;k<N;k++)
        env->tmp[k] = STICKER(env, s[3].face, s[3].row + s[3].dr*k, s[3].col + s[3].dc*k);
    //Shift
    for (int j=3;j>0;j--) {
        for (int k=0;k<N;k++) {
            STICKER(env, s[j].face, s[j].row + s[j].dr*k, s[j].col + s[j].dc*k) =
                STICKER(env, s[j-1].face, s[j-1].row + s[j-1].dr*k, s[j-1].col + s[j-1].dc*k);
        }
    }
    //Copy last back
    for (int k=0;k<N;k++)
        STICKER(env, s[0].face, s[0].row + s[0].dr*k, s[0].col + s[0].dc*k) = env->tmp[k];
}

// Rotates the strips COUNTER-CLOCKWISE
static void rotate_strips_ccw(Cube *env, strip_t s[4]) {
    //Copy first strip
    for (int k=0;k<N;k++)
        env->tmp[k] = STICKER(env, s[0].face,s[0].row + s[0].dr*k,s[0].col + s[0].dc*k);
    //shift others
    for (int j=0;j<3;j++) {
        for (int k=0;k<N;k++) {
            STICKER(env, s[j].face,s[j].row + s[j].dr*k,s[j].col + s[j].dc*k) =
                STICKER(env, s[j+1].face,s[j+1].row + s[j+1].dr*k,s[j+1].col + s[j+1].dc*k);
        }
    }
    //Copy first
    for (int k=0;k<N;k++)
        STICKER(env, s[3].face,s[3].row + s[3].dr*k,s[3].col + s[3].dc*k) = env->tmp[k];
}

//Just rotates face stickers counter-clockwise
static void rotate_face_ccw(Cube *env, int f) {
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            R_TMP(N-1-j,i) = STICKER(env,f,i,j);
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            STICKER(env,f,i,j) = R_TMP(i,j);
}

//Just rotates the face stickers CLOCKWISE
static void rotate_face(Cube *env, int f) {
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            R_TMP(j,N-1-i) = STICKER(env,f,i,j);
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            STICKER(env,f,i,j) = R_TMP(i,j);
}

// Execute a face turn: rotate the adjacency strips and the face's own stickers.
// Supports multiple quarter-turns (turns 1-3); validated against pycuber for all
// faces, single + multi-turn, via the local cross-check tests.
void move(Cube *env, int face, int turns) {
    int dir = (turns > 0) ? +1 : -1;
    turns = abs(turns) % 4;
    // D's sticker grid is stored "viewed from above" like U (its near edge is row 0,
    // mirroring U), but a physical D turn is CW *from below* — the reverse matrix
    // direction. So D's face-sticker rotation is flipped relative to its strip
    // rotation; all other faces have matrix-CW aligned with physical-CW.
    int face_dir = (face == D) ? -dir : dir;
    for (int t=0; t<turns; t++) {
        if (dir > 0) {
            rotate_strips(env, env->strips[face]);
        } else {
            rotate_strips_ccw(env, env->strips[face]);
        }
        if (face_dir > 0) {
            rotate_face(env, face);
        } else {
            rotate_face_ccw(env, face);
        }
    }
}

static inline void decode_action(int action, int *face, int *turns) {
    *face = action / 2;
    *turns = (action % 2 == 0) ? +1 : -1;
}

//Distance from solved based on centre sticker as the colour for that face
//VERY rough heuristic
float score(Cube *env) {
    float temp_score = 1.0f;
    for (int f = 0; f < 6; f++) {
        int t_colour = f;
        int face_score = 0;
        for (int r = 0; r < env->cube_n; r++) {
            for (int c = 0; c < env->cube_n; c++) {
                if (STICKER(env, f, r, c) == t_colour)
                    face_score++;
            }
        }
        temp_score *= (float)face_score;
    }
    return temp_score;
}

//NB in this code we dont move centre stickers so face colour = centre sticker as in score
int is_solved(Cube *env) {
    for (int f = 0; f < 6; f++) {
        int color = f;
        for (int r = 0; r < env->cube_n; r++) {
            for (int c = 0; c < env->cube_n; c++) {
                if (STICKER(env, f, r, c) != color) {
                    return 0;
                }
            }
        }
    }
    return 1;
}

void shuffle(Cube* env, int shuffles) {
    // Each step is one random quarter-turn, so `shuffles` is the exact scramble
    // depth (curriculum knob): shuffles=1 -> a single quarter-turn from solved.
    for (int i = 0; i < shuffles; i++) {
        int face = rand_r(&env->rng) % 6;
        int turns = (rand_r(&env->rng) % 2) ? 1 : -1;  // single quarter-turn, random direction
        move(env, face, turns);
    }
}

// Dense reward: fraction of stickers whose colour matches their face's centre
// sticker (centres never move, so face f's centre stays colour f). 1.0 == solved.
float matching_fraction(Cube *env) {
    int total = 6 * env->cube_n * env->cube_n;
    int matched = 0;
    for (int f = 0; f < 6; f++) {
        int center = STICKER(env, f, env->cube_n / 2, env->cube_n / 2);
        for (int r = 0; r < env->cube_n; r++)
            for (int c = 0; c < env->cube_n; c++)
                if (STICKER(env, f, r, c) == center) matched++;
    }
    return (float)matched / (float)total;
}

void compute_observations(Cube* env) {
    memcpy(env->observations, env->stickers, 6 * env->cube_n * env->cube_n);
}

void init(Cube* env) {
    env->cube_n = N;
    env->size = 6 * N * N;
    env->curriculum_max = (env->shuffles > 0) ? 1 : 0;  // adaptive curriculum starts at depth 1
    env->render = 0;
    env->user_mode = 0;
    env->highlight_axis = 0;
    env->highlight_layer = 0;
    precompute_strips(env);
    // NOTE: do NOT touch env->rng here — vecenv seeded it before calling my_init.
}

void c_reset(Cube* env) {
    reset_stickers(env);
    // Curriculum: scramble by a random depth in [1, shuffles] quarter-turns each reset
    // (shuffles=0 leaves the cube solved, as the local tests rely on).
    env->scramble_depth = (env->curriculum_max > 0) ? (1 + rand_r(&env->rng) % env->curriculum_max) : 0;
    shuffle(env, env->scramble_depth);  // shuffle(.,0) is a no-op
    env->tick = 0;
    env->score = 0;
    env->episode_return = 0;
    compute_observations(env);
}

//Some debugging functions

void print_stickers_file(Cube* env, FILE *out) {
    for (int f=0; f<6; f++) {
        fprintf(out, "Face %d:\n", f);
        for (int r=0; r<env->cube_n; r++) {
            for (int c=0; c<env->cube_n; c++) {
                fprintf(out, "%d ", STICKER(env,f,r,c));
            }
            fprintf(out, "\n");
        }
        fprintf(out, "\n");
    }
}

void print_stickers(Cube* env) {
    print_stickers_file(env, stdout);
}

void print_strips(Cube *env) {
    const char *names[6] = {"U","D","R","L","F","B"};
    for (int f=0; f<6; f++) {
        printf("Face %s strips:\n", names[f]);
        for (int s=0; s<4; s++) {
            printf("  Strip %d: ", s);
            for (int k=0; k<env->cube_n; k++) {
                int r = env->strips[f][s].row + env->strips[f][s].dr * k;
                int c = env->strips[f][s].col + env->strips[f][s].dc * k;
                printf("(%d,%d,%d) ", env->strips[f][s].face, r, c);
            }
            printf("\n");
        }
    }
}


/* MAIN RENDERING CODE */

static Color sticker_colors[6];

void init_sticker_colors(void) {
    sticker_colors[0] = WHITE;
    sticker_colors[1] = YELLOW;
    sticker_colors[2] = RED;
    sticker_colors[3] = ORANGE;
    sticker_colors[4] = GREEN;
    sticker_colors[5] = BLUE;
}

//Holds info for the animation
typedef struct {
    int rotating;      // 0 idle, 1 anim
    float elapsed;
    float duration;    // e.g. 0.5f
    int axis;          // 0=X,1=Y,2=Z
    int layer;         // 0..N-1
    int dir;           // +1 or -1
} MoveState;

static MoveState anim = {0};

static inline Vector3 axis_vector(int axis) {
    return (axis==0)? (Vector3){1,0,0} :
           (axis==1)? (Vector3){0,1,0} :
                      (Vector3){0,0,1};
}

static inline int in_layer(Vector3 pos, int axis, int layer, int cube_size) {
    float half = (cube_size - 1) / 2.0f;
    // spacing must match cubelet spacing in c_render
    float spacing = 1.1f;
    int coord = (axis==0)? (int)roundf(pos.x/spacing + half) :
                (axis==1)? (int)roundf(pos.y/spacing + half) :
                           (int)roundf(pos.z/spacing + half);
    return coord == layer;
}

static void DrawQuad(Vector3 v1, Vector3 v2, Vector3 v3, Vector3 v4, Color color) {
    rlBegin(RL_QUADS);
        rlColor4ub(color.r, color.g, color.b, color.a);
        rlVertex3f(v1.x, v1.y, v1.z);
        rlVertex3f(v2.x, v2.y, v2.z);
        rlVertex3f(v3.x, v3.y, v3.z);
        rlVertex3f(v4.x, v4.y, v4.z);
    rlEnd();
}

void DrawCubelet(Vector3 pos, float size, Color faceColors[6]) {
    float h = size * 0.5f;

    // +X
    DrawQuad(
        (Vector3){pos.x+h, pos.y-h, pos.z+h},
        (Vector3){pos.x+h, pos.y-h, pos.z-h},
        (Vector3){pos.x+h, pos.y+h, pos.z-h},
        (Vector3){pos.x+h, pos.y+h, pos.z+h},
        faceColors[0]);

    // -X
    DrawQuad(
        (Vector3){pos.x-h, pos.y-h, pos.z-h},
        (Vector3){pos.x-h, pos.y-h, pos.z+h},
        (Vector3){pos.x-h, pos.y+h, pos.z+h},
        (Vector3){pos.x-h, pos.y+h, pos.z-h},
        faceColors[1]);

    // +Y
    DrawQuad(
        (Vector3){pos.x-h, pos.y+h, pos.z+h},
        (Vector3){pos.x+h, pos.y+h, pos.z+h},
        (Vector3){pos.x+h, pos.y+h, pos.z-h},
        (Vector3){pos.x-h, pos.y+h, pos.z-h},
        faceColors[2]);

    // -Y
    DrawQuad(
        (Vector3){pos.x-h, pos.y-h, pos.z-h},
        (Vector3){pos.x+h, pos.y-h, pos.z-h},
        (Vector3){pos.x+h, pos.y-h, pos.z+h},
        (Vector3){pos.x-h, pos.y-h, pos.z+h},
        faceColors[3]);

    // +Z
    DrawQuad(
        (Vector3){pos.x-h, pos.y-h, pos.z+h},
        (Vector3){pos.x+h, pos.y-h, pos.z+h},
        (Vector3){pos.x+h, pos.y+h, pos.z+h},
        (Vector3){pos.x-h, pos.y+h, pos.z+h},
        faceColors[4]);

    // -Z
    DrawQuad(
        (Vector3){pos.x+h, pos.y-h, pos.z-h},
        (Vector3){pos.x-h, pos.y-h, pos.z-h},
        (Vector3){pos.x-h, pos.y+h, pos.z-h},
        (Vector3){pos.x+h, pos.y+h, pos.z-h},
        faceColors[5]);
}

void c_render(Cube* env) {
    env->render = 1; //Important global window for anims so need to turn on for this env only
    static int initialized = 0;
    static Camera camera;
    float half = (env->cube_n - 1) / 2.0f;
    float spacing = 1.1f; //Needs to match 'in layer' code

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    if (!initialized) {
        if (!IsWindowReady()) {
            InitWindow(800, 600, "PufferLib Rubik's");
            SetTargetFPS(60);
            init_sticker_colors();
        }

        camera.position = (Vector3){10.0f,10.0f,10.0f};
        camera.target   = (Vector3){0.0f,0.0f,0.0f};
        camera.up       = (Vector3){0.0f,1.0f,0.0f};
        camera.fovy     = 45.0f;
        camera.projection = CAMERA_PERSPECTIVE;

        initialized = 1;
    }

    BeginDrawing();
    ClearBackground((Color){6,24,24,255});
    BeginMode3D(camera);
    UpdateCamera(&camera, CAMERA_THIRD_PERSON);

    //BACKGROUND
    float size = 20.0f;   // half-size of the room
    int steps = 20;       // subdivisions per wall
    float step = (2*size) / steps;
    Color cyan = (Color){0,255,255,255};

    // XY planes at z = ±size
    for (int i = 0; i <= steps; i++) {
        float x = -size + i*step;
        DrawLine3D((Vector3){x,-size,-size}, (Vector3){x,size,-size}, cyan);
        DrawLine3D((Vector3){x,-size, size}, (Vector3){x,size, size}, cyan);
    }
    for (int j = 0; j <= steps; j++) {
        float y = -size + j*step;
        DrawLine3D((Vector3){-size,y,-size}, (Vector3){ size,y,-size}, cyan);
        DrawLine3D((Vector3){-size,y, size}, (Vector3){ size,y, size}, cyan);
    }

    // XZ planes at y = ±size
    for (int i = 0; i <= steps; i++) {
        float x = -size + i*step;
        DrawLine3D((Vector3){x,-size,-size}, (Vector3){x,-size, size}, cyan);
        DrawLine3D((Vector3){x, size,-size}, (Vector3){x, size, size}, cyan);
    }
    for (int j = 0; j <= steps; j++) {
        float z = -size + j*step;
        DrawLine3D((Vector3){-size,-size,z}, (Vector3){ size,-size,z}, cyan);
        DrawLine3D((Vector3){-size, size,z}, (Vector3){ size, size,z}, cyan);
    }

    // YZ planes at x = ±size
    for (int i = 0; i <= steps; i++) {
        float y = -size + i*step;
        DrawLine3D((Vector3){-size,y,-size}, (Vector3){-size,y, size}, cyan);
        DrawLine3D((Vector3){ size,y,-size}, (Vector3){ size,y, size}, cyan);
    }
    for (int j = 0; j <= steps; j++) {
        float z = -size + j*step;
        DrawLine3D((Vector3){-size,-size,z}, (Vector3){-size, size,z}, cyan);
        DrawLine3D((Vector3){ size,-size,z}, (Vector3){ size, size,z}, cyan);
    }


    //CUBE
    for (int x=0; x<env->cube_n; x++) {
        for (int y=0; y<env->cube_n; y++) {
            for (int z=0; z<env->cube_n; z++) {
                Vector3 pos = (Vector3){
                    (x-half)*spacing,
                    (y-half)*spacing,
                    (z-half)*spacing
                };
                Color faces[6] = { BLACK, BLACK, BLACK, BLACK, BLACK, BLACK };
                // Right (+X)
                if (x == env->cube_n - 1)
                    faces[0] = sticker_colors[ STICKER(env, R, env->cube_n - 1 - y, env->cube_n - 1 - z) ];
                // Left (−X)
                if (x == 0)
                    faces[1] = sticker_colors[ STICKER(env, L, env->cube_n - 1 - y, z) ];
                // Up (+Y)
                if (y == env->cube_n - 1)
                    faces[2] = sticker_colors[ STICKER(env, U, z, x) ];
                // Down (−Y)
                if (y == 0)
                    faces[3] = sticker_colors[ STICKER(env, D, env->cube_n-1-z, x) ];
                // Front (+Z)
                if (z == env->cube_n - 1)
                    faces[4] = sticker_colors[ STICKER(env, F, env->cube_n - 1 - y, x) ];
                // Back (−Z)
                if (z == 0)
                    faces[5] = sticker_colors[ STICKER(env, B, env->cube_n - 1 - y, env->cube_n - 1 - x) ];
                rlPushMatrix();
                // rotate only the turning layer while animating
                if (anim.rotating && in_layer(pos, anim.axis, anim.layer, env->cube_n)) {
                    Vector3 axis = axis_vector(anim.axis);
                    rlRotatef(anim.dir * (anim.elapsed / anim.duration) * 90.0f,
                              axis.x, axis.y, axis.z);
                }
                rlTranslatef(pos.x, pos.y, pos.z);
                DrawCubelet((Vector3){0,0,0}, 1.0f, faces);
                rlPopMatrix();
            }
        }
    }
    //for highlights
    if (env->user_mode){

        rlDisableDepthTest();
        rlDisableBackfaceCulling();

        // change axis
        if (IsKeyPressed(KEY_UP))    env->highlight_axis = (env->highlight_axis + 1) % 3;
        if (IsKeyPressed(KEY_DOWN))  env->highlight_axis = (env->highlight_axis + 2) % 3;

        // toggle between external layers
        if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_LEFT)) {
            env->highlight_layer = (env->highlight_layer == 0) ? env->cube_n - 1 : 0;


}        // draw highlight
        float spacing = 1.1f;
        float half = (env->cube_n-1)/2.0f;
        float coord = (env->highlight_layer-half)*spacing;
        float extent = (env->cube_n*spacing)/2.0f + 0.1f;
        Color highlight = (Color){0,255,255,100}; // translucent yellow


        float thickness = spacing;  // slab thickness
        Vector3 pos = {0,0,0};
        float dx = 2*extent, dy = 2*extent, dz = 2*extent;

        if (env->highlight_axis == 0) {
            pos = (Vector3){coord, 0, 0};
            dx = thickness;   // thin along X
        }
        else if (env->highlight_axis == 1) {
            pos = (Vector3){0, coord, 0};
            dy = thickness;   // thin along Y
        }
        else {
            pos = (Vector3){0, 0, coord};
            dz = thickness;   // thin along Z
        }
        float hx = dx * 0.5f;
        float hy = dy * 0.5f;
        float hz = dz * 0.5f;

        // +X
        DrawQuad((Vector3){pos.x+hx, pos.y-hy, pos.z-hz},
                 (Vector3){pos.x+hx, pos.y-hy, pos.z+hz},
                 (Vector3){pos.x+hx, pos.y+hy, pos.z+hz},
                 (Vector3){pos.x+hx, pos.y+hy, pos.z-hz}, highlight);

        // -X
        DrawQuad((Vector3){pos.x-hx, pos.y-hy, pos.z+hz},
                 (Vector3){pos.x-hx, pos.y-hy, pos.z-hz},
                 (Vector3){pos.x-hx, pos.y+hy, pos.z-hz},
                 (Vector3){pos.x-hx, pos.y+hy, pos.z+hz}, highlight);

        // +Y
        DrawQuad((Vector3){pos.x-hx, pos.y+hy, pos.z-hz},
                 (Vector3){pos.x+hx, pos.y+hy, pos.z-hz},
                 (Vector3){pos.x+hx, pos.y+hy, pos.z+hz},
                 (Vector3){pos.x-hx, pos.y+hy, pos.z+hz}, highlight);

        // -Y
        DrawQuad((Vector3){pos.x-hx, pos.y-hy, pos.z+hz},
                 (Vector3){pos.x+hx, pos.y-hy, pos.z+hz},
                 (Vector3){pos.x+hx, pos.y-hy, pos.z-hz},
                 (Vector3){pos.x-hx, pos.y-hy, pos.z-hz}, highlight);

        // +Z
        DrawQuad((Vector3){pos.x-hx, pos.y-hy, pos.z+hz},
                 (Vector3){pos.x+hx, pos.y-hy, pos.z+hz},
                 (Vector3){pos.x+hx, pos.y+hy, pos.z+hz},
                 (Vector3){pos.x-hx, pos.y+hy, pos.z+hz}, highlight);

        // -Z
        DrawQuad((Vector3){pos.x+hx, pos.y-hy, pos.z-hz},
                 (Vector3){pos.x-hx, pos.y-hy, pos.z-hz},
                 (Vector3){pos.x-hx, pos.y+hy, pos.z-hz},
                 (Vector3){pos.x+hx, pos.y+hy, pos.z-hz}, highlight);

    rlEnableDepthTest();
    }
    EndMode3D();


    rlEnableBackfaceCulling();
    char buf[50];
    snprintf(buf, sizeof(buf), "Tick %d", env->tick);
    DrawText(buf, 10, 10, 20, WHITE);

    snprintf(buf, sizeof(buf), "Score %.2f", env->score);
    DrawText(buf, 10, 40, 20, WHITE);

    EndDrawing();
}

void c_step(Cube* env) {
    env->rewards[0] = 0;
    env->terminals[0] = 0;
    env->tick += 1;

    float match_before = matching_fraction(env);  // for potential-based (delta) reward

    int face, turns;
    decode_action((int)env->actions[0], &face, &turns);   // CHANGE vs PR: cast float action to int

    static const int FACE_AXIS[6]  = {1, 1, 0, 0, 2, 2};
    static const int FACE_LAYER[6] = {1, 0, 1, 0, 1, 0};
    static const int FACE_SIGN[6]  = {-1,-1,-1,+1,-1,+1};
    int dir = (turns > 0) ? +1 : -1;

    if (env->render) {
        anim.rotating = 1;
        anim.axis     = FACE_AXIS[face];
        anim.layer    = FACE_LAYER[face] ? env->cube_n - 1 : 0;
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

    // Potential-based (delta) reward: change in % stickers matching their face centre.
    // Telescopes to (final - initial) match over the episode, so episode length carries
    // no reward (prevents stalling); reaching solved (match=1.0) is optimal.
    float match_after = matching_fraction(env);
    env->rewards[0] = match_after - match_before;
    env->score = match_after;

    if (is_solved(env)) {
        env->terminals[0] = 1;
        env->rewards[0] = 1.0f;  // solve = max reward, capped at 1.0 (pufferlib clamps reward to [-1,1])
        // Adaptive curriculum: solving at the current frontier widens randint by 1 (cap = shuffles).
        if (env->scramble_depth >= env->curriculum_max && env->curriculum_max < env->shuffles)
            env->curriculum_max++;
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
