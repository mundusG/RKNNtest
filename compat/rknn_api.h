#pragma once
/*
 * Mock rknn_api.h — provides stub types and functions so test_batch.cpp
 * compiles and runs on a PC without Rockchip hardware.
 *
 * rknn_run / rknn_outputs_get return fabricated feature-map data that
 * decodes to a few fake bounding boxes, exercising the full post-processing
 * pipeline (decode, NMS, draw, save).
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---- Types ----------------------------------------------------------------

typedef void* rknn_context;

typedef enum {
    RKNN_TENSOR_UINT8 = 0,
    RKNN_TENSOR_FLOAT32 = 1,
} rknn_tensor_type;

typedef enum {
    RKNN_TENSOR_NHWC = 0,
    RKNN_TENSOR_NCHW = 1,
} rknn_tensor_format;

typedef enum {
    RK_FORMAT_RGB_888 = 0,
    RK_FORMAT_BGR_888 = 1,
} rknn_input_format;

typedef struct {
    uint32_t index;
    rknn_tensor_type type;
    uint32_t size;
    uint8_t  pass_through;
    rknn_tensor_format fmt;
    void*    buf;
} rknn_input;

typedef struct {
    uint8_t  want_float;
    uint8_t  is_prealloc;
    uint32_t index;
    void*    buf;
    uint32_t size;
} rknn_output;

typedef struct {
    uint32_t n_input;
    uint32_t n_output;
} rknn_input_output_num;

typedef enum {
    RKNN_QUERY_IN_OUT_NUM  = 0,
    RKNN_QUERY_INPUT_ATTR  = 1,
    RKNN_QUERY_OUTPUT_ATTR = 2,
} rknn_query_cmd;

typedef struct {
    uint32_t index;
    uint8_t  pass_through;
    rknn_tensor_type type;
    rknn_tensor_format fmt;
    char     name[64];
    uint32_t n_dims;
    uint32_t dims[4];
} rknn_tensor_attr;

// ---- Global fake-output storage -------------------------------------------
// These are filled by rknn_inputs_set / rknn_run and returned by
// rknn_outputs_get so the post-processing code has data to work on.

// 3 output tensors sizes: 80×80×21, 40×40×21, 20×20×21 (21 = 3 anchors × 7)
#define OUT0_H 80
#define OUT0_W 80
#define OUT1_H 40
#define OUT1_W 40
#define OUT2_H 20
#define OUT2_W 20

// Per-anchor-per-cell stride: 5 + num_classes = 7
#define OUT_CH 7
#define OUT_ANCHORS 3

static float  g_fake_out0[OUT0_H * OUT0_W * OUT_CH * OUT_ANCHORS];
static float  g_fake_out1[OUT1_H * OUT1_W * OUT_CH * OUT_ANCHORS];
static float  g_fake_out2[OUT2_H * OUT2_W * OUT_CH * OUT_ANCHORS];
static int    g_fake_initialised = 0;

// ---- Helpers to write one detection into a feature map --------------------
static void set_detection(float* feat, int grid_h, int grid_w,
                          int anchor_idx, int gy, int gx,
                          float bx, float by, float bw, float bh,
                          float conf, float cls0, float cls1) {
    int grid_size  = grid_h * grid_w;
    int anchor_off = anchor_idx * (OUT_CH * grid_size);
    int cell       = gy * grid_w + gx;

    feat[anchor_off + cell + 0 * grid_size] = bx;
    feat[anchor_off + cell + 1 * grid_size] = by;
    feat[anchor_off + cell + 2 * grid_size] = bw;
    feat[anchor_off + cell + 3 * grid_size] = bh;
    feat[anchor_off + cell + 4 * grid_size] = conf;
    feat[anchor_off + cell + 5 * grid_size] = cls0;
    feat[anchor_off + cell + 6 * grid_size] = cls1;
}

static void build_fake_outputs(void) {
    if (g_fake_initialised) return;
    g_fake_initialised = 1;

    // Zero everything first
    memset(g_fake_out0, 0, sizeof(g_fake_out0));
    memset(g_fake_out1, 0, sizeof(g_fake_out1));
    memset(g_fake_out2, 0, sizeof(g_fake_out2));

    // --- Stride-8  (80×80) : one class-0 box near centre -------------------
    // bx=0.5, by=0.5  → centre of image
    // bw=0.3, bh=0.3  → ~192×192 px on a 640×640 input
    set_detection(g_fake_out0, OUT0_H, OUT0_W, /*anchor*/0, /*gy*/40, /*gx*/40,
                  0.5f, 0.5f,        // bx, by
                  0.3f, 0.3f,        // bw, bh
                  0.90f,              // conf (objectness)
                  0.95f, 0.10f);     // cls0, cls1 → class 0 wins

    // --- Stride-16 (40×40) : one class-1 box upper-right -------------------
    set_detection(g_fake_out1, OUT1_H, OUT1_W, /*anchor*/1, /*gy*/8, /*gx*/30,
                  0.72f, 0.28f,      // bx, by
                  0.15f, 0.22f,      // bw, bh
                  0.85f,              // conf
                  0.08f, 0.92f);     // cls0, cls1 → class 1 wins

    // --- Stride-32 (20×20) : another class-0 box lower-left ----------------
    set_detection(g_fake_out2, OUT2_H, OUT2_W, /*anchor*/0, /*gy*/14, /*gx*/5,
                  0.22f, 0.72f,      // bx, by
                  0.12f, 0.16f,      // bw, bh
                  0.78f,              // conf
                  0.88f, 0.35f);     // cls0, cls1 → class 0 wins
}

// ---- API stubs ------------------------------------------------------------

static int rknn_init(rknn_context* ctx, void* model_data, uint32_t size,
                     uint32_t flag) {
    (void)model_data; (void)size; (void)flag;
    *ctx = (rknn_context)(uintptr_t)0x1;  // non-null dummy
    build_fake_outputs();
    return 0;  // success
}

static int rknn_query(rknn_context ctx, rknn_query_cmd cmd, void* info,
                      uint32_t size) {
    (void)ctx; (void)size;
    if (cmd == RKNN_QUERY_IN_OUT_NUM) {
        rknn_input_output_num* num = (rknn_input_output_num*)info;
        num->n_input  = 1;
        num->n_output = 3;
    } else if (cmd == RKNN_QUERY_INPUT_ATTR) {
        rknn_tensor_attr* attr = (rknn_tensor_attr*)info;
        if (attr->index == 0) {
            snprintf(attr->name, sizeof(attr->name), "input");
            attr->type    = RKNN_TENSOR_UINT8;
            attr->fmt     = RKNN_TENSOR_NHWC;
            attr->n_dims  = 4;
            attr->dims[0] = 1;
            attr->dims[1] = 640;
            attr->dims[2] = 640;
            attr->dims[3] = 3;
        }
    } else if (cmd == RKNN_QUERY_OUTPUT_ATTR) {
        rknn_tensor_attr* attr = (rknn_tensor_attr*)info;
        uint32_t h = (attr->index == 0) ? 80 : (attr->index == 1) ? 40 : 20;
        uint32_t w = h;
        snprintf(attr->name, sizeof(attr->name), "output_%u", attr->index);
        attr->type    = RKNN_TENSOR_FLOAT32;
        attr->fmt     = RKNN_TENSOR_NCHW;
        attr->n_dims  = 4;
        attr->dims[0] = 1;
        attr->dims[1] = OUT_CH * OUT_ANCHORS;  // 21
        attr->dims[2] = h;
        attr->dims[3] = w;
    }
    return 0;
}

static int rknn_inputs_set(rknn_context ctx, uint32_t n_inputs,
                           rknn_input inputs[]) {
    (void)ctx; (void)n_inputs; (void)inputs;
    return 0;
}

static int rknn_run(rknn_context ctx, void* ext) {
    (void)ctx; (void)ext;
    return 0;
}

static int rknn_outputs_get(rknn_context ctx, uint32_t n_outputs,
                            rknn_output outputs[], void* ext) {
    (void)ctx; (void)ext;
    // Hand out pointers to our pre-built fake feature maps
    for (uint32_t i = 0; i < n_outputs; i++) {
        uint32_t idx = outputs[i].index;
        size_t   sz  = 0;
        void*    ptr = NULL;
        if (idx == 0) { ptr = g_fake_out0; sz = sizeof(g_fake_out0); }
        if (idx == 1) { ptr = g_fake_out1; sz = sizeof(g_fake_out1); }
        if (idx == 2) { ptr = g_fake_out2; sz = sizeof(g_fake_out2); }
        if (outputs[i].want_float) {
            outputs[i].buf  = ptr;
            outputs[i].size = (uint32_t)sz;
        }
    }
    return 0;
}

static int rknn_outputs_release(rknn_context ctx, uint32_t n_outputs,
                                rknn_output outputs[]) {
    (void)ctx; (void)n_outputs; (void)outputs;
    return 0;
}

static int rknn_destroy(rknn_context ctx) {
    (void)ctx;
    return 0;
}

#ifdef __cplusplus
}
#endif
