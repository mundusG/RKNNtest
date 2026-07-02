#pragma once
/*
 * Mock rga.h — Rockchip Graphics Acceleration types.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void*    vir_addr;
    void*    fd;
    uint32_t width;
    uint32_t height;
    uint32_t wstride;
    uint32_t hstride;
    uint32_t format;
} rga_buffer_t;

typedef enum {
    IM_STATUS_SUCCESS       =  0,
    IM_STATUS_FAILED        = -1,
    IM_STATUS_INVALID_PARAM = -2,
    IM_STATUS_NOT_SUPPORTED = -3,
    IM_STATUS_NOSPACE       = -4,
} IM_STATUS;

typedef enum {
    IM_SYNC  = 0,
    IM_ASYNC = 1,
} im_mode;

#ifdef __cplusplus
}
#endif
