#pragma once
/*
 * Mock im2d.h — provides a software implementation of improcess that does
 * a simple nearest-neighbour resize with letterbox padding.
 *
 * This allows the visual output to be verified: the dst_buf is filled with
 * actual pixel data (instead of being left as dummy gray), so when mocked
 * RKNN outputs point to boxes, the saved result images display correctly.
 */

#include "rga.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int x;
    int y;
    int width;
    int height;
} im_rect;

// ---- wrapbuffer -----------------------------------------------------------

static rga_buffer_t wrapbuffer_virtualaddr(void* addr, int width, int height,
                                           int format) {
    rga_buffer_t buf;
    memset(&buf, 0, sizeof(buf));
    buf.vir_addr = addr;
    buf.width    = (uint32_t)width;
    buf.height   = (uint32_t)height;
    buf.wstride  = (uint32_t)width;
    buf.hstride  = (uint32_t)height;
    buf.format   = (uint32_t)format;
    return buf;
}

// ---- improcess (software resize + letterbox) ------------------------------

static IM_STATUS improcess(rga_buffer_t src, rga_buffer_t dst,
                           rga_buffer_t pat,
                           im_rect src_rect, im_rect dst_rect,
                           im_rect pat_rect, int mode) {
    (void)pat; (void)pat_rect; (void)mode;

    uint8_t* src_ptr = (uint8_t*)src.vir_addr;
    uint8_t* dst_ptr = (uint8_t*)dst.vir_addr;
    int src_w = src_rect.width;
    int src_h = src_rect.height;
    int dst_x = dst_rect.x;
    int dst_y = dst_rect.y;
    int new_w = dst_rect.width;
    int new_h = dst_rect.height;
    int dst_stride = (int)dst.width;       // dst buffer full width
    int bpp = 3;                           // RGB_888

    if (!src_ptr || !dst_ptr) return IM_STATUS_INVALID_PARAM;
    if (new_w <= 0 || new_h <= 0) return IM_STATUS_INVALID_PARAM;

    // Nearest-neighbour resize into the destination letterbox region
    for (int dy = 0; dy < new_h; dy++) {
        int sy = dy * src_h / new_h;
        if (sy >= src_h) sy = src_h - 1;
        uint8_t* drow = dst_ptr + ((dst_y + dy) * dst_stride + dst_x) * bpp;
        uint8_t* srow = src_ptr + sy * src_w * bpp;
        for (int dx = 0; dx < new_w; dx++) {
            int sx = dx * src_w / new_w;
            if (sx >= src_w) sx = src_w - 1;
            drow[dx * bpp + 0] = srow[sx * bpp + 0];
            drow[dx * bpp + 1] = srow[sx * bpp + 1];
            drow[dx * bpp + 2] = srow[sx * bpp + 2];
        }
    }

    return IM_STATUS_SUCCESS;
}

// ---- imStrError -----------------------------------------------------------

static const char* imStrError(IM_STATUS status) {
    switch (status) {
        case IM_STATUS_SUCCESS:       return "Success";
        case IM_STATUS_FAILED:        return "Failed";
        case IM_STATUS_INVALID_PARAM: return "Invalid parameter";
        case IM_STATUS_NOT_SUPPORTED: return "Not supported";
        case IM_STATUS_NOSPACE:       return "No space";
        default:                      return "Unknown error";
    }
}

#ifdef __cplusplus
}
#endif
