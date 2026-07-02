/*
 * gen_test_img.cpp — generates simple BMP test images for test_batch.
 *
 * Creates 3 images in ./images/:
 *   1. test1.bmp  — red rectangle on white background
 *   2. test2.bmp  — green circle on black background
 *   3. test3.bmp  — blue gradient
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(p) _mkdir(p)
#else
#define MKDIR(p) mkdir(p, 0755)
#endif

#pragma pack(push, 1)
struct BMPHeader {
    uint16_t type = 0x4D42;
    uint32_t size;
    uint16_t reserved1 = 0;
    uint16_t reserved2 = 0;
    uint32_t offset = 54;
    uint32_t header_size = 40;
    int32_t  width;
    int32_t  height;
    uint16_t planes = 1;
    uint16_t bits = 24;
    uint32_t compression = 0;
    uint32_t image_size;
    int32_t  x_ppm = 2835;
    int32_t  y_ppm = 2835;
    uint32_t used = 0;
    uint32_t important = 0;
};
#pragma pack(pop)

void save_bmp(const char* path, uint8_t* rgb, int w, int h) {
    int pad = (4 - (w * 3) % 4) % 4;
    int row_size = w * 3 + pad;

    BMPHeader hdr;
    hdr.width  = w;
    hdr.height = h;
    hdr.image_size = row_size * h;
    hdr.size       = 54 + hdr.image_size;

    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }
    fwrite(&hdr, 54, 1, f);

    // BMP rows are bottom-to-top, BGR order
    for (int y = h - 1; y >= 0; y--) {
        const uint8_t* row = rgb + y * w * 3;
        fwrite(row, 3, w, f);
        uint8_t pad_bytes[3] = {0, 0, 0};
        fwrite(pad_bytes, 1, pad, f);
    }
    fclose(f);
    printf("Created: %s (%dx%d)\n", path, w, h);
}

void fill_white(uint8_t* img, int w, int h) {
    memset(img, 255, w * h * 3);
}

void fill_rect(uint8_t* img, int w, int h,
               int x, int y, int rw, int rh, uint8_t r, uint8_t g, uint8_t b) {
    for (int dy = y; dy < y + rh && dy < h; dy++)
        for (int dx = x; dx < x + rw && dx < w; dx++) {
            uint8_t* p = img + (dy * w + dx) * 3;
            p[0] = r; p[1] = g; p[2] = b;
        }
}

int main() {
    MKDIR("./images");
    MKDIR("./results");

    const int W = 640, H = 480;
    uint8_t* img = (uint8_t*)malloc(W * H * 3);

    // ---- test1.bmp: white background + red rectangle (class-0 style) ------
    fill_white(img, W, H);
    fill_rect(img, W, H, 200, 150, 250, 200, 255, 0, 0);  // red rect (RGB)
    save_bmp("./images/test1.bmp", img, W, H);

    // ---- test2.bmp: black bg + green circle area --------------------------
    memset(img, 0, W * H * 3);  // black
    int cx = 450, cy = 100, cr = 80;
    for (int dy = cy - cr; dy < cy + cr; dy++) {
        for (int dx = cx - cr; dx < cx + cr; dx++) {
            if (dx < 0 || dx >= W || dy < 0 || dy >= H) continue;
            if ((dx - cx) * (dx - cx) + (dy - cy) * (dy - cy) <= cr * cr) {
                uint8_t* p = img + (dy * W + dx) * 3;
                p[0] = 0; p[1] = 255; p[2] = 0;  // green (RGB)
            }
        }
    }
    save_bmp("./images/test2.bmp", img, W, H);

    // ---- test3.bmp: gradient (blue horizontal, green vertical) ------------
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            uint8_t* p = img + (y * W + x) * 3;
            p[0] = (uint8_t)(x * 255 / W);    // R = horizontal gradient
            p[1] = (uint8_t)(y * 255 / H);    // G = vertical gradient
            p[2] = 128;                        // B = fixed
        }
    save_bmp("./images/test3.bmp", img, W, H);

    free(img);
    printf("\nDone. Generated 3 test images in ./images/\n");
    return 0;
}
