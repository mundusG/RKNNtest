#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <unordered_set>
#include <chrono>
#include <dirent.h>
#include <sys/stat.h>

#include <rga.h>
#include <RgaUtils.h>
// #include <dma_alloc.h>
#include "im2d.h"

// #include "utils.h"
#include <rknn_api.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define DEFAULT_MODEL_PATH  "climb0325_2.rv1126.rknn"
#define DEFAULT_IMAGE_DIR   "./images"
#define DEFAULT_OUTPUT_DIR  "./results"
#define ALIGN_16(x)      (((x) + 15) & ~15)
#define ALIGN_32(x)      (((x) + 31) & ~31)

// ===================== 数据结构 =====================

struct Box {
    float x1, y1, x2, y2;
    float conf;
    int class_id;
};

// ===================== 图像读写 (stb_image 全格式支持) =====================

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ===================== 画框（BGR 顺序）=====================

void draw_rect(uint8_t* img, int width, int height, int channels,
               int x1, int y1, int x2, int y2,
               uint8_t c0, uint8_t c1, uint8_t c2, int thickness = 3) {
    x1 = std::max(0, x1);        y1 = std::max(0, y1);
    x2 = std::min(width-1, x2);  y2 = std::min(height-1, y2);
    if (x1 >= x2 || y1 >= y2) return;

    for (int t = 0; t < thickness; t++) {
        for (int x = x1; x <= x2; x++) {
            if (y1+t < height) {
                uint8_t* p = img + ((y1+t)*width + x)*channels;
                p[0]=c0; p[1]=c1; p[2]=c2;
            }
            if (y2-t >= 0) {
                uint8_t* p = img + ((y2-t)*width + x)*channels;
                p[0]=c0; p[1]=c1; p[2]=c2;
            }
        }
        for (int y = y1; y <= y2; y++) {
            if (x1+t < width) {
                uint8_t* p = img + (y*width + x1+t)*channels;
                p[0]=c0; p[1]=c1; p[2]=c2;
            }
            if (x2-t >= 0) {
                uint8_t* p = img + (y*width + x2-t)*channels;
                p[0]=c0; p[1]=c1; p[2]=c2;
            }
        }
    }
}

// ===================== 后处理 =====================

std::vector<int> nms_boxes(std::vector<Box> boxes, float nms_thresh) {
    int n = boxes.size();
    if (n == 0) return {};
    std::vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i;
    sort(order.begin(), order.end(), [&](int a, int b){
        return boxes[a].conf > boxes[b].conf;
    });
    std::vector<float> area(n);
    for (int i = 0; i < n; i++)
        area[i] = (boxes[i].x2-boxes[i].x1)*(boxes[i].y2-boxes[i].y1);
    std::vector<int> keep;
    std::vector<bool> removed(n, false);
    for (int i = 0; i < n; i++) {
        int idx = order[i];
        if (removed[idx]) continue;
        keep.push_back(idx);
        for (int j = i+1; j < n; j++) {
            int jdx = order[j];
            if (removed[jdx]) continue;
            float xx1 = std::max(boxes[idx].x1, boxes[jdx].x1);
            float yy1 = std::max(boxes[idx].y1, boxes[jdx].y1);
            float xx2 = std::min(boxes[idx].x2, boxes[jdx].x2);
            float yy2 = std::min(boxes[idx].y2, boxes[jdx].y2);
            float inter = std::max(0.0f,xx2-xx1+1e-5f)*std::max(0.0f,yy2-yy1+1e-5f);
            float ovr = inter / (area[idx]+area[jdx]-inter);
            if (ovr > nms_thresh) removed[jdx] = true;
        }
    }
    return keep;
}

std::vector<Box> process_feature(
    const float* feat, int grid_h, int grid_w,
    const std::vector<std::pair<float,float>>& anchors,
    int stride, int num_anchors, int num_classes, float obj_thresh,
    int src_width, int src_height, int pad_w, int pad_h, float scale
) {
    std::vector<Box> boxes;
    int grid_size  = grid_h * grid_w;
    int anchor_size = (5 + num_classes) * grid_size;

    for (int a = 0; a < num_anchors; a++) {
        // 获取当前网格的坐标
        for (int row = 0; row < grid_h; row++) {
            for (int col = 0; col < grid_w; col++) {
                int b = row * grid_w + col;
                int base = a * anchor_size + b;
                
                float bx   = feat[base];
                float by   = feat[base + grid_size];
                float bw   = feat[base + 2*grid_size];
                float bh   = feat[base + 3*grid_size];
                float conf = feat[base + 4*grid_size];
                
                if (conf > obj_thresh) {
                    float max_cls_prob = 0.0f;
                    int max_cls_id = -1;
                    for (int c = 0; c < num_classes; c++) {
                        float cls_prob = feat[base + (5 + c) * grid_size];
                        if (cls_prob > max_cls_prob) {
                            max_cls_prob = cls_prob;
                            max_cls_id = c;
                        }
                    }

                    float score = conf * max_cls_prob;
                    if (score > obj_thresh) { 
                        Box box;
                        // 加上网格偏移，乘以步长和锚框
                        float cx = (bx * 2.0f - 0.5f + col) * stride;
                        float cy = (by * 2.0f - 0.5f + row) * stride;
                        float w  = pow(bw * 2.0f, 2.0f) * anchors[a].first;
                        float h  = pow(bh * 2.0f, 2.0f) * anchors[a].second;

                        // 中心点转为左上右下
                        float x1 = cx - w / 2.0f;
                        float y1 = cy - h / 2.0f;
                        float x2 = cx + w / 2.0f;
                        float y2 = cy + h / 2.0f;
                        
                        // 还原到原图尺寸（去除 Letterbox 的 padding 和缩放）
                        box.x1 = std::max(0.0f, (x1-pad_w)/scale);
                        box.y1 = std::max(0.0f, (y1-pad_h)/scale);
                        box.x2 = std::min((float)src_width,  (x2-pad_w)/scale);
                        box.y2 = std::min((float)src_height, (y2-pad_h)/scale);
                        box.conf     = score;
                        box.class_id = max_cls_id;
                        boxes.push_back(box);
                    }
                }
            }
        }
    }
    return boxes;
}

std::vector<Box> post_process(
    const float* out0, int h0, int w0,
    const float* out1, int h1, int w1,
    const float* out2, int h2, int w2,
    const std::vector<std::pair<float,float>>& anchors0,
    const std::vector<std::pair<float,float>>& anchors1,
    const std::vector<std::pair<float,float>>& anchors2,
    int num_classes, float obj_thresh, float nms_thresh,
    int src_width, int src_height, int pad_w, int pad_h, float scale
) {
    std::vector<Box> boxes;
    auto b0 = process_feature(out0,h0,w0,anchors0, 8,(int)anchors0.size(),num_classes,obj_thresh,src_width,src_height,pad_w,pad_h,scale);
    auto b1 = process_feature(out1,h1,w1,anchors1,16,(int)anchors1.size(),num_classes,obj_thresh,src_width,src_height,pad_w,pad_h,scale);
    auto b2 = process_feature(out2,h2,w2,anchors2,32,(int)anchors2.size(),num_classes,obj_thresh,src_width,src_height,pad_w,pad_h,scale);
    boxes.insert(boxes.end(), b0.begin(), b0.end());
    boxes.insert(boxes.end(), b1.begin(), b1.end());
    boxes.insert(boxes.end(), b2.begin(), b2.end());
    printf("  候选框: %d\n", (int)boxes.size());

    std::vector<Box> final_boxes;
    for (int c = 0; c < num_classes; c++) {
        std::vector<Box> cls_boxes;
        for (auto& b : boxes) if (b.class_id == c) cls_boxes.push_back(b);
        auto keep = nms_boxes(cls_boxes, nms_thresh);
        for (int k : keep) final_boxes.push_back(cls_boxes[k]);
    }
    return final_boxes;
}

// ===================== 主函数 =====================

int main(int argc, char* argv[]) {
    // 0. 解析命令行参数
    const char* model_path = (argc > 1) ? argv[1] : DEFAULT_MODEL_PATH;
    const char* image_dir  = (argc > 2) ? argv[2] : DEFAULT_IMAGE_DIR;
    const char* output_dir = (argc > 3) ? argv[3] : DEFAULT_OUTPUT_DIR;
    printf("[INFO] 模型: %s\n", model_path);
    printf("[INFO] 输入: %s\n", image_dir);
    printf("[INFO] 输出: %s\n", output_dir);

    // 1. 创建输出目录
    mkdir(output_dir, 0755);

    // 2. 加载模型
    rknn_context ctx = 0;
    FILE* mfp = fopen(model_path, "rb");
    if (!mfp) { printf("[ERROR] 无法打开模型: %s\n", model_path); return -1; }
    fseek(mfp, 0, SEEK_END); size_t model_size = ftell(mfp); fseek(mfp, 0, SEEK_SET);
    void* model_data = malloc(model_size);
    fread(model_data, 1, model_size, mfp);
    fclose(mfp);

    int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    if (ret < 0) { printf("[ERROR] rknn_init 失败: %d\n", ret); free(model_data); return -1; }
    printf("[INFO] 模型加载成功\n");

    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    // 打印模型的输入输出维度属性
    for (int i = 0; i < io_num.n_output; i++) {
        rknn_tensor_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        printf("[INFO] 输出 %d: name=%s, fmt=%d, type=%d, n_dims=%d, dims=[%d, %d, %d, %d]\n",
               i, attr.name, attr.fmt, attr.type, attr.n_dims,
               attr.dims[0], attr.dims[1], attr.dims[2], attr.dims[3]);
    }

    // 3. 查询模型输入尺寸（利用 YOLO 输入为正方形 H==W，从 dims 自动推导）
    rknn_tensor_attr input_attr;
    memset(&input_attr, 0, sizeof(input_attr));
    input_attr.index = 0;
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
    int d0 = input_attr.dims[0], d1 = input_attr.dims[1];
    int d2 = input_attr.dims[2], d3 = input_attr.dims[3];
    printf("[INFO] 模型输入原始: dims=[%d,%d,%d,%d], fmt=%d\n", d0, d1, d2, d3, input_attr.fmt);

    // YOLO 输入总是正方形，找到 dims 中相等且 >16 的两个值 = H/W，剩下的 = 通道数
    int dst_height, dst_width, dst_bpp;
    if      (d0 == d1 && d0 > 16)      { dst_height=d0; dst_width=d1; dst_bpp=d2; }  // [H,W,C,1] RV1126
    else if (d1 == d2 && d1 > 16)      { dst_height=d1; dst_width=d2; dst_bpp=d3; }  // [1,H,W,C] NHWC
    else if (d2 == d3 && d2 > 16)      { dst_bpp=d1; dst_height=d2; dst_width=d3; }  // [1,C,H,W] NCHW
    else if (d0 == d2 && d0 > 16)      { dst_height=d0; dst_width=d2; dst_bpp=d1; }  // [H,C,W,1]
    else {
        printf("[WARN] 无法通过正方形推断输入尺寸，回退到 640x640x3\n");
        dst_height=640; dst_width=640; dst_bpp=3;
    }
    // RGA 要求 RGB888 的字节步长 4 对齐：width * 3 % 4 == 0
    int dst_align_w = (dst_width + 3) & ~3;
    const int dst_format    = RK_FORMAT_RGB_888;
    const int dst_buf_size  = dst_align_w * dst_height * dst_bpp;  // 对齐后的缓冲区大小
    const int dst_data_size = dst_width * dst_height * dst_bpp;    // 实际数据大小 (给 RKNN)
    printf("[INFO] 模型输入: %dx%d, bpp=%d, stride=%d\n",
           dst_width, dst_height, dst_bpp, dst_align_w);
    uint8_t* dst_buf = (uint8_t*)malloc(dst_buf_size);

    auto anchors0 = std::vector<std::pair<float,float>>{{10,13},{16,30},{33,23}};
    auto anchors1 = std::vector<std::pair<float,float>>{{30,61},{62,45},{59,119}};
    auto anchors2 = std::vector<std::pair<float,float>>{{116,90},{156,198},{373,326}};

    // 4. 遍历目录
    DIR* dir = opendir(image_dir);
    if (!dir) {
        printf("[ERROR] 无法打开目录: %s\n", image_dir);
        free(dst_buf); rknn_destroy(ctx); free(model_data); return -1;
    }

    int img_count = 0;
    struct dirent* entry;

    while ((entry = readdir(dir)) != NULL) {
        const char* name = entry->d_name;
        size_t len = strlen(name);
        if (len < 5) continue;

        // 找扩展名，支持 .bmp .jpg .jpeg .png .BMP .JPG .JPEG .PNG
        const char* ext = NULL;
        for (int ei = (int)len - 1; ei >= 0; ei--) {
            if (name[ei] == '.') { ext = name + ei; break; }
        }
        if (!ext) continue;
        bool is_img = (strcasecmp(ext, ".bmp")  == 0) ||
                      (strcasecmp(ext, ".jpg")  == 0) ||
                      (strcasecmp(ext, ".jpeg") == 0) ||
                      (strcasecmp(ext, ".png")  == 0);
        if (!is_img) continue;

        // 路径拼接
        char img_path[512], out_path[512], stem[512];
        int stem_len = (int)(ext - name);
        snprintf(img_path, sizeof(img_path), "%s/%s", image_dir, name);
        memcpy(stem, name, stem_len); stem[stem_len] = '\0';
        snprintf(out_path, sizeof(out_path), "%s/%s_result.bmp", output_dir, stem);

        printf("\n[%d] %s\n", ++img_count, img_path);

        // 4a. 读图 (stb_image 返回 RGB 数据)
        int src_width, src_height, src_c;
        uint8_t* bmp_data = stbi_load(img_path, &src_width, &src_height, &src_c, 3);
        src_c = 3;
        if (!bmp_data) { printf("  [WARN] 读取失败，跳过\n"); continue; }

        // 4b. 像素宽度对齐拷贝 (满足 RGA 严格的内存对齐要求)
        int align_w = (src_width + 3) & (~3); // 强行将宽度向上取整为 4 的倍数 (如 526 变成 528)
        int src_wstride_bytes = align_w * src_c;
        uint8_t* src_buf = (uint8_t*)malloc(src_wstride_bytes * src_height);
        memset(src_buf, 0, src_wstride_bytes * src_height); // 必须清空内存，防止黑边有杂色

        // 将读取到的 BMP 原图逐行拷贝到对齐后的内存中
        for (int i = 0; i < src_height; i++) {
            memcpy(src_buf + i * src_wstride_bytes, bmp_data + i * src_width * src_c, src_width * src_c);
        }

        // 4c. 灰色填充 dst_buf
        memset(dst_buf, 0x80, dst_buf_size);

        // 4d. letterbox 参数
        float r     = fminf((float)dst_height/src_height, (float)dst_width/src_width);
        int   new_w = (int)roundf(src_width  * r);
        int   new_h = (int)roundf(src_height * r);
        int   left  = (dst_width  - new_w) / 2;
        int   top   = (dst_height - new_h) / 2;

        // 4e. RGA 缩放 (终极安全版)
        rga_buffer_t src_img, dst_img, pat_img;
        im_rect src_rect, dst_rect, pat_rect;

        // 🌟 强力清空内存，杜绝脏数据
        memset(&src_img, 0, sizeof(rga_buffer_t));
        memset(&dst_img, 0, sizeof(rga_buffer_t));
        memset(&pat_img, 0, sizeof(rga_buffer_t));
        memset(&src_rect, 0, sizeof(im_rect));
        memset(&dst_rect, 0, sizeof(im_rect));
        memset(&pat_rect, 0, sizeof(im_rect));

        // 🌟 致命陷阱修复：必须将未使用的 fd 设为 -1
        src_img.fd = -1;
        dst_img.fd = -1;
        pat_img.fd = -1; // 告诉底层：没有 pat 图像！

        // 🎨 硬件自动色彩转换：告诉 RGA 原图是 BGR，目标要 RGB
        src_img = wrapbuffer_virtualaddr(src_buf, src_width, src_height, RK_FORMAT_RGB_888, align_w, src_height);
        dst_img = wrapbuffer_virtualaddr(dst_buf, dst_width, dst_height, RK_FORMAT_RGB_888, dst_align_w, dst_height);
        
        // 严格设定坐标
        src_rect.x = 0;     src_rect.y = 0;   src_rect.width = src_width; src_rect.height = src_height;
        dst_rect.x = left;  dst_rect.y = top; dst_rect.width = new_w;     dst_rect.height = new_h;

        if (left + new_w > dst_width || top + new_h > dst_height) {
            printf("  [WARN] dst_rect 越界\n");
            free(src_buf); stbi_image_free(bmp_data); continue;
        }

        // 调用 RGA
        IM_STATUS status = improcess(src_img, dst_img, pat_img, src_rect, dst_rect, pat_rect, IM_SYNC);
        if (status != IM_STATUS_SUCCESS) {
            printf("  [WARN] RGA 失败: %s\n", imStrError(status));
            free(src_buf); stbi_image_free(bmp_data); continue;
        }

        // RGA 写入使用了对齐步长，若 dst_align_w > dst_width 需要压缩掉行间填充
        if (dst_align_w != dst_width) {
            for (int y = 1; y < dst_height; y++) {
                memmove(dst_buf + y * dst_width * dst_bpp,
                        dst_buf + y * dst_align_w * dst_bpp,
                        dst_width * dst_bpp);
            }
        }

        // 4f. 推理
        rknn_input inputs[1] = {};
        inputs[0].index=0; inputs[0].type=RKNN_TENSOR_UINT8;
        inputs[0].size=dst_data_size; inputs[0].pass_through=0;
        inputs[0].fmt=RKNN_TENSOR_NHWC; inputs[0].buf=dst_buf;

        if (rknn_inputs_set(ctx,1,inputs)<0 || rknn_run(ctx,NULL)<0) {
            printf("  [WARN] 推理失败\n");
            free(src_buf); stbi_image_free(bmp_data); continue;
        }

        // 4g. 取输出
        rknn_output outputs[io_num.n_output];
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < (int)io_num.n_output; i++) {
            outputs[i].want_float = 1; 
        }
        if (rknn_outputs_get(ctx, io_num.n_output, outputs, NULL) < 0) {
            printf("  [WARN] 获取输出失败\n");
            free(src_buf); stbi_image_free(bmp_data); continue;
        }

        // 动态推断：从输出 tensor 维度获取 grid 尺寸和类别数
        int dynamic_num_classes = -1;
        int grid_h[3] = {0, 0, 0};
        int grid_w[3] = {0, 0, 0};
        std::vector<std::vector<float>> f_outputs(io_num.n_output);

        for (int i = 0; i < (int)io_num.n_output; i++) {
            rknn_tensor_attr attr;
            memset(&attr, 0, sizeof(attr));
            attr.index = i;
            rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));

            // 从 dims 推导 grid_h / grid_w / 通道数
            // 兼容多种格式：标准 [1,H,W,C] / [1,C,H,W] 以及 RV1126 的 [H,W,C,1]
            // 利用 YOLO grid 为正方形 (H==W)，找相等的两个非 1 值
            int d0=attr.dims[0], d1=attr.dims[1], d2=attr.dims[2], d3=attr.dims[3];
            int ch;
            if      (d0 == d1 && d0 > 1) { grid_h[i]=d0; grid_w[i]=d1; ch=d2; }  // [H,W,C,1] RV1126
            else if (d1 == d2 && d1 > 1) { grid_h[i]=d1; grid_w[i]=d2; ch=d3; }  // [1,H,W,C] NHWC
            else if (d2 == d3 && d2 > 1) { grid_h[i]=d2; grid_w[i]=d3; ch=d1; }  // [1,C,H,W] NCHW
            else if (d0 == d2 && d0 > 1) { grid_h[i]=d0; grid_w[i]=d2; ch=d1; }  // [H,C,W,1]
            else                         { grid_h[i]=d1; grid_w[i]=d2; ch=d3; }  // 兜底

            if (i == 0) {
                dynamic_num_classes = (ch / (int)anchors0.size()) - 5;
            }

            int elems = attr.dims[0] * d1 * d2 * d3;
            f_outputs[i].resize(elems);

            if (outputs[i].size == elems) {
                uint8_t* src = (uint8_t*)outputs[i].buf;
                for (int j = 0; j < elems; j++) {
                    f_outputs[i][j] = (src[j] - attr.zp) * attr.scale;
                }
            }
            else if (outputs[i].size == elems * 4) {
                memcpy(f_outputs[i].data(), outputs[i].buf, elems * 4);
            }
        }

        // 4h. 后处理 (grid 尺寸从模型输出维度动态获取)
        std::vector<Box> final_boxes = post_process(
            f_outputs[0].data(), grid_h[0], grid_w[0],
            f_outputs[1].data(), grid_h[1], grid_w[1],
            f_outputs[2].data(), grid_h[2], grid_w[2],
            anchors0, anchors1, anchors2,
            dynamic_num_classes,
            0.85f,
            0.3f,
            src_width, src_height, left, top, r
        );
        rknn_outputs_release(ctx, io_num.n_output, outputs);
        
        // 4i. 画框并存图
        for (auto& box : final_boxes) {
            int x1 = (int)box.x1, y1 = (int)box.y1, x2 = (int)box.x2, y2 = (int)box.y2;
            int cid = box.class_id;

            uint8_t b_color = (cid * 85 + 100) % 255;
            uint8_t g_color = (cid * 145 + 50) % 255;
            uint8_t r_color = (cid * 205 + 150) % 255;

            printf("  [class_%d conf=%.3f] (%d,%d)-(%d,%d)\n",
                   cid, box.conf, x1, y1, x2, y2);

            draw_rect(bmp_data, src_width, src_height, src_c, x1, y1, x2, y2,
                      b_color, g_color, r_color, 3);
        }
        stbi_write_bmp(out_path, src_width, src_height, 3, bmp_data);

        free(src_buf);
        stbi_image_stbi_image_free(bmp_data);
    }

    closedir(dir);

    // 5. 释放全局资源
    free(dst_buf);
    rknn_destroy(ctx);
    free(model_data);

    printf("\n[INFO] 完成，共处理 %d 张，结果保存在: %s\n", img_count, output_dir);
    return 0;
}
