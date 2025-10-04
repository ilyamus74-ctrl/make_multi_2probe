// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <memory>
#include "buffer.h"

#include "yolov8.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"

// --- CPU letterbox без RGA (src/dst: RGB888) ---
static inline uint8_t clamp_u8(int v) { return (v < 0) ? 0 : (v > 255 ? 255 : v); }

static void fill_rgb(uint8_t* dst, int W, int H, int color) {
    // color как у демо: 114 => серый; храним одинаково в R,G,B
    uint8_t c = (uint8_t)color;
    for (int y = 0; y < H; ++y) {
        uint8_t* row = dst + y * W * 3;
        for (int x = 0; x < W; ++x) {
            row[3*x+0] = c;
            row[3*x+1] = c;
            row[3*x+2] = c;
        }
    }
}

static void resize_nn_rgb888(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) {
    // nearest-neighbor, быстрый и простейший
    for (int y = 0; y < dh; ++y) {
        int sy = (int)((1.0 * y * sh) / dh);
        const uint8_t* srow = src + sy * sw * 3;
        uint8_t* drow = dst + y * dw * 3;
        for (int x = 0; x < dw; ++x) {
            int sx = (int)((1.0 * x * sw) / dw);
            const uint8_t* spx = srow + sx * 3;
            uint8_t* dpx = drow + x * 3;
            dpx[0] = spx[0];
            dpx[1] = spx[1];
            dpx[2] = spx[2];
        }
    }
}

// src->format == IMAGE_FORMAT_RGB888, dst->format == IMAGE_FORMAT_RGB888
static int convert_image_with_letterbox_cpu(image_buffer_t* src, image_buffer_t* dst,
                                            letterbox_t* lb, int bg_color)
{
    if (!src || !dst || !lb || !src->virt_addr || !dst->virt_addr) return -1;
    if (src->format != IMAGE_FORMAT_RGB888 || dst->format != IMAGE_FORMAT_RGB888) return -2;

    const int sw = src->width,  sh = src->height;
    const int dw = dst->width,  dh = dst->height;
    if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return -3;

    // 1) фон
    fill_rgb(dst->virt_addr, dw, dh, bg_color);

    // 2) рассчёт масштаб/пэдинги
    double scale = std::min( (double)dw / sw, (double)dh / sh );
    int rw = (int)std::round(sw * scale);
    int rh = (int)std::round(sh * scale);
    if (rw <= 0) rw = 1;
    if (rh <= 0) rh = 1;

    int x_pad = (dw - rw) / 2;
    int y_pad = (dh - rh) / 2;

    // 3) ресайз исходника во временный буфер (или прямо в место назначения)
    std::vector<uint8_t> tmp(rw * rh * 3);
    resize_nn_rgb888(src->virt_addr, sw, sh, tmp.data(), rw, rh);

    // 4) копируем по центру
    for (int y = 0; y < rh; ++y) {
        uint8_t* drow = dst->virt_addr + ((y + y_pad) * dw + x_pad) * 3;
        const uint8_t* srow = tmp.data() + y * rw * 3;
        memcpy(drow, srow, rw * 3);
    }

    // 5) letterbox info для post_process
    memset(lb, 0, sizeof(*lb));
    lb->x_pad = x_pad;
    lb->y_pad = y_pad;
    lb->scale = (float)scale;

    return 0;
}

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

int init_yolov8_model(const char *model_path, rknn_app_context_t *app_ctx)
{
    int ret;
    int model_len = 0;
    char *model_raw = nullptr;
    rknn_context ctx = 0;

    // Load RKNN Model
    model_len = read_data_from_file(model_path, &model_raw);
    std::unique_ptr<char, decltype(&free)> model(model_raw, &free);
    if (!model)
    {
        printf("load_model fail!\n");
        return -1;
    }

    ret = rknn_init(&ctx, model.get(), model_len, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }

// --- use all NPU cores (RK3588: 3 ядра) ---
/*
{
    int ret_mask = rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2);
    if (ret_mask != RKNN_SUCC) {
        fprintf(stderr, "warn: rknn_set_core_mask failed: %d\n", ret_mask);
    } else {
        fprintf(stderr, "rknn core mask: 0_1_2 (all cores)\n");
    }
}
*/
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
    
    // Get Model Input Info
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    app_ctx->rknn_ctx = ctx;

    // TODO
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type == RKNN_TENSOR_INT8)
    {
        app_ctx->is_quant = true;
    }
    else
    {
        app_ctx->is_quant = false;
    }

    app_ctx->io_num = io_num;
    app_ctx->input_attrs.assign(input_attrs, input_attrs + io_num.n_input);
    app_ctx->output_attrs.assign(output_attrs, output_attrs + io_num.n_output);

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height = input_attrs[0].dims[2];
        app_ctx->model_width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        app_ctx->model_height = input_attrs[0].dims[1];
        app_ctx->model_width = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
           app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return 0;
}

int release_yolov8_model(rknn_app_context_t *app_ctx)
{
    app_ctx->input_attrs.clear();
    app_ctx->output_attrs.clear();
    if (app_ctx->rknn_ctx != 0)
    {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

int inference_yolov8_model(rknn_app_context_t *app_ctx, image_buffer_t *img, object_detect_result_list *od_results)
{
    int ret;
    image_buffer_t dst_img;
    Buffer<uint8_t> dst_buffer;
    letterbox_t letter_box;
    rknn_input inputs[app_ctx->io_num.n_input];
    rknn_output outputs[app_ctx->io_num.n_output];
    const float nms_threshold = NMS_THRESH;      // 默认的NMS阈值
    const float box_conf_threshold = BOX_THRESH; // 默认的置信度阈值
    int bg_color = 114;

    if ((!app_ctx) || !(img) || (!od_results))
    {
        return -1;
    }

    memset(od_results, 0x00, sizeof(*od_results));
    memset(&letter_box, 0, sizeof(letterbox_t));
    memset(&dst_img, 0, sizeof(image_buffer_t));
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    image_buffer_t src_img = *img;
    Buffer<uint8_t> gray_buffer;
    if (img->format == IMAGE_FORMAT_GRAY8) {
        src_img.format = IMAGE_FORMAT_RGB888;
        src_img.size = img->width * img->height * 3;
        gray_buffer.resize(src_img.size);
        src_img.virt_addr = gray_buffer.data();
        const uint8_t* src = img->virt_addr;
        uint8_t* dst = src_img.virt_addr;
        for (int i = 0; i < img->width * img->height; ++i) {
            uint8_t v = src[i];
            dst[3*i] = v;
            dst[3*i+1] = v;
            dst[3*i+2] = v;
        }
    }


    // Pre Process
    dst_img.width = app_ctx->model_width;
    dst_img.height = app_ctx->model_height;
    dst_img.format = IMAGE_FORMAT_RGB888;
    dst_img.size = get_image_size(&dst_img);
    dst_buffer.resize(dst_img.size);
    dst_img.virt_addr = dst_buffer.data();
    if (dst_buffer.empty())
    {
        printf("malloc buffer size:%d fail!\n", dst_img.size);
        return -1;
    }

    // letterbox
    // --- было ---
// ret = convert_image_with_letterbox(img, &dst_img, &letter_box, bg_color);
// if (ret < 0) { printf("convert_image_with_letterbox fail! ret=%d\n", ret); return -1; }

// --- стало: CPU letterbox без RGA ---
/*    ret = convert_image_with_letterbox_cpu(img, &dst_img, &letter_box, bg_color);
    if (ret < 0) {
        printf("convert_image_with_letterbox_cpu fail! ret=%d\n", ret);
        return -1;
    }
*/
// letterbox с RGA (быстрее и разгружает CPU)
/*    ret = convert_image_with_letterbox(img, &dst_img, &letter_box, bg_color);
    if (ret < 0) {
	printf("convert_image_with_letterbox fail! ret=%d\n", ret);
	return -1;
    }
*/

        ret = convert_image_with_letterbox(&src_img, &dst_img, &letter_box, bg_color);
            if (ret < 0) {
                fprintf(stderr, "RGA letterbox failed (%d), fallback to CPU\n", ret);
                ret = convert_image_with_letterbox_cpu(&src_img, &dst_img, &letter_box, bg_color);
                    if (ret < 0) {
                        fprintf(stderr, "CPU letterbox also failed (%d)\n", ret);
                        return -1;
                    }
            }

    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    inputs[0].buf = dst_img.virt_addr;

    ret = rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    // Run
    printf("rknn_run\n");
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // Get Output
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < app_ctx->io_num.n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = (!app_ctx->is_quant);
    }
    ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        goto out;
    }

    // Post Process
    post_process(app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold, od_results);

    // Remeber to release rknn output
    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);

out:

    return ret;
}