//
// Create by nanmi
// 2021 / 7 / 12
//

#include "tinydet.h"
#include <benchmark.h>
// #include <iostream>

inline float fast_exp(float x) 
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) 
{
    return 1.0f / (1.0f + fast_exp(-x));
}

template<typename _Tp>
int activation_function_softmax(const _Tp* src, _Tp* dst, int length)
{
    const _Tp alpha = *std::max_element(src, src + length);
    _Tp denominator{ 0 };

    for (int i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }

    return 0;
}

bool TinyDet::hasGPU = false;
TinyDet* TinyDet::detector = nullptr;

TinyDet::TinyDet(const char* param, const char* bin, bool useGPU)
{
    this->Net = new ncnn::Net();
    // opt 
#if NCNN_VULKAN
    this->hasGPU = ncnn::get_gpu_count() > 0;
#endif
    this->Net->opt.use_vulkan_compute = this->hasGPU && useGPU;
    this->Net->opt.use_fp16_arithmetic = true;
    this->Net->load_param(param);
    this->Net->load_model(bin);
}

TinyDet::~TinyDet()
{
    delete this->Net;
}

void TinyDet::preprocess(cv::Mat& image, ncnn::Mat& in)
{
    int img_w = image.cols;
    int img_h = image.rows;

    in = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR, img_w, img_h);
    //in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR, img_w, img_h, this->input_width, this->input_height);

    const float mean_vals[3] = { 103.53f, 116.28f, 123.675f };
    const float norm_vals[3] = { 0.017429f, 0.017507f, 0.017125f };
    in.substract_mean_normalize(mean_vals, norm_vals);
}

std::vector<BoxInfo> TinyDet::detect(cv::Mat image, float score_threshold, float nms_threshold)
{
    ncnn::Mat input;
    preprocess(image, input);

    //double start = ncnn::get_current_time();

    auto ex = this->Net->create_extractor();
    ex.set_light_mode(false);
    ex.set_num_threads(4);
#if NCNN_VULKAN
    ex.set_vulkan_compute(this->hasGPU);
#endif
    ex.input("input.1", input);

    std::vector<std::vector<BoxInfo>> results;
    results.resize(this->num_class);

    for (const auto& head_info : this->heads_info)
    {
        ncnn::Mat dis_pred;
        ncnn::Mat cls_pred;
        ex.extract(head_info.dis_layer.c_str(), dis_pred);
        ex.extract(head_info.cls_layer.c_str(), cls_pred); 
        // std::cout << "c:" << cls_pred.c << " h:" << cls_pred.h <<" w:" <<cls_pred.w <<std::endl;

        
        this->decode_infer(cls_pred, dis_pred, head_info.stride, score_threshold, results);
    }

    std::vector<BoxInfo> dets;
    for (int i = 0; i < (int)results.size(); i++)
    {
        this->nms(results[i], nms_threshold);
        
        for (auto box : results[i])
        {
            dets.push_back(box);
        }
    }

    //double end = ncnn::get_current_time();
    //double time = end - start;
    //printf("Detect Time:%7.2f \n", time);

    return dets;
}

void TinyDet::nms(std::vector<BoxInfo>& input_boxes, float NMS_THRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
            * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else {
                j++;
            }
        }
    }
}


// postprocess
// ==========================
void TinyDet::decode_infer(ncnn::Mat& cls_pred, ncnn::Mat& dis_pred, int stride, float threshold, std::vector<std::vector<BoxInfo>>& results)
{
    int feature_h = this->input_size[1] / stride;
    int feature_w = this->input_size[0] / stride;

    //cv::Mat debug_heatmap = cv::Mat(feature_h, feature_w, CV_8UC3);
    for (int idx = 0; idx < feature_h * feature_w; idx++)
    {
        const float* scores = cls_pred.row(idx);

        int row = idx / feature_w;
        int col = idx % feature_w;
        float score = 0;
        int cur_label = 0;
        float all = 0;
        for (int label = 0; label < this->num_class; label++)
        {
            all += scores[label];
            if (scores[label] > score)
            {
                score = scores[label];
                cur_label = label;
            }
        }

        if (score > threshold)
        {
            const float* bbox_pred = dis_pred.row(idx);
            results[cur_label].push_back(this->disPred2Bbox(bbox_pred, cur_label, score, col, row, stride));
        }

    }
}

BoxInfo TinyDet::disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride)
{
    float ct_x = (x + 0.5) * stride;
    float ct_y = (y + 0.5) * stride;
    std::vector<float> dis_pred;
    dis_pred.resize(4);
    for (int i = 0; i < 4; i++)
    {
        float dis = 0;
        float* dis_after_sm = new float[this->reg_max + 1];
        activation_function_softmax(dfl_det + i * (this->reg_max + 1), dis_after_sm, this->reg_max + 1);
        for (int j = 0; j < this->reg_max + 1; j++)
        {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        //std::cout << "dis:" << dis << std::endl;
        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }
    float xmin = (std::max)(ct_x - dis_pred[0], .0f);
    float ymin = (std::max)(ct_y - dis_pred[1], .0f);
    float xmax = (std::min)(ct_x + dis_pred[2], (float)this->input_size[0]);
    float ymax = (std::min)(ct_y + dis_pred[3], (float)this->input_size[1]);

    // std::cout << xmin << "," << ymin << "," << xmax << "," << xmax << "," << std::endl;
    return BoxInfo { xmin, ymin, xmax, ymax, score, label };
}
