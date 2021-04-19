#include <torch/script.h>
#include "torch/torch.h"
#include "torch/jit.h"
#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include <math.h>
#include "configure.h"

static Logger gLogger;
cv::RotatedRect expandBox(const cv::RotatedRect& inBox, float ratio = 1.0) {
    cv::Size size = inBox.size;
    int neww = size.width * ratio;
    int newh = size.height *ratio;
    return cv::RotatedRect(inBox.center, cv::Size(neww, newh), inBox.angle);
}
float paddimg(cv::Mat& In_Out_img, int shortsize = 960) {
    int w = In_Out_img.cols;
    int h = In_Out_img.rows;
    float scale = 1.f;
    if (w < h) {
        scale = (float)shortsize / w;
        h = scale * h;
        w = shortsize;
    }
    else {
        scale = (float)shortsize / h;
        w = scale * w;
        h = shortsize;
    }

    if (h % 32 != 0) {
        h = (h / 32 + 1) * 32;
    }
    if (w % 32 != 0) {
        w = (w / 32 + 1) * 32;
    }

    cv::resize(In_Out_img, In_Out_img, cv::Size(w, h));
    return scale;
}

IActivationLayer* bottleneck(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int ch, int stride, std::string lname, int branch_type)
{
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    std::cout<<"name="<<lname + ".conv1.weights"<<std::endl;
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, ch, DimsHW{ 1, 1 }, weightMap[lname + ".conv1.weight"], emptywts);
    assert(conv1);

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn1", 1e-5);
    assert(bn1);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), ch, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], emptywts);
    conv2->setStrideNd(DimsHW{ stride, stride });
    conv2->setPaddingNd(DimsHW{ 1, 1 });
    assert(conv2);

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".bn2", 1e-5);
    assert(bn2);

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), ch * 4, DimsHW{ 1, 1 }, weightMap[lname + ".conv3.weight"], emptywts);
    assert(conv3);

    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + ".bn3", 1e-5);
    assert(bn3);
    IElementWiseLayer* ew1;
    // branch_type 0:shortcut,1:conv+bn+shortcut,2:maxpool+shortcut
    if (branch_type == 0)
    {
        ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
        assert(ew1);
    }
    else if (branch_type == 1)
    {
        IConvolutionLayer* conv4 = network->addConvolutionNd(input, ch * 4, DimsHW{ 1, 1 }, weightMap[lname + ".downsample.0.weight"], emptywts);
        conv4->setStrideNd(DimsHW{ stride, stride });
        assert(conv4);
        IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + ".downsample.1", 1e-5);
        assert(bn4);
        ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
        assert(ew1);
    }
    else
    {
        IPoolingLayer* pool = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{ 1, 1 });
        pool->setStrideNd(DimsHW{ 2, 2 });
        assert(pool);
        ew1 = network->addElementWise(*pool->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
        assert(ew1);
    }
    IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    return relu3;
}

IActivationLayer* addConvBnRelu(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int kernel, int stride, int pad, std::string lname_conv, std::string lname_bn)
{
    IConvolutionLayer* conv = network->addConvolutionNd(input, outch, DimsHW{ kernel, kernel }, weightMap[lname_conv + ".weight"], weightMap[lname_conv + ".bias"]);
    conv->setStrideNd(DimsHW{ stride, stride });
    conv->setPaddingNd(DimsHW{ pad, pad });
    assert(conv);

    IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname_bn, 1e-5);

    IActivationLayer* ac = network->addActivation(*bn->getOutput(0), ActivationType::kRELU);
    assert(ac);
    return ac;
}

IResizeLayer* upsample(INetworkDefinition* network, IActivationLayer* x, IActivationLayer* y)
{
    IResizeLayer* x_resize = network->addResize(*x->getOutput(0));
    auto y_shape = network->addShape(*y->getOutput(0))->getOutput(0);
    x_resize->setInput(1, *y_shape);
    x_resize->setResizeMode(ResizeMode::kLINEAR);
    x_resize->setAlignCorners(true);
    assert(x_resize);
    return x_resize;
}

IResizeLayer* upsample_2(INetworkDefinition* network, IConvolutionLayer* x, IConvolutionLayer* y)
{
    IResizeLayer* x_resize = network->addResize(*x->getOutput(0));
    auto y_shape = network->addShape(*y->getOutput(0))->getOutput(0);
    x_resize->setInput(1, *y_shape);
    x_resize->setResizeMode(ResizeMode::kLINEAR);
    x_resize->setAlignCorners(true);
    assert(x_resize);
    return x_resize;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims4{ 1, 3, -1, -1 });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(path_wts);
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    /* ------ Resnet18 backbone------ */
    // Add convolution layer with 6 outputs and a 5x5 filter.
    IConvolutionLayer* conv1 = network->addConvolution(*data, 64, DimsHW{ 7, 7 }, weightMap["conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{ 2, 2 });
    conv1->setPadding(DimsHW{ 3, 3 });
    assert(conv1);

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);
    assert(bn1);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{ 3, 3 });
    pool1->setStrideNd(DimsHW{ 2, 2 });
    pool1->setPaddingNd(DimsHW{ 1, 1 }); //    pool1->setPrePadding(DimsHW{ 0, 0 });  //    pool1->setPostPadding(DimsHW{ 1, 1 });
    assert(pool1);

    IActivationLayer* x;
    x = bottleneck(network, weightMap, *pool1->getOutput(0), 64, 1, "layer1.0", 1);
    x = bottleneck(network, weightMap, *x->getOutput(0), 64, 1, "layer1.1", 0);
    // C2
    IActivationLayer* block1 = bottleneck(network, weightMap, *x->getOutput(0), 64, 1, "layer1.2", 0); //2

    x = bottleneck(network, weightMap, *block1->getOutput(0), 128, 2, "layer2.0", 1);
    x = bottleneck(network, weightMap, *x->getOutput(0), 128, 1, "layer2.1", 0);
    x = bottleneck(network, weightMap, *x->getOutput(0), 128, 1, "layer2.2", 0);
    // C3
    IActivationLayer* block2 = bottleneck(network, weightMap, *x->getOutput(0), 128, 1, "layer2.3", 0);

    x = bottleneck(network, weightMap, *block2->getOutput(0), 256, 2, "layer3.0", 1);
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 1, "layer3.1", 0);
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 1, "layer3.2", 0);
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 1, "layer3.3", 0);
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 1, "layer3.4", 0);
    // C4
    IActivationLayer* block3 = bottleneck(network, weightMap, *x->getOutput(0), 256, 1, "layer3.5", 0);

    x = bottleneck(network, weightMap, *block3->getOutput(0), 512, 2, "layer4.0", 1);
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 1, "layer4.1", 0);
    // C5
    IActivationLayer* block4 = bottleneck(network, weightMap, *x->getOutput(0), 512, 1, "layer4.2", 0);

    IActivationLayer* P5 = addConvBnRelu(network, weightMap, *block4->getOutput(0), 256, 1, 1, 0, "toplayer", "toplayer_bn");
    IActivationLayer* C4 = addConvBnRelu(network, weightMap, *block3->getOutput(0), 256, 1, 1, 0, "latlayer1", "latlayer1_bn");

    IResizeLayer* P5_resize = upsample(network, P5, C4);
    IElementWiseLayer* P44 = network->addElementWise(*P5_resize->getOutput(0), *C4->getOutput(0), ElementWiseOperation::kSUM);
    assert(P44);
    IActivationLayer* P4 = addConvBnRelu(network, weightMap, *P44->getOutput(0), 256, 3, 1, 1, "smooth1", "smooth1_bn");
    assert(P4);

    IActivationLayer* C3 = addConvBnRelu(network, weightMap, *block2->getOutput(0), 256, 1, 1, 0, "latlayer2", "latlayer2_bn");
    IResizeLayer* P4_resize = upsample(network, P4, C3);
    IElementWiseLayer* P33 = network->addElementWise(*P4_resize->getOutput(0), *C3->getOutput(0), ElementWiseOperation::kSUM);
    assert(P33);
    IActivationLayer* P3 = addConvBnRelu(network, weightMap, *P33->getOutput(0), 256, 3, 1, 1, "smooth2", "smooth2_bn");
    assert(P3);

    IActivationLayer* C2 = addConvBnRelu(network, weightMap, *block1->getOutput(0), 256, 1, 1, 0, "latlayer3", "latlayer3_bn");
    IResizeLayer* P3_resize = upsample(network, P3, C2);
    IElementWiseLayer* P22 = network->addElementWise(*P3_resize->getOutput(0), *C2->getOutput(0), ElementWiseOperation::kSUM);
    assert(P22);
    IActivationLayer* P2 = addConvBnRelu(network, weightMap, *P22->getOutput(0), 256, 3, 1, 1, "smooth3", "smooth3_bn");
    assert(P2);

    IResizeLayer* P3_end = upsample(network, P3, P2);
    IResizeLayer* P4_end = upsample(network, P4, P2);
    IResizeLayer* P5_end = upsample(network, P5, P2);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //由于trt的upsample通道数也会一致 就是比如 a [1,256,320,320]  b[1,3,640,640]  upsample(a,b)之后结果就是[1,3,640,640]
    //所以这里构造一个tmp_conv，其通道数和a一样，hw和b一样
    float tmp_data[3*7] = {1};
    Weights emptywts1{ DataType::kFLOAT, tmp_data, 3*7 };

    IResizeLayer* tmp_resize = network->addResize(*P5_end->getOutput(0));
    auto y_shape = network->addShape(*data)->getOutput(0);/////
    tmp_resize->setInput(1, *y_shape);
    tmp_resize->setResizeMode(ResizeMode::kLINEAR);
    tmp_resize->setAlignCorners(true);
    assert(tmp_resize); //  tmp_resize [b c_data(3) h_data w_data]

    IConvolutionLayer* tmp_conv = network->addConvolutionNd(*tmp_resize->getOutput(0), 7, DimsHW{ 1, 1 }, emptywts1, emptywts);
    tmp_conv->setStrideNd(DimsHW{ 1, 1 });
    tmp_conv->setPaddingNd(DimsHW{ 0, 0 });
    assert(tmp_conv);  //tmp_conv [b c_data(7) h_data w_data]
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ITensor* out_tmp[] = {P2->getOutput(0), P3_end->getOutput(0), P4_end->getOutput(0), P5_end->getOutput(0)};
    auto out = network->addConcatenation(out_tmp, 4);

    int num_classes = 7;
    IActivationLayer* out_1 = addConvBnRelu(network, weightMap, *out->getOutput(0), 256, 3, 1, 1, "conv2", "bn2");
    IConvolutionLayer* out_2 = network->addConvolutionNd(*out_1->getOutput(0), num_classes, DimsHW{ 1, 1 }, weightMap["conv3.weight"], weightMap["conv3.bias"]);
    out_2->setStrideNd(DimsHW{ 1, 1 });
    out_2->setPaddingNd(DimsHW{ 0, 0 });
    assert(out_2);

    IResizeLayer* out_end_2 = upsample_2(network, out_2, tmp_conv);

    out_end_2->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*out_end_2->getOutput(0));

    IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kMIN, Dims4(1, 3, MIN_INPUT_SIZE, MIN_INPUT_SIZE));
    profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kOPT, Dims4(1, 3, OPT_INPUT_H, OPT_INPUT_W));
    profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kMAX, Dims4(1, 3, MAX_INPUT_SIZE, MAX_INPUT_SIZE));
    config->addOptimizationProfile(profile);

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  //20: 16MB   28:256
#ifdef FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }
    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    //ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

bool save_tensor_txt(torch::Tensor tensor_in_,std::string path_txt)
{
#include "fstream"
    std::ofstream outfile(path_txt);
    torch::Tensor tensor_in = tensor_in_.clone();
//    tensor_in = tensor_in.view({-1,1});
    tensor_in = tensor_in.reshape({-1,1});
    tensor_in = tensor_in.to(torch::kCPU);

    auto result_data = tensor_in.accessor<float, 2>();

    for(int i=0;i<result_data.size(0);i++)
    {
        float val = result_data[i][0];
//        std::cout<<"val="<<val<<std::endl;
        outfile<<val<<std::endl;
    }
    return true;
}

std::vector<cv::Mat> general_process_pse_mingpai(cv::Mat& m_,int kernel_id_start,int kernel_num,
        IExecutionContext& context)
{
    int input_h = m_.rows;
    int input_w = m_.cols;
    int output_h = m_.rows;
    int output_w = m_.cols;
    int output_ch = 7;

    std::vector<float> m_mean = {0.485, 0.456, 0.406};
    std::vector<float> m_std = {0.229, 0.224, 0.225};

    std::vector<cv::Mat> kernals;
    if(m_.empty() || kernel_num <= 0) { return kernals; }
    cv::Mat m = m_.clone();
    if(1 == m.channels()) { cv::cvtColor(m,m,CV_GRAY2BGR); }
    // Load tensor from cv::Mat, shape is (H, W, C)
    std::vector<int64_t> sizes = {m.rows, m.cols, m.channels()};
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kByte);//.dtype(torch::kByte);
    torch::Tensor tensor_image = torch::from_blob(m.data, torch::IntList(sizes), options);
    // Permute tensor, shape is (C, H, W)
    tensor_image = tensor_image.permute({2, 0, 1});

    // Convert tensor dtype to float32, and range from [0, 255] to [0, 1]
    tensor_image = tensor_image.toType(torch::ScalarType::Float).div_(255.0f);
    // Subtract mean value
    for (int i = 0; i < std::min<int64_t>(m_mean.size(), tensor_image.size(0)); i++) {
        tensor_image[i] = tensor_image[i].sub_(m_mean[i]);
    }
    // Divide by std value
    for (int i = 0; i < std::min<int64_t>(m_std.size(), tensor_image.size(0)); i++) {
        tensor_image[i] = tensor_image[i].div_(m_std[i]);
    }

    tensor_image = tensor_image.reshape({-1});
    void* input = tensor_image.data_ptr();
//    tensor_image.print();
//    std::cout << "CUDA:   " << torch::cuda::is_available() << std::endl;
//    std::cout << "CUDNN:  " << torch::cuda::cudnn_is_available() << std::endl;
//    std::cout << "GPU(s): " << torch::cuda::device_count() << std::endl;
/////////////////////////////////////////////////////////////////////
    const ICudaEngine& engine = context.getEngine();
    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    context.setBindingDimensions(inputIndex, Dims4(1, 3, input_h, input_w));

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_h * input_w * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], output_ch * output_h * output_w * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueueV2(buffers, stream, nullptr);
    ///////////////////////////////////////////////////////////////////////
    torch::Tensor m_result_tensor = torch::from_blob(buffers[outputIndex],{1,7,output_h,output_w}).cuda();//.toType(torch::kFloat64);
    m_result_tensor = m_result_tensor.sigmoid().gt(0.5).detach().cpu().squeeze();

    torch::Tensor b_tmp = (m_result_tensor.sign()+1)/2.0;
    torch::Tensor text = torch::slice(b_tmp,0,0,1);

    int id_start = kernel_id_start;
    if(id_start > 7) { id_start = 7;}
    int id_end = id_start + kernel_num;
    if(id_end > 7) { id_end = 7;}
    torch::Tensor output_1 = at::slice(b_tmp,0,id_start,id_end);//at::Tensor output_1 = at::slice(b_tmp,0,0,kernel_num);
    torch::Tensor kernel = output_1 * text;
    kernel = kernel.toType(torch::kByte);

    kernel = kernel.cpu();
    auto ptr_kernel = kernel.data_ptr();
    int offset = m.rows*m.cols;
    for(int i=0;i<kernel.size(0);i++)
    {
        cv::Mat mask(m.rows, m.cols, CV_8UC1, (uchar*)ptr_kernel + i*offset);
        kernals.emplace_back(mask * 255); //* 255  //qt工程中没有×255  这里不×255有问题
        if(0) {
            mask = mask * 255;
            cv::namedWindow("mask",0);
            cv::imshow("mask",mask);
            cv::namedWindow("image",0);
            cv::imshow("image",m);
            cv::waitKey(0);
        }
    }
    cudaStreamSynchronize(stream);
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    return kernals;
}

void growing_text_line(std::vector<cv::Mat> &kernals, std::vector<std::vector<int>> &text_line, float min_area) {
    cv::Mat label_mat;
    int label_num = connectedComponents(kernals[kernals.size() - 1], label_mat, 4);
    int area[label_num + 1];
    memset(area, 0, sizeof(area));
    for (int x = 0; x < label_mat.rows; ++x) {
        for (int y = 0; y < label_mat.cols; ++y) {
            int label = label_mat.at<int>(x, y);
            if (label == 0) continue;
            area[label] += 1;
        }
    }

    std::queue<cv::Point> queue, next_queue;
    for (int x = 0; x < label_mat.rows; ++x) {
        std::vector<int> row(label_mat.cols);
        for (int y = 0; y < label_mat.cols; ++y) {
            int label = label_mat.at<int>(x, y);

            if (label == 0) continue;
            if (area[label] < min_area) continue;

            cv::Point point(x, y);
            queue.push(point);
            row[y] = label;
        }
        text_line.emplace_back(row);
    }

    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};

    for (int kernal_id = kernals.size() - 2; kernal_id >= 0; --kernal_id) {
        while (!queue.empty()) {
            cv::Point point = queue.front(); queue.pop();
            int x = point.x;
            int y = point.y;
            int label = text_line[x][y];

            bool is_edge = true;
            for (int d = 0; d < 4; ++d) {
                int tmp_x = x + dx[d];
                int tmp_y = y + dy[d];

                if (tmp_x < 0 || tmp_x >= (int)text_line.size()) continue;
                if (tmp_y < 0 || tmp_y >= (int)text_line[1].size()) continue;
                if (kernals[kernal_id].at<char>(tmp_x, tmp_y) == 0) continue;
                if (text_line[tmp_x][tmp_y] > 0) continue;

                cv::Point point(tmp_x, tmp_y);
                queue.push(point);
                text_line[tmp_x][tmp_y] = label;
                is_edge = false;
            }

            if (is_edge) {
                next_queue.push(point);
            }
        }
        swap(queue, next_queue);
    }
}

cv::Mat img_scale(cv::Mat &m_src,int long_size)
{
    float scale = long_size *1.0 / MAX(m_src.cols,m_src.rows);
    cv::Mat m ;
    cv::resize(m_src,m,cv::Size(0,0),scale,scale);
    return m;
}

int get_mask_img(std::vector<std::vector<int>> text_line,cv::Mat &m_mask_bi)
{
    if(0 == text_line.size()) { return -1;}
    m_mask_bi = cv::Mat(text_line.size(),text_line[0].size(),CV_8UC1,cv::Scalar(0));
    int max = -1;
    for(int i=0;i<text_line.size();i++)
    {
      for(int j=0;j<text_line[0].size();j++)
      {
          int pix = text_line[i][j];
          if(0 != pix)
          {
             m_mask_bi.at<uchar>(i,j) = pix;//m_mask_bi.at<uchar>(i,j) = 255;
          }
          if(pix > max)
          {
            max = pix;
          }
      }
    }
    return max;
}

std::vector<std::vector<cv::Point>> process_contour(cv::Mat &image, int long_size, IExecutionContext& context, int kernel_start_id=0, int kernel_num=3, bool b_debug=false)
{
    int T_print = 0;
    std::vector<std::vector<cv::Point>> contours_glob;
    if(image.empty()) { return contours_glob; }
    cv::Mat m = img_scale(image, long_size);
    float ratio_w = m.cols*1.0/image.cols;
    float ratio_h = m.rows*1.0/image.rows;
//    std::cout<<"m_size="<<m.size()<<std::endl;

    std::vector<cv::Mat> kernals = general_process_pse_mingpai(m,kernel_start_id,kernel_num,context);

    double start_0 = cv::getTickCount();
    float min_area = 10.0;
    std::vector<std::vector<int>> text_line;
    growing_text_line(kernals, text_line, min_area);
    double time = ((double)cv::getTickCount() - start_0) / cv::getTickFrequency();
    if (T_print) std::cout<<"time consum growing_text_line==="<<time<<std::endl;
////////////////////////////////////////////////////////////////////////////////////////////////////
//    double start_0 = getTickCount();
    cv::Mat text_img;
    int total_text = get_mask_img(text_line,text_img);
    if(text_img.empty()) { return contours_glob;}

    std::vector<std::vector<cv::Point>> contours_t2;
    for(int j = 1; j <= total_text; j++) {
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::Mat bin;
        cv::threshold(text_img, bin, j - 1, 255, cv::THRESH_TOZERO);
        cv::threshold(bin, bin, j, 255, cv::THRESH_TOZERO_INV);
        cv::threshold(bin, bin, 0, 255, cv::THRESH_BINARY);
        cv::findContours(bin, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
        for (int i = 0; i < contours.size(); i++){
            contours_t2.push_back(contours[i]);
        }
    }

    std::vector<std::vector<cv::Point>> contours_glob_opti;
    for(int i=0;i<contours_t2.size();i++)
    {
        std::vector<cv::Point> contour_t = contours_t2[i];
        std::vector<cv::Point> contour_tt;
        for(int j=0;j<contour_t.size();j++)
        {
            cv::Point pt = contour_t[j];
            cv::Point pt_new = cv::Point(ceil(pt.x *1.0 / ratio_w), ceil(pt.y *1.0 / ratio_h)); //e.g.  ceil(4.11) = 5
            if(pt_new.x >= image.cols) { pt_new.x = image.cols-1;}
            if(pt_new.y >= image.rows) { pt_new.y = image.rows-1;}
            contour_tt.push_back(pt_new);
        }
        contours_glob_opti.push_back(contour_tt);
    }

//    std::vector<std::vector<Point>> contours_0;
//    std::vector<Vec4i> hierarchy;
//    cv::Mat bin;
//    findContours(kernals[1], contours_0, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
//
//    vector<vector<Point>> contours_1;
//    for(int i=0;i<contours_0.size();i++)
//    {
//        vector<Point> contour_t = contours_0[i];
//        vector<Point> contour_tt;
//        for(int j=0;j<contour_t.size();j++)
//        {
//            Point pt = contour_t[j];
//            Point pt_new = Point(pt.x / ratio_w, pt.y / ratio_h);
//            if(pt_new.x >= image.cols) { pt_new.x = image.cols-1;}
//            if(pt_new.y >= image.rows) { pt_new.y = image.rows-1;}
//            contour_tt.push_back(pt_new);
//        }
//        contours_1.push_back(contour_tt);
//    }

    if(b_debug || 0) {
//        cv::Mat m_draw1 = image.clone();
//        cv::drawContours(m_draw1, contours_1, -1, Scalar(0, 255, 255), 4);
//        cv::namedWindow("m_draw_src", 0);
//        cv::imshow("m_draw_src", m_draw1);
//        cv::waitKey(0);

        cv::Mat m_draw2 = image.clone();
        cv::drawContours(m_draw2, contours_glob_opti, -1, cv::Scalar(0, 255, 255), 4);
        cv::namedWindow("m_draw_src_optimize", 0);
        cv::imshow("m_draw_src_optimize", m_draw2);
        cv::waitKey(0);
    }

    return contours_glob_opti;
}

bool save_tensor_txt(float *data,int num, std::string path_txt)
{//验证tensor精度用
#include "fstream"
    std::ofstream outfile(path_txt);

    for(int i=0;i<num;i++)
    {
        float val = data[i];
//        std::cout<<"val="<<val<<std::endl;
        outfile<<val<<std::endl;
    }
    outfile.close();

    return true;
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{ nullptr };
    size_t size{ 0 };

#ifdef SERIALIZE
    IHostMemory* modelStream{ nullptr };
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p(path_save_engin, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;

#elif defined(INFER)
    std::ifstream file(path_read_engin, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
#else
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./psenet -s  // serialize model to plan file" << std::endl;
        std::cerr << "./psenet -d  // deserialize plan file and run inference" << std::endl;
        return -1;
#endif

    // prepare input data ---------------------------
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    std::vector<std::string> file_names;
    if (read_files_in_dir(dir_img, file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }
    int cnt = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (auto f: file_names) {
        std::string path_img = std::string(dir_img) + f;
        std::cout<<++cnt<<"    ::path_img="<<path_img<<std::endl;
        cv::Mat img = cv::imread(path_img);
        if(0 == img.cols*img.rows) { std::cout<<"--------img empty----------"<<std::endl;  continue;}
        int pos = path_img.find_last_of("/");
        std::string name_txt = path_img.substr(pos+1,path_img.size()-pos-5) + ".txt";

        std::vector<std::vector<cv::Point>> vvp_1 = process_contour(img, long_size, *context, kernel_start_id, kernel_num, b_show);

        //下面这段平常用不到/////////////////////////////////////////////////////////////////////
        //为了测试precision, recall, hmean,保存txt
        if(b_test_save_dir)
        {
            std::ofstream out(save_path + name_txt);
            for(int i=0;i<vvp_1.size();i++)
            {
                std::vector<cv::Point> v_pt = vvp_1[i];
                for(int j=0;j<v_pt.size();j++)
                {
                    if(j != v_pt.size()-1)
                    {
                        out << v_pt[j].x << "," << v_pt[j].y<< ",";
                    } else
                    {
                        out << v_pt[j].x << "," << v_pt[j].y << std::endl;
                    }
                }
            }
        }
    }

    auto time_cunsum = std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::steady_clock::now() - t0).count();
    std::cout<<"ave time="<<time_cunsum * 1.0  /cnt<<std::endl;
    std::cout << "all consume time="<<time_cunsum<<"ms"<<std::endl;

    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}