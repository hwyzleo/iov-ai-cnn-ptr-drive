#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/videoio/registry.hpp>
#include <iostream>
#include <memory>
#include <filesystem>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <regex>

#ifdef TENSORRT_AVAILABLE
#include "NvInfer.h"
#include "NvOnnxParser.h"
#endif

#ifdef ONNXRUNTIME_AVAILABLE
#include <onnxruntime_cxx_api.h>
#include <dlfcn.h>
#endif

#ifdef LIBTORCH_AVAILABLE
#include <torch/script.h>
#include <torch/torch.h>
#endif

void printCurrentTime() {
    // 获取当前时间点
    auto now = std::chrono::system_clock::now();

    // 转换为time_t对象
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    // 转换为本地时间并打印
    std::cout << "当前时间: " << std::put_time(std::localtime(&now_time), "%Y-%m-%d %H:%M:%S") << std::endl;
}

void printSystemArchitecture() {
    std::cout << "系统架构: ";

#if defined(__x86_64__) || defined(_M_X64)
    std::cout << "x86_64 (64位)" << std::endl;
#elif defined(__i386) || defined(_M_IX86)
    std::cout << "x86 (32位)" << std::endl;
#elif defined(__aarch64__) || defined(_M_ARM64)
    std::cout << "ARM64" << std::endl;
#elif defined(__arm__) || defined(_M_ARM)
    std::cout << "ARM" << std::endl;
#elif defined(__powerpc64__) || defined(__ppc64__)
        std::cout << "PowerPC 64位" << std::endl;
#elif defined(__powerpc__) || defined(__ppc__)
        std::cout << "PowerPC 32位" << std::endl;
#else
        std::cout << "未知架构" << std::endl;
#endif
}

#ifdef TENSORRT_AVAILABLE
int process(const std::string &videoPath) {
    std::cout << "使用 TensorRT 进行推理\n";
    std::cout << "TensorRT版本: "
                  << NV_TENSORRT_MAJOR << "."
                  << NV_TENSORRT_MINOR << "."
                  << NV_TENSORRT_PATCH << "."
                  << NV_TENSORRT_BUILD << std::endl;

    // 创建自定义Logger
    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING)
                std::cout << msg << std::endl;
        }
    } gLogger;

    // 创建builder和网络
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder) {
        std::cerr << "创建 builder 失败" << std::endl;
        return -1;
    }

    // 创建网络定义
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        std::cerr << "创建网络失败\n";
        return -1;
    }

    // 配置builder
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        std::cerr << "创建builder配置失败" << std::endl;
        return -1;
    }
    size_t workspaceSize = 1 << 32; // 4GB
    config->setMaxWorkspaceSize(workspaceSize);
    std::cout << "设置Workspace大小: " << workspaceSize << " bytes" << std::endl;

    size_t free, total;
    cudaMemGetInfo(&free, &total);
    std::cout << "设置Workspace为 " << workspaceSize << " bytes（可用显存: " << free/(1024*1024) << "MB）" << std::endl;

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    if (!runtime) {
        std::cerr << "创建运行时失败" << std::endl;
        return -1;
    }

    // 读取 TensorRT 模型文件
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::ifstream engineFile("models/model.trt", std::ios::binary);
    if (!engineFile) {
        // 创建ONNX解析器
        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
        if (!parser) {
            std::cerr << "创建解析器失败" << std::endl;
            return -1;
        }

        // 解析ONNX模型
        std::string modelPath = "models/model_int32.onnx";
        std::cout << "推理模型地址: " << "models/model_int32.onnx" << std::endl;
        bool parsed = parser->parseFromFile(modelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
        if (!parsed) {
            std::cerr << "解析ONNX模型失败" << "models/model_int32.onnx" << std::endl;
            return -1;
        }
        std::cout << "解析ONNX模型成功" << std::endl;

        // 构建引擎
        auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
        if (!plan) {
            std::cerr << "创建序列化网络失败" << std::endl;
            return -1;
        }
        std::cout << "创建序列化网络成功" << std::endl;

        engine.reset(runtime->deserializeCudaEngine(plan->data(), plan->size()));
        if (!engine) {
            std::cerr << "创建引擎失败" << std::endl;
            return -1;
        }
        std::cout << "创建引擎成功" << std::endl;
    } else {
        std::cout << "推理模型地址: " << "models/model.trt" << std::endl;
        engineFile.seekg(0, engineFile.end);
        size_t size = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);

        std::vector<char> engineData(size);
        engineFile.read(engineData.data(), size);
        engineFile.close();

        // 反序列化 TensorRT 引擎
        engine.reset(runtime->deserializeCudaEngine(engineData.data(), size));
        if (!engine) {
            std::cerr << "创建引擎失败" << std::endl;
            return -1;
        }
    }

    // 创建执行上下文
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context) {
        std::cerr << "创建执行上下文失败" << std::endl;
        return -1;
    }

    // 获取输入和输出信息
    int nbBindings = engine->getNbBindings();
    std::cout << "绑定数量: " << nbBindings << std::endl;
    std::vector<const char*> tensorNames;

    // 收集所有输入/输出张量名称
    for (int i = 0; i < nbBindings; i++) {
        tensorNames.push_back(engine->getBindingName(i));
    }

    // 假设第一个是输入，最后一个是输出（根据实际模型调整）
    const char* inputName = tensorNames[0];
    const char* outputName = tensorNames[tensorNames.size() - 1];

    // 获取输入和输出维度
    int inputIndex = engine->getBindingIndex(inputName);
    int outputIndex = engine->getBindingIndex(outputName);
    nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
    nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);

    // 计算输入和输出大小
    size_t inputSize = 1;
    for (int i = 0; i < inputDims.nbDims; i++) {
        inputSize *= inputDims.d[i];
    }
    inputSize *= sizeof(float);

    size_t outputSize = 1;
    for (int i = 0; i < outputDims.nbDims; i++) {
        outputSize *= outputDims.d[i];
    }
    outputSize *= sizeof(float);

    // 创建CPU端缓冲区
    std::vector<float> hostInputBuffer(inputSize / sizeof(float));
    std::vector<float> hostOutputBuffer(outputSize / sizeof(float));

    // 创建GPU端缓冲区
    void* deviceInputBuffer = nullptr;
    void* deviceOutputBuffer = nullptr;
    if (cudaMalloc(&deviceInputBuffer, inputSize) != cudaSuccess) {
        std::cerr << "CUDA内存分配失败: deviceInputBuffer" << std::endl;
        return -1;
    }
    if (cudaMalloc(&deviceOutputBuffer, outputSize) != cudaSuccess) {
        std::cerr << "CUDA内存分配失败: deviceOutputBuffer" << std::endl;
        return -1;
    }

    // 创建绑定数组，用于executeV2
    std::vector<void*> bindings = {deviceInputBuffer, deviceOutputBuffer};
    
    cv::VideoCapture cap;
    // 设置解码器
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('H', '2', '6', '4'));
    // 设置缓冲区大小
    cap.set(cv::CAP_PROP_BUFFERSIZE, 3);
    if (!cap.open(videoPath, cv::CAP_FFMPEG)) {
        std::cout << "使用FFMPEG后端打开视频:" << videoPath << "失败\n";
        std::cout << "错误码: " << cap.get(cv::CAP_PROP_FOURCC) << std::endl;
        // 尝试使用默认后端
        if (!cap.open(videoPath, cv::CAP_ANY)) {
            std::cerr << "使用默认后端也无法打开视频\n";
            return -1;
        }
    }

    // 获取视频的基本信息
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    std::cout << "视频FPS: " << fps << "\n";
    std::cout << "总帧数: " << total_frames << "\n";

    cv::Mat frame;
    int frame_count = 0;

    while (cap.read(frame)) {
        auto start = std::chrono::high_resolution_clock::now();
        frame_count++;
        std::cout << "处理第 " << frame_count << " 帧\n";

        // 调整帧大小到模型要求的尺寸
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, cv::Size(360, 360));
        // std::cout << "调整图片帧大小: " << resized_frame.size() << "\n";

        // 转换为float类型并归一化到[0,1]
        cv::Mat float_frame;
        resized_frame.convertTo(float_frame, CV_32F, 1.0 / 255.0);
        // std::cout << "调整图片帧类型: " << float_frame.type() << "\n";

        // 重新排列BGR通道为RGB
        cv::cvtColor(float_frame, float_frame, cv::COLOR_BGR2RGB);
        // std::cout << "重新排列BGR通道为RGB\n";

        // 填充输入数据
        // 假设输入尺寸为 [1, 3, height, width]
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < float_frame.rows; h++) {
                for (int w = 0; w < float_frame.cols; w++) {
                    hostInputBuffer[c * float_frame.rows * float_frame.cols + h * float_frame.cols + w] =
                        float_frame.at<cv::Vec3f>(h, w)[c];
                }
            }
        }

        // 将数据复制到GPU
        cudaMemcpy(deviceInputBuffer, hostInputBuffer.data(), inputSize, cudaMemcpyHostToDevice);

        // 执行推理
        bool status = context->executeV2(bindings.data());
        if (!status) {
            std::cerr << "执行推理失败\n";
            continue;
        }

        // 将结果从GPU复制回CPU
        cudaMemcpy(hostOutputBuffer.data(), deviceOutputBuffer, outputSize, cudaMemcpyDeviceToHost);

        // 处理结果
        // 首先找出最大值(用于数值稳定性)
        float max_val = *std::max_element(hostOutputBuffer.begin(), hostOutputBuffer.end());

        // 计算softmax
        std::vector<float> probabilities(hostOutputBuffer.size());
        float sum_exp = 0.0f;

        std::transform(hostOutputBuffer.begin(), hostOutputBuffer.end(), probabilities.begin(),
            [max_val, &sum_exp](float val) {
                float exp_val = std::exp(val - max_val);
                sum_exp += exp_val;
                return exp_val;
            });

        // 归一化为概率
        for (auto& prob : probabilities) {
            prob /= sum_exp;
        }

        // 找到最大概率及其索引
        auto max_it = std::max_element(probabilities.begin(), probabilities.end());
        int max_idx = std::distance(probabilities.begin(), max_it);
        float max_prob = *max_it;

        std::cout << "帧 " << frame_count << " 预测索引值: " << max_idx
                  << ", 信心值: " << max_prob << "\n";

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "帧 " << frame_count << " 处理耗时: " << duration.count() << "ms\n";
    }

    // 释放资源
    cudaFree(deviceInputBuffer);
    cudaFree(deviceOutputBuffer);
    cap.release();

    return 0;
}
#endif

#ifdef ONNXRUNTIME_AVAILABLE
int process(const std::string &videoPath) {
    std::cout << "使用 ONNX Runtime 进行推理\n";
    std::cout << "ONNX Runtime 版本: " << OrtGetApiBase()->GetVersionString() << std::endl;

    // 创建ONNX Runtime环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx-inference");

    // 创建会话选项
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
#ifdef CUDA_AVAILABLE
        std::cout << "检测到CUDA支持，启用CUDA执行提供程序" << std::endl;
        try {
            OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
            if (status != nullptr) {
                const char* msg = Ort::GetApi().GetErrorMessage(status);
                std::cerr << "启用CUDA执行提供程序失败: " << msg << std::endl;
                Ort::GetApi().ReleaseStatus(status);
            }
        } catch (const Ort::Exception& e) {
            std::cerr << "启用CUDA执行提供程序时出错: " << e.what() << std::endl;
        }
#endif

    // 创建推理会话
    const char* model_path = "models/model.onnx";
    std::cout << "推理模型地址: " << "models/model.onnx" << std::endl;
    Ort::Session session(env, model_path, session_options);

    cv::VideoCapture cap;
    // 设置解码器
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('H', '2', '6', '4'));
    // 设置缓冲区大小
    cap.set(cv::CAP_PROP_BUFFERSIZE, 3);
    if (!cap.open(videoPath, cv::CAP_FFMPEG)) {
        std::cerr << "使用FFMPEG后端打开视频失败:" << cap.get(cv::CAP_PROP_BACKEND) << std::endl;
        // 简单检查 - 尝试获取视频属性
        double width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps = cap.get(cv::CAP_PROP_FPS);

        std::cerr << "尝试获取视频属性结果:" << std::endl;
        std::cerr << "宽度: " << width << std::endl;
        std::cerr << "高度: " << height << std::endl;
        std::cerr << "帧率: " << fps << std::endl;
        // 尝试使用默认后端
        if (!cap.open(videoPath)) {
            std::cerr << "使用默认后端也无法打开视频:" << cap.get(cv::CAP_PROP_BACKEND) << std::endl;
            return -1;
        }
    }

    // 获取视频的基本信息
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    std::cout << "视频FPS: " << fps << "\n";
    std::cout << "总帧数: " << total_frames << "\n";

    cv::Mat frame;
    int frame_count = 0;

    // 获取模型信息
    Ort::AllocatorWithDefaultOptions allocator;

    // 获取输入节点数量和名称
    size_t num_input_nodes = session.GetInputCount();
    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<int64_t> input_node_dims;

    // 获取第一个输入节点的名称和维度
    auto input_name = session.GetInputNameAllocated(0, allocator);
    input_node_names[0] = input_name.get();

    // 逐帧处理视频
    while (cap.read(frame)) {
        auto start = std::chrono::high_resolution_clock::now();
        frame_count++;
        std::cout << "处理第 " << frame_count << " 帧\n";

        // 调整帧大小到模型要求的尺寸
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, cv::Size(360, 360));
        // std::cout << "调整图片帧大小: " << resized_frame.size() << "\n";

        // 转换为float类型并归一化到[0,1]
        cv::Mat float_frame;
        resized_frame.convertTo(float_frame, CV_32F, 1.0 / 255.0);
        // std::cout << "调整图片帧类型: " << float_frame.type() << "\n";

        // 重新排列BGR通道为RGB
        cv::cvtColor(float_frame, float_frame, cv::COLOR_BGR2RGB);
        // 准备输入数据（更高效的HWC到CHW转换）
        std::vector<float> input_tensor_values(1 * 3 * 360 * 360);
        float* tensor_data = input_tensor_values.data();

        // 优化的CHW转换 - 避免通道分离和多次内存复制
        int height = 360, width = 360;
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                const float* pixel = float_frame.ptr<float>(h, w);
                for (int c = 0; c < 3; c++) {
                    tensor_data[c * height * width + h * width + w] = pixel[c];
                }
            }
        }

        // 定义输入形状
        std::vector<int64_t> input_shape = {1, 3, 360, 360};

        // 创建输入tensor
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size());

        // 获取输出节点名称
        size_t num_output_nodes = session.GetOutputCount();
        std::vector<const char*> output_node_names(num_output_nodes);

        auto output_name = session.GetOutputNameAllocated(0, allocator);
        output_node_names[0] = output_name.get();

        // 运行推理
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_node_names.data(),
            &input_tensor, 1,
            output_node_names.data(),
            num_output_nodes);

        // 检查输出
        if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
            std::cerr << "无效的输出张量" << std::endl;
            return -1;
        }

        // 获取输出数据
        float* output_data = output_tensors[0].GetTensorMutableData<float>();

        // 处理输出数据 - 查找最大值和对应索引
        size_t count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        if (count > 0) {
            // 应用softmax并找出最大概率
            std::vector<float> probs(count);
            float sum_exp = 0.0f;
            float max_val = *std::max_element(output_data, output_data + count);

            for (size_t i = 0; i < count; i++) {
                probs[i] = std::exp(output_data[i] - max_val);
                sum_exp += probs[i];
            }

            for (size_t i = 0; i < count; i++) {
                probs[i] /= sum_exp;
            }

            // 找到最大概率及其索引
            auto max_it = std::max_element(probs.begin(), probs.end());
            int max_idx = std::distance(probs.begin(), max_it);
            float max_prob = *max_it;

            std::cout << "帧 " << frame_count << " 预测索引值: " << max_idx
                      << ", 信心值: " << max_prob << "\n";
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "帧 " << frame_count << " 处理耗时: " << duration.count() << "ms\n";
    }

    return 0;
}
#endif

#ifdef LIBTORCH_AVAILABLE
int process(const std::string &videoPath) {
    std::cout << "使用 LibTorch 推理\n";
    std::cout << "LibTorch版本: "
              << TORCH_VERSION_MAJOR << "."
              << TORCH_VERSION_MINOR << "."
              << TORCH_VERSION_PATCH << std::endl;

    try {
        // 设置线程数，避免过度使用
        torch::set_num_interop_threads(4);
        torch::set_num_threads(4);

        // 加载模型
        torch::jit::script::Module module;
        try {
            // 确保路径正确
            module = torch::jit::load("models/model.pt");
            std::cout << "推理模型地址: " << "models/model.pt" << std::endl;
            for (const auto &param: module.parameters()) {
                // std::cout << "Parameter device: " << param.device() << "\n";
            }
        }
        catch (const c10::Error &e) {
            std::cerr << "模型加载失败: " << e.msg() << "\n";
            return -1;
        }

        // 设置为评估模式
        module.eval();

        // 检查可用的设备并相应地移动模型和输入
        if (torch::cuda::is_available()) {
            // 将模型移到 CUDA 设备
            module.to(torch::kCUDA);
            std::cout << "使用 CUDA 设备\n";
        } else if (torch::mps::is_available()) {
            // 将模型移到 MPS 设备
            module.to(torch::kMPS);
            std::cout << "使用 MPS 设备\n";
        } else {
            std::cout << "使用 CPU 设备\n";
        }

        cv::VideoCapture cap;
        // 设置解码器
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('H', '2', '6', '4'));
        // 设置缓冲区大小
        cap.set(cv::CAP_PROP_BUFFERSIZE, 3);
        if (!cap.open(videoPath, cv::CAP_FFMPEG)) {
            std::cerr << "使用FFMPEG后端打开视频失败\n";
            // 尝试使用默认后端
            if (!cap.open(videoPath)) {
                std::cerr << "使用默认后端也无法打开视频\n";
                return -1;
            }
        }

        // 获取视频的基本信息
        double fps = cap.get(cv::CAP_PROP_FPS);
        int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
        std::cout << "视频FPS: " << fps << "\n";
        std::cout << "总帧数: " << total_frames << "\n";

        cv::Mat frame;
        int frame_count = 0;

        // 逐帧处理视频
        while (cap.read(frame)) {
            auto start = std::chrono::high_resolution_clock::now();
            frame_count++;
            std::cout << "处理第 " << frame_count << " 帧\n";

            // 调整帧大小到模型要求的尺寸
            cv::Mat resized_frame;
            cv::resize(frame, resized_frame, cv::Size(360, 360));
            // std::cout << "调整图片帧大小: " << resized_frame.size() << "\n";

            // 转换为float类型并归一化到[0,1]
            cv::Mat float_frame;
            resized_frame.convertTo(float_frame, CV_32F, 1.0 / 255.0);
            // std::cout << "调整图片帧类型: " << float_frame.type() << "\n";

            // 转换为torch tensor
            auto input_tensor = torch::from_blob(float_frame.data, {1, 360, 360, 3}, torch::kFloat32);
            // std::cout << "输入张量加载图片帧大小: " << input_tensor.sizes() << "\n";

            // 检查可用的设备并相应地移动模型和输入
            if (torch::cuda::is_available()) {
                // 将输入张量移到 CUDA 设备
                input_tensor = input_tensor.to(torch::kCUDA);
            } else if (torch::mps::is_available()) {
                // 将输入张量移到 MPS 设备
                input_tensor = input_tensor.to(torch::kMPS);
            }
            // std::cout << "输入张量处理设备: " << input_tensor.device() << "\n";

            // 检查张量的值范围
            input_tensor = input_tensor.contiguous();
            // std::cout << "张量最小值: " << input_tensor.min().item<float>() << "\n";
            // std::cout << "张量最大值: " << input_tensor.max().item<float>() << "\n";

            // 调整通道顺序从HWC到CHW
            input_tensor = input_tensor.permute({0, 3, 1, 2});

            // 准备输入数据并执行推理
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);

            // 执行推理
            auto output = module.forward(inputs).toTensor();

            // 应用softmax处理
            auto softmax_output = torch::softmax(output, 1);  // 沿类别维度应用softmax

            // 获取预测结果
            auto max_result = softmax_output.max(1);
            auto max_index = std::get<1>(max_result).item<int64_t>();
            auto probability = std::get<0>(max_result).item<float>();

            std::cout << "帧 " << frame_count << " 预测索引值: " << max_index
                      << ", 信心值: " << probability << "\n";

            // 可选：添加处理间隔，控制处理速度
            // cv::waitKey(1);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "帧 " << frame_count << " 处理耗时: " << duration.count() << "ms\n";
        }

        // 释放视频资源
        cap.release();
    }
    catch (const c10::Error &e) {
        std::cerr << "LibTorch 执行出错: " << e.msg() << "\n";
        return -1;
    }

    return 0;
}
#endif

int main() {
    try {
        printSystemArchitecture();
        std::cout << "OpenCV版本: " << CV_VERSION << std::endl;
        std::cout << cv::getBuildInformation() << std::endl;
        std::regex pattern(R"(FFMPEG:\s+YES)", std::regex_constants::icase);
        std::cout << "OpenCV对FFMPEG支持: " << (std::regex_search(cv::getBuildInformation(), pattern)) << std::endl;
        std::cout << "获取FFMPEG版本:" << std::endl;
        int result = system("ffmpeg -version | head -n 1");
        if (result != 0) {
            std::cout << "无法执行ffmpeg命令或获取版本信息" << std::endl;
        }
        // 获取视频文件
        std::string videoPath = std::filesystem::absolute("datasets/validate/test_video.mp4").string();
        std::cout << "推理视频地址: " << "datasets/validate/test_video.mp4" << std::endl;
        process(videoPath);
    }
    catch (const std::exception &e) {
        std::cerr << "执行出错: " << e.what() << "\n";
        return -1;
    }

    return 0;
}