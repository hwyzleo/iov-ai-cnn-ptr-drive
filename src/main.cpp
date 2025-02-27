#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <filesystem>
#include <stdlib.h>

#ifdef TENSORRT_AVAILABLE
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <nvdrv/system.h>
#endif

#ifdef ONNXRUNTIME_AVAILABLE
#include <onnxruntime_cxx_api.h>
#endif

#ifdef LIBTORCH_AVAILABLE
#include <torch/script.h>
#include <torch/torch.h>
#endif

#ifdef TENSORRT_AVAILABLE
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;

int process(const std::string &videoPath) {
    std::cout << "使用 TensorRT 推理\n";
    // TensorRT实现
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

    // 创建ONNX解析器
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

    // 解析ONNX文件
    parser->parseFromFile("models/model.onnx", static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

    // 构建引擎
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 20);
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    // 创建执行上下文
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    // 视频处理代码继续...
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        throw std::runtime_error("无法打开视频文件");
    }

    // 分配输入和输出缓冲区
    void* buffers[2];
    int inputIndex = engine->getBindingIndex("input");
    int outputIndex = engine->getBindingIndex("output");

    // 获取输入输出维度
    auto inputDims = engine->getBindingDimensions(inputIndex);
    auto outputDims = engine->getBindingDimensions(outputIndex);

    // 分配GPU内存
    cudaMalloc(&buffers[inputIndex], batchSize * 3 * 360 * 360 * sizeof(float));
    cudaMalloc(&buffers[outputIndex], batchSize * numClasses * sizeof(float));

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int frame_count = 0;
    cv::Mat frame;
    while (cap.read(frame)) {
        auto start = std::chrono::high_resolution_clock::now();
        frame_count++;

        // 预处理图像
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, cv::Size(360, 360));
        cv::Mat float_frame;
        resized_frame.convertTo(float_frame, CV_32F, 1.0/255.0);

        // 将数据复制到GPU
        cudaMemcpyAsync(buffers[inputIndex], float_frame.data,
                       inputDims.d[0] * inputDims.d[1] * inputDims.d[2] * inputDims.d[3] * sizeof(float),
                       cudaMemcpyHostToDevice, stream);

        // 执行推理
        context->enqueueV2(buffers, stream, nullptr);

        // 分配输出内存并从GPU复制结果
        std::vector<float> output(outputDims.d[1]);
        cudaMemcpyAsync(output.data(), buffers[outputIndex],
                       outputDims.d[1] * sizeof(float),
                       cudaMemcpyDeviceToHost, stream);

        // 等待处理完成
        cudaStreamSynchronize(stream);

        // 找到最大概率的类别
        auto max_element = std::max_element(output.begin(), output.end());
        int max_index = std::distance(output.begin(), max_element);
        float probability = *max_element;

        std::cout << "帧 " << frame_count << " 预测索引值: " << max_index
                  << ", 信心值: " << probability << "\n";

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "帧 " << frame_count << " 处理耗时: " << duration.count() << "ms\n";
    }

    // 清理资源
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
    cap.release();

    return 0;
}
#endif

#ifdef ONNXRUNTIME_AVAILABLE
int process(const std::string &videoPath) {
    std::cout << "使用 ONNX Runtime 推理\n";
    // 创建ONNX Runtime环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx-inference");

    // 创建会话选项
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // 设置使用的执行提供程序（默认为CPU）
    // 如果要使用CUDA，可以添加：
    // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);

    // 创建推理会话
    const char* model_path = "models/model.onnx";
    Ort::Session session(env, model_path, session_options);

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
        cv::Mat channels[3];
        cv::split(float_frame, channels);

        // 准备输入数据 (NCHW格式)
        size_t input_tensor_size = 1 * 3 * 360 * 360;
        std::vector<float> input_tensor_values(input_tensor_size);

        // 填充数据 (CHW格式)
        int channel_length = 360 * 360;
        for (int c = 0; c < 3; c++) {
            cv::Mat channel = channels[2 - c]; // BGR to RGB
            std::memcpy(input_tensor_values.data() + c * channel_length,
                       channel.data, channel_length * sizeof(float));
        }

        // 定义输入形状
        std::vector<int64_t> input_shape = {1, 3, 360, 360};

        // 创建输入tensor
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_size,
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

        // 获取输出的形状
//        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
//        std::cout << "输出形状: ";
//        for (size_t i = 0; i < output_shape.size(); i++) {
//            std::cout << output_shape[i] << " ";
//        }
//        std::cout << std::endl;

        // 处理输出数据 - 查找最大值和对应索引
        size_t count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        if (count > 0) {
            float max_val = output_data[0];
            int max_idx = 0;

            for (size_t i = 1; i < count; i++) {
                if (output_data[i] > max_val) {
                    max_val = output_data[i];
                    max_idx = i;
                }
            }
            std::cout << "帧 " << frame_count << " 预测索引值: " << max_idx
                      << ", 信心值: " << max_val << "\n";
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

    try {
        // 设置线程数，避免过度使用
        torch::set_num_interop_threads(4);
        torch::set_num_threads(4);

        // 加载模型
        torch::jit::script::Module module;
        try {
            // 确保路径正确
            module = torch::jit::load("models/model.pt");
            std::cout << "模型加载成功\n";
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

            // 获取预测结果
            auto max_result = output.max(1);
            auto max_index = std::get < 1 > (max_result).item<int64_t>();
            auto probability = std::get < 0 > (max_result).item<float>();

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
        // 获取视频文件
        std::string videoPath = std::filesystem::absolute("datasets/validate/test_video.mp4").string();
        process(videoPath);
    }
    catch (const std::exception &e) {
        std::cerr << "执行出错: " << e.what() << "\n";
        return -1;
    }

    return 0;
}