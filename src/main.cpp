#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>

int main() {
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

        // 加载图片
        cv::Mat image = cv::imread("datasets/validate/test_image.png");
        if (image.empty()) {
            std::cerr << "图片读取失败.\n";
            return -1;
        }
        std::cout << "图片加载成功\n";
        std::cout << "原始图片大小: " << image.size() << "\n";

        // 调整图片大小到模型要求的尺寸
        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size(360, 360));
        std::cout << "调整图片大小: " << resized_image.size() << "\n";

        // 转换为float类型并归一化到[0,1]
        cv::Mat float_image;
        resized_image.convertTo(float_image, CV_32F, 1.0 / 255.0);
        std::cout << "调整图片类型: " << float_image.type() << "\n";

        // 转换为torch tensor
        auto input_tensor = torch::from_blob(float_image.data, {1, 360, 360, 3}, torch::kFloat32);
        std::cout << "输入张量加载图片大小: " << input_tensor.sizes() << "\n";
        input_tensor = input_tensor.contiguous();
        std::cout << "输入张量处理设备: " << input_tensor.device() << "\n";

        // 检查可用的设备并相应地移动模型和输入
        if (torch::cuda::is_available()) {
            // 将模型移到 CUDA 设备
            module.to(torch::kCUDA);
            // 将输入张量移到 CUDA 设备
            input_tensor = input_tensor.to(torch::kCUDA);
            std::cout << "使用 CUDA 设备\n";
        } else if (torch::mps::is_available()) {
            // 将模型移到 MPS 设备
            module.to(torch::kMPS);
            // 将输入张量移到 MPS 设备
            input_tensor = input_tensor.to(torch::kMPS);
            std::cout << "使用 MPS 设备\n";
        } else {
            std::cout << "使用 CPU 设备\n";
        }
        // 检查张量的值范围
        std::cout << "张量最小值: " << input_tensor.min().item<float>() << "\n";
        std::cout << "张量最大值: " << input_tensor.max().item<float>() << "\n";

        // 调整通道顺序从HWC到CHW
        input_tensor = input_tensor.permute({0, 3, 1, 2});
        std::cout << "调整通道顺序\n";

        // 准备输入数据
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        std::cout << "准备输入数据\n";

        // 执行推理
        auto output = module.forward(inputs).toTensor();
        std::cout << "执行推理\n";

        // 获取预测结果
        auto max_result = output.max(1);
        auto max_index = std::get < 1 > (max_result).item<int64_t>();
        auto probability = std::get < 0 > (max_result).item<float>();

        std::cout << "预测索引值: " << max_index << "\n";
        std::cout << "信心值: " << probability << "\n";
    }
    catch (const c10::Error &e) {
        std::cerr << "执行出错: " << e.msg() << "\n";
        return -1;
    }

    return 0;
}