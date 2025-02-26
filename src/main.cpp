#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <filesystem>

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

        // 打开视频文件
        std::string videoPath = std::filesystem::absolute("datasets/validate/test_video.mp4").string();
        cv::VideoCapture cap;
        // 设置硬件加速
#if CV_VERSION_MAJOR >= 4
        // OpenCV 4.0及以上版本支持硬件加速
    cap.set(cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY);
#endif
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
        std::cerr << "执行出错: " << e.msg() << "\n";
        return -1;
    }

    return 0;
}