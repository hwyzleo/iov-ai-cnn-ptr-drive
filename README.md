# iov-ai-cnn-ptr-drive
从零开始练手车联网人工智能CNN路面类型识别针对 NVIDIA DRIVE 的 C++ 实现

# 相关性能
## Mac M1 + 64G
### ARM64 + OpenCV 4.11.0 + FFMPEG 7.1 + LibTorch 2.6.0 + MPS
推理耗时 ≈ 15ms
### ARM64 + OpenCV 4.11.0 + FFMPEG 7.1 + ONNX 1.20.1 + CPU
推理耗时 ≈ 85ms
## TencentOS Server NVIDIA V100 + 40G
### x86_64 (64位) + OpenCV 3.4.6 + FFMPEG 4.2.10 + TensorRT 10.1 GA + CUDA 12.4
推理耗时 ≈ 8ms
### x86_64 (64位) + OpenCV 3.4.6 + FFMPEG 4.2.10 + ONNX 1.20.1 + CPU
推理耗时 ≈ 160ms
### x86_64 (64位) + OpenCV 3.4.6 + FFMPEG 4.2.10 + LibTorch 2.6.0 + CPU
推理耗时 ≈ 100ms
### x86_64 (64位) + OpenCV 3.4.6 + FFMPEG 4.2.10 + LibTorch 2.6.0 + CUDA 12.4
推理耗时 ≈ 8ms
## Jetson Orin Nano + 4G
### ARM64 + OpenCV 4.2.0 + FFMPEG 4.2.7 + ONNX 1.20.1 + CPU
推理耗时 ≈ 270ms