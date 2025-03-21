cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(iov-ai-cnn-ptr-drive)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include_directories(SYSTEM /usr/include)
# 强制使用 C++17 标准
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
endif ()

# 查找 OpenCV
if (DEFINED OpenCV_INCLUDE_DIR AND DEFINED OpenCV_LIB_DIR)
    include_directories(${OpenCV_INCLUDE_DIR})
    message(STATUS "添加 OpenCV INCLUDE 目录: ${OpenCV_INCLUDE_DIR}")
    set(OpenCV_LIBS
            ${OpenCV_LIB_DIR}/libopencv_core.so
            ${OpenCV_LIB_DIR}/libopencv_highgui.so
            ${OpenCV_LIB_DIR}/libopencv_videoio.so
            ${OpenCV_LIB_DIR}/libopencv_imgproc.so
            )
    message(STATUS "添加 OpenCV LIB 目录: ${OpenCV_LIB_DIR}")
    if (DEFINED FFMPEG_INCLUDE_DIR AND DEFINED FFMPEG_LIB_DIR)
        include_directories(${FFMPEG_INCLUDE_DIR})
        message(STATUS "添加 FFMPEG INCLUDE 目录: ${FFMPEG_INCLUDE_DIR}")
        set(FFMPEG_LIBS
                ${FFMPEG_LIB_DIR}/libavcodec.so
                ${FFMPEG_LIB_DIR}/libavdevice.so
                ${FFMPEG_LIB_DIR}/libavfilter.so
                ${FFMPEG_LIB_DIR}/libavformat.so
                ${FFMPEG_LIB_DIR}/libavresample.so
                ${FFMPEG_LIB_DIR}/libavutil.so
                ${FFMPEG_LIB_DIR}/libpostproc.so
                ${FFMPEG_LIB_DIR}/libswresample.so
                ${FFMPEG_LIB_DIR}/libswscale.so
                )
        set(EXTRA_LIBS ${EXTRA_LIBS} ${FFMPEG_LIBS})
        message(STATUS "添加 FFMPEG LIB 目录: ${FFMPEG_LIB_DIR}")
    endif ()
    if (DEFINED X264_INCLUDE_DIR AND DEFINED X264_LIB_DIR)
        include_directories(${X264_INCLUDE_DIR})
        message(STATUS "添加 X264 INCLUDE 目录: ${X264_INCLUDE_DIR}")
        set(X264_LIBS
                ${X264_LIB_DIR}/libx264.so
                )
        set(EXTRA_LIBS ${EXTRA_LIBS} ${X264_LIBS})
        message(STATUS "添加 X264 LIB 目录: ${X264_LIB_DIR}")
    endif ()
else ()
    find_package(OpenCV)
    if (NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV没找到，构建中止")
    endif ()
endif ()


# 查找推理库
# 优先查找 TensorRT
if (DEFINED ENV{TENSORRT_HOME})
    message(STATUS "环境变量发现 TENSORRT_HOME: ${TENSORRT_HOME}")
    set(TENSORRT_INCLUDE_DIR "$ENV{TENSORRT_HOME}/include")
    set(TENSORRT_LIB_DIR "$ENV{TENSORRT_HOME}/lib")
endif ()
if (NOT FOUND_INFERENCING_LIB AND DEFINED TENSORRT_INCLUDE_DIR AND DEFINED TENSORRT_LIB_DIR)
    add_definitions(-DTENSORRT_AVAILABLE)
    include_directories(${TENSORRT_INCLUDE_DIR})
    message(STATUS "添加 TENSORRT INCLUDE 目录: ${TENSORRT_INCLUDE_DIR}")
    set(TENSORRT_LIBS
            ${TENSORRT_LIB_DIR}/libnvinfer.so
            ${TENSORRT_LIB_DIR}/libnvonnxparser.so
            ${TENSORRT_LIB_DIR}/libnvinfer_plugin.so
            ${TENSORRT_LIB_DIR}/libnvdla_compiler.so
            ${TENSORRT_LIB_DIR}/libnvmedia_dla.so
            ${TENSORRT_LIB_DIR}/libnvmedia_tensor.so
            ${TENSORRT_LIB_DIR}/libnvscibuf.so
            ${TENSORRT_LIB_DIR}/libnvrm_mem.so
            ${TENSORRT_LIB_DIR}/libnvrm_gpu.so
            ${TENSORRT_LIB_DIR}/libnvrm_host1x.so
            ${TENSORRT_LIB_DIR}/libnvrm_sync.so
            ${TENSORRT_LIB_DIR}/libnvdla_runtime.so
            ${TENSORRT_LIB_DIR}/libnvscicommon.so
            ${TENSORRT_LIB_DIR}/libnvtegrahv.so
            ${TENSORRT_LIB_DIR}/libnvsocsys.so
            ${TENSORRT_LIB_DIR}/libnvsciipc.so
            ${TENSORRT_LIB_DIR}/libnvos.so
            ${TENSORRT_LIB_DIR}/libnvidia-eglcore.so
            ${TENSORRT_LIB_DIR}/libEGL_nvidia.so.0
            ${TENSORRT_LIB_DIR}/libnvidia-glsi.so
            ${TENSORRT_LIB_DIR}/libnvidia-rmapi-tegra.so
            )
    set(EXTRA_LIBS ${EXTRA_LIBS} ${TENSORRT_LIBS})
    message(STATUS "添加 TENSORRT LIB 目录: ${TENSORRT_LIB_DIR}")
    set(FOUND_INFERENCING_LIB ON)
    if (DEFINED ENV{CUDA_HOME})
        message(STATUS "环境变量发现 CUDA_HOME: ${CUDA_HOME}")
        set(CUDA_INCLUDE_DIR "$ENV{CUDA_HOME}/include")
        set(CUDA_LIB_DIR "$ENV{CUDA_HOME}/lib")
    endif ()
    if (DEFINED CUDA_INCLUDE_DIR AND DEFINED CUDA_LIB_DIR)
        add_definitions(-DCUDA_AVAILABLE)
        include_directories(${CUDA_INCLUDE_DIR})
        message(STATUS "添加 CUDA INCLUDE 目录: ${CUDA_INCLUDE_DIR}")
        set(CUDA_LIBS
                ${CUDA_LIB_DIR}/libcublasLt.so
                ${CUDA_LIB_DIR}/libcublas.so
                ${CUDA_LIB_DIR}/libcudart.so
                )
        set(EXTRA_LIBS ${EXTRA_LIBS} ${CUDA_LIBS})
        message(STATUS "添加 CUDA LIB 目录: ${CUDA_LIB_DIR}")
        set(CMAKE_INSTALL_RPATH "${CUDA_LIB_DIR}")
        set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
        set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
        if (DEFINED CUDNN_INCLUDE_DIR AND DEFINED CUDNN_LIB_DIR)
            include_directories(${CUDNN_INCLUDE_DIR})
            message(STATUS "添加 cuDNN INCLUDE 目录: ${CUDNN_INCLUDE_DIR}")
            set(CUDNN_LIBS
                    ${CUDNN_LIB_DIR}/libcudnn.so
                    )
            set(EXTRA_LIBS ${EXTRA_LIBS} ${CUDNN_LIBS})
            message(STATUS "添加 cuDNN LIB 目录: ${CUDNN_LIB_DIR}")
        endif ()
    else ()
        message(FATAL_ERROR "TensorRT存在但CUDA不存在，构建中止")
    endif ()
endif ()
# 其次查找 ONNX Runtime
if (NOT FOUND_INFERENCING_LIB AND DEFINED ENV{ONNXRUNTIME_HOME})
    set(ONNXRUNTIME_HOME $ENV{ONNXRUNTIME_HOME})
    message(STATUS "环境变量发现 ONNXRUNTIME_HOME: ${ONNXRUNTIME_HOME}")
    add_definitions(-DONNXRUNTIME_AVAILABLE)
    set(ONNXRUNTIME_DIR "${ONNXRUNTIME_HOME}")
    message(STATUS "设置 ONNXRUNTIME 根目录: ${ONNXRUNTIME_DIR}")
    set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_DIR}/include")
    include_directories(${ONNXRUNTIME_INCLUDE_DIRS})
    message(STATUS "添加 ONNXRUNTIME INCLUDE 目录: ${ONNXRUNTIME_INCLUDE_DIRS}")
    set(ONNXRUNTIME_LIB_DIRS "${ONNXRUNTIME_DIR}/lib")
    find_library(ONNXRUNTIME_LIBRARY onnxruntime PATHS ${ONNXRUNTIME_LIB_DIRS} REQUIRED)
    message(STATUS "添加 ONNXRUNTIME LIB 目录: ${ONNXRUNTIME_LIB_DIRS}")
    set(EXTRA_LIBS ${EXTRA_LIBS} ${ONNXRUNTIME_LIBRARY})
    if (DEFINED ENV{CUDA_HOME})
        find_package(CUDA REQUIRED)
        include_directories(${CUDA_INCLUDE_DIRS})
        set(CUDA_HOME $ENV{CUDA_HOME})
        message(STATUS "环境变量发现 CUDA_HOME: ${CUDA_HOME}")
        add_definitions(-DCUDA_AVAILABLE)
        set(CUDA_DIR "${CUDA_HOME}")
        message(STATUS "设置 CUDA 根目录: ${CUDA_DIR}")
        set(CUDA_INCLUDE_DIRS "${CUDA_DIR}/include")
        message(STATUS "添加 CUDA INCLUDE 目录: ${CUDA_INCLUDE_DIRS}")
        set(CUDA_LIB_DIRS "${CUDA_DIR}/lib64")
        set(CMAKE_INSTALL_RPATH "${CUDA_LIB_DIRS}")
        set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
        set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
        message(STATUS "添加 CUDA LIB 目录: ${CUDA_LIB_DIRS}")
        set(EXTRA_LIBS ${EXTRA_LIBS} ${CUDA_LIBRARIES})
        message(STATUS "CUDA库路径: ${CUDA_LIBRARIES}")
        set(EXTRA_LIBS ${EXTRA_LIBS} ${CUDA_CUBLAS_LIBRARIES})
        message(STATUS "CUDA BLAS库路径: ${CUDA_CUBLAS_LIBRARIES}")
        find_library(CUDA_CUBLASLT_LIBRARY cublasLt PATHS ${CUDA_LIB_DIRS} REQUIRED)
        set(EXTRA_LIBS ${EXTRA_LIBS} ${CUDA_CUBLASLT_LIBRARY})
        message(STATUS "CUDA BLASLT库路径: ${CUDA_CUBLASLT_LIBRARY}")
        set(EXTRA_LIBS ${EXTRA_LIBS} ${CUDA_CUDART_LIBRARY})
        message(STATUS "CUDA DART库路径: ${CUDA_CUDART_LIBRARY}")
#        find_library(CUDA_CUDNN_LIBRARY cudnn PATHS ${CUDA_LIB_DIRS} REQUIRED)
#        set(EXTRA_LIBS ${EXTRA_LIBS} ${CUDA_CUDNN_LIBRARY})
#        message(STATUS "CUDA DNN库路径: ${CUDA_CUDNN_LIBRARY}")
        find_library(CUDA_CUFFT_LIBRARY cufft PATHS ${CUDA_LIB_DIRS} REQUIRED)
        set(EXTRA_LIBS ${EXTRA_LIBS} ${CUDA_CUFFT_LIBRARY})
        message(STATUS "CUDA FFT库路径: ${CUDA_CUFFT_LIBRARY}")
        find_library(CUDA_CURAND_LIBRARY curand PATHS ${CUDA_LIB_DIRS} REQUIRED)
        set(EXTRA_LIBS ${EXTRA_LIBS} ${CUDA_CURAND_LIBRARY})
        message(STATUS "CUDA RAND库路径: ${CUDA_CURAND_LIBRARY}")
    endif ()
    set(FOUND_INFERENCING_LIB ON)
endif ()
# 最后查找 LibTorch
if (NOT FOUND_INFERENCING_LIB AND DEFINED ENV{TORCH_HOME})
    set(TORCH_HOME $ENV{TORCH_HOME})
    message(STATUS "环境变量发现 TORCH_HOME: ${TORCH_HOME}")
    set(Torch_DIR "${TORCH_HOME}/share/cmake/Torch")
    message(STATUS "设置 Torch 根目录: ${Torch_DIR}")
    find_package(Torch QUIET)
    if (Torch_FOUND)
        message(STATUS "发现 LibTorch 库文件: ${TORCH_LIBRARIES}")
        add_definitions(-DLIBTORCH_AVAILABLE)
        list(APPEND CMAKE_PREFIX_PATH "${TORCH_HOME}")
        set(EXTRA_LIBS ${EXTRA_LIBS} ${TORCH_LIBRARIES})
        include_directories(${TORCH_INCLUDE_DIRS})
        set(FOUND_INFERENCING_LIB ON)
    endif ()
endif ()

# 没找到推理库
if (NOT FOUND_INFERENCING_LIB)
    message(FATAL_ERROR "没有发现可用推理库，构建中止")
endif ()

# 添加可执行文件
add_executable(ptr src/main.cpp)

# 链接库
target_link_libraries(ptr
        PRIVATE dl
        ${OpenCV_LIBS}
        ${EXTRA_LIBS}
        )

# 针对 M1 的特殊设置
if (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -O2")
    set_property(TARGET ptr PROPERTY CXX_STANDARD 17)
endif ()