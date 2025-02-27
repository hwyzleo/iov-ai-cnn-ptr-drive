cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(iov-ai-cnn-ptr-drive)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# 强制使用 C++17 标准
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
endif ()

# 查找 OpenCV
find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found, building suspend")
endif ()

# 查找推理库

# 优先查找 TensorRT
find_package(TensorRT QUIET)
if (TensorRT_FOUND)
    set(FOUND_INFERENCING_LIB ON)
    message(STATUS "Found TensorRT: ${TensorRT_LIBRARIES}")
    add_definitions(-DTENSORRT_AVAILABLE)
    set(EXTRA_LIBS ${EXTRA_LIBS} ${TensorRT_LIBRARIES} nvonnxparser)
endif ()

# 其次查找 ONNX Runtime
if (NOT FOUND_INFERENCING_LIB)
    find_package(onnxruntime QUIET)
    if (onnxruntime_FOUND AND DEFINED onnxruntime_LIBRARIES)
        message(STATUS "Found ONNX Runtime: ${onnxruntime_LIBRARIES}")
        add_definitions(-DONNXRUNTIME_AVAILABLE)
        add_definitions(-DONNX_NO_SCHEMA_REGISTRATIONS=1)
        add_compile_definitions(ONNX_ML=1)
        set(EXTRA_LIBS ${EXTRA_LIBS} ${ONNXRUNTIME_LIBRARIES})
        set(FOUND_INFERENCING_LIB ON)
    elseif (DEFINED ENV{ONNXRUNTIME_HOME})
        set(ONNXRUNTIME_HOME $ENV{ONNXRUNTIME_HOME})
        message(STATUS "Found ENV ONNXRUNTIME_HOME: ${ONNXRUNTIME_HOME}")
        add_definitions(-DONNXRUNTIME_AVAILABLE)
        add_definitions(-DONNX_NO_SCHEMA_REGISTRATIONS=1)
        add_compile_definitions(ONNX_ML=1)
        set(ONNXRUNTIME_DIR "${ONNXRUNTIME_HOME}")
        message(STATUS "Using ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")
        set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_DIR}/include")
        # 添加头文件路径
        include_directories(${ONNXRUNTIME_INCLUDE_DIRS})
        message(STATUS "Add ONNXRUNTIME_INCLUDE_DIRS: ${ONNXRUNTIME_INCLUDE_DIRS}")
        set(ONNXRUNTIME_LIB_DIRS "${ONNXRUNTIME_DIR}/lib")
        find_library(ONNXRUNTIME_LIBRARY onnxruntime PATHS ${ONNXRUNTIME_LIB_DIRS} REQUIRED)
        set(EXTRA_LIBS ${EXTRA_LIBS} ${ONNXRUNTIME_LIBRARY})
        message(STATUS "Add ONNXRUNTIME_LIB_DIRS: ${ONNXRUNTIME_LIB_DIRS}")
        set(FOUND_INFERENCING_LIB ON)
    endif ()
endif ()

# 最后查找 LibTorch
if (NOT FOUND_INFERENCING_LIB)
    if (DEFINED ENV{TORCH_HOME})
        set(TORCH_HOME $ENV{TORCH_HOME})
        message(STATUS "Found ENV TORCH_HOME: ${TORCH_HOME}")
        set(Torch_DIR "${TORCH_HOME}/share/cmake/Torch")
        message(STATUS "Using Torch_DIR: ${Torch_DIR}")
    endif ()
    find_package(Torch QUIET)
    if (Torch_FOUND)
        message(STATUS "Found LibTorch: ${TORCH_LIBRARIES}")
        add_definitions(-DLIBTORCH_AVAILABLE)
        list(APPEND CMAKE_PREFIX_PATH "${TORCH_HOME}")
        message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
        set(EXTRA_LIBS ${EXTRA_LIBS} ${TORCH_LIBRARIES})
        include_directories(${TORCH_INCLUDE_DIRS})
        set(FOUND_INFERENCING_LIB ON)
    endif ()
endif ()

# 没找到推理库
if (NOT FOUND_INFERENCING_LIB)
    message(FATAL_ERROR "No inferencing lib found, building suspend")
endif ()

# 添加可执行文件
add_executable(ptr src/main.cpp)

# 链接库
target_link_libraries(ptr
        ${OpenCV_LIBS}
        ${EXTRA_LIBS}
        )

# 针对 M1 的特殊设置
if (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -O2")
    set_property(TARGET ptr PROPERTY CXX_STANDARD 17)
endif ()