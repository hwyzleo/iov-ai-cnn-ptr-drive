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

# 查找 LibTorch
set(TORCH_HOME $ENV{TORCH_HOME})
message(STATUS "Using TORCH_HOME: ${TORCH_HOME}")
set(Torch_DIR "${TORCH_HOME}/share/cmake/Torch")
message(STATUS "Using Torch_DIR: ${Torch_DIR}")
list(APPEND CMAKE_PREFIX_PATH "${TORCH_HOME}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
if (Torch_FOUND)
    message(STATUS "Found Torch: ${TORCH_LIBRARIES}")
else ()
    message(FATAL_ERROR "Failed to find Torch configuration file.")
endif ()
include_directories(${TORCH_INCLUDE_DIRS})

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 添加可执行文件
add_executable(ptr src/main.cpp)

# 链接库
target_link_libraries(ptr
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        )

# 针对 M1 的特殊设置
if (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -O2")
    set_property(TARGET ptr PROPERTY CXX_STANDARD 17)
endif ()