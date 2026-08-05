parabix_add_executable(
NAME
    demo
SRC
    demo.cpp
DEPS
    image
    kernel.basis
    kernel.io
    kernel.streamutils
)

parabix_add_executable(
NAME
    image_benchmark
SRC
    benchmark.cpp
DEPS
    image
    kernel.io
    kernel.streamutils
)

target_include_directories(image_benchmark PRIVATE ${PROJECT_SOURCE_DIR}/lib/image)
