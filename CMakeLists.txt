cmake_minimum_required(VERSION 3.1)
project(example)

message("-- CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}")
message("-- CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}")
message("-- @CMAKE_CURRENT_LIST_DIR ${CMAKE_CURRENT_LIST_DIR}")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
message("-- CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}")

# libigl
option(LIBIGL_WITH_STATIC_LIBRARY    "Use libigl as static library" OFF)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_TRIANGLE           "Use Triangle"      ON)
# option(LIBIGL_WITH_OPENGL_GLFW_IMGUI        "Use ImgGui"      ON)

set(CGAL_DIR "/home/server/MaJing/cpp/CGAL-5.1-beta2")
find_package(CGAL REQUIRED)
find_package(LIBIGL REQUIRED)

# Add your project files
# file(GLOB SRCFILES bbw.cpp)
# add_executable(main bbw.cpp)
# target_link_libraries(main igl::core igl::opengl igl::opengl_glfw)

# find_package(Boost REQUIRED)
# add_executable(main template.cpp)

find_package(OpenCV REQUIRED)

include_directories(src/)
file(GLOB SRC src/*.cpp)
add_executable(main render.cpp ${SRC})
target_link_libraries(main igl::core CGAL::CGAL ${OpenCV_LIBRARIES})

# add_executable(main test.cpp)