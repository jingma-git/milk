cmake_minimum_required(VERSION 3.1)
project(example)

set(CMAKE_BUILD_TYPE RELEASE)

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

find_package(LIBIGL REQUIRED)

# set(CGAL_DIR "/home/server/MaJing/cpp/CGAL-5.1-beta2")
# find_package(CGAL REQUIRED)

# find_package(Boost REQUIRED)

# find_package(OpenCV REQUIRED)

# --------------------------------------Add your project files--------------------------------
# ---------------------------Bounded Bihamonic Weights
# include_directories(libigl/include/)
# add_executable(main bbw.cpp)
# target_link_libraries(main igl::core igl::opengl igl::opengl_glfw)


# # ---------------------------Arap
# include_directories(libigl/include/)
# add_executable(main arap.cpp)
# target_link_libraries(main igl::core igl::opengl igl::opengl_glfw)

# ---------------------------'template' and 'typename' keywords
# add_executable(main template.cpp)

# --------------------------- optimal bounding box
# include_directories(src/)
# file(GLOB SRC src/*.cpp)
# add_executable(main view_obb.cpp ${SRC})
# target_link_libraries(main igl::core CGAL::CGAL ${OpenCV_LIBRARIES})

# ---------------------------  OpenGL visulization
# add_executable(gui gui.cpp)
# target_link_libraries(gui igl::core igl::opengl igl::opengl_glfw)

# ---------------------------- Fast-software for Box-intersection
# https://zhuanlan.zhihu.com/p/163706375

# ---------------------------   monster mash, Laplace Inflate
# add_executable(main monster_mash.cpp)
# target_link_libraries(main igl::core igl::opengl igl::opengl_glfw)

# ---------------------------   monster mash ARAP_L
add_executable(main arap_l.cpp)
target_link_libraries(main igl::core igl::opengl igl::opengl_glfw)

# ---------------------------   G_T * T * G == L
# add_executable(main laplace_by_gradient.cpp)
# target_link_libraries(main igl::core igl::opengl igl::opengl_glfw)

# ---------------------------   Langrange Multiplier to solve Linear Equality Constraint
# add_executable(main laplace_equality_constraint.cpp)
# target_link_libraries(main igl::core igl::opengl igl::opengl_glfw)


# ---------------------------   Active set to solve In-Equality Constraint
# add_executable(main quadratic_programming_inequality_constraint.cpp)
# target_link_libraries(main igl::core igl::opengl igl::opengl_glfw)


# --------------------------- Traits and Curiously Recurring Template Pattern
# add_executable(main mini_eigen.cpp)

# ---------------------------   Test
# include_directories(src/)
# add_executable(test test.cpp)
