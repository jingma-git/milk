#include <iostream>

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/readOFF.h>

#include <opencv2/opencv.hpp>

#include "rasterizer.h"
#include "transform.h"
#include "matrix_to_list.h"
using namespace std;
using namespace Eigen;
using namespace rst;

const int WIDTH = 600;
const int HEIGHT = 600;
enum Proj
{
    Ortho = 0,
    Perspective
};

enum VIEW
{
    FRONT = 0,
    BACK,
    LEFT,
    RIGHT,
    TOP,
    BOTTOM,
    NUMS
};
vector<std::string> proj_names = {
    "Ortho",
    "Persp"};

vector<std::string> view_names = {
    "front",
    "back",
    "left",
    "right",
    "top",
    "bottom"};

vector<Vector3f> eye_pos = {
    {0, 0, 2},
    {0, 0, -2},
    {-2, 0, 0},
    {2, 0, 0},
    {0, 2, 0},
    {0, -2, 0}};
vector<Vector3f> up_dir = {
    {0, 1, 0},
    {0, 1, 0},
    {0, 1, 0},
    {0, 1, 0},
    {0, 0, -1},
    {0, 0, 1}};

int main(int argc, char **argv)
{
    MatrixXf V;
    MatrixXi F;
    vector<Vector3f> v_list;
    vector<Vector3i> f_list;

    // igl::readOFF("./data/parts/cube.off", V, F);
    std::string obj_name = std::string(argv[1]);
    int proj_mode = Proj::Ortho;
    int view_mode = VIEW::FRONT;
    if (argc == 3)
        proj_mode = std::atoi(argv[2]);
    if (argc == 4)
        view_mode = std::atoi(argv[3]);

    igl::readOBJ("./data/parts/" + obj_name + ".obj", V, F);
    cout << "V, F=" << V.rows() << ", " << F.rows() << endl;

    matrix_to_list(V, v_list);
    matrix_to_list(F, f_list);

    Rasterizer rasterizer(WIDTH, HEIGHT);
    PosBufId pos_buf = rasterizer.load_positions(v_list);
    IndBufId ind_buf = rasterizer.load_indices(f_list);

    Matrix4f model = Matrix4f::Identity();
    Matrix4f view, proj;
    Vector3f eye = eye_pos[view_mode], target(0, 0, 0), up = up_dir[view_mode];
    look_at(eye, target, up, view);
    cout << "view\n"
         << view << endl;
    if (proj_mode == Proj::Perspective)
    {
        perspective_proj(2.0, 2.0, 0.5, 5.0, proj);
        cout << "proj\n"
             << proj << endl;
    }
    else
    {
        ortho_proj(1.5, 1.5, 0.1, 5, proj);
    }

    rasterizer.set_model(model);
    rasterizer.set_view(view);
    rasterizer.set_projection(proj);

    rasterizer.clear(Buffers::Color | Buffers::Depth);
    rasterizer.draw(pos_buf, ind_buf);
    cv::Mat img(HEIGHT, WIDTH, CV_32FC3, rasterizer.frame_buf.data());
    img = img * 255;
    img.convertTo(img, CV_8UC3);
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);

    cv::imwrite("./result/" + obj_name + "_" + proj_names[proj_mode] + "_" + view_names[view_mode] + ".png", img);
}