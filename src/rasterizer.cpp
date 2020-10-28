#include "rasterizer.h"
#include <iostream>

using namespace std;
using namespace rst;
using namespace Eigen;

static Vector4f to_vector4(const Vector3f &v)
{
    return Vector4f(v(0), v(1), v(2), 1.0);
}

static string vec_string(const Vector4f &v)
{
    char str[100];
    sprintf(str, "%.6f %.6f %.6f %.6f", v(0), v(1), v(2), v(3));
    return str;
}

static bool insideTriangle(int x, int y, const Eigen::Vector4f *_v)
{
    using namespace Eigen;
    Vector3f v[3];
    for (int i = 0; i < 3; i++)
        v[i] = {_v[i].x(), _v[i].y(), 1.0};
    Vector3f f0, f1, f2;
    f0 = v[1].cross(v[0]);
    f1 = v[2].cross(v[1]);
    f2 = v[0].cross(v[2]);
    Vector3f p(x, y, 1.);
    if ((p.dot(f0) * f0.dot(v[2]) > 0) && (p.dot(f1) * f1.dot(v[0]) > 0) && (p.dot(f2) * f2.dot(v[1]) > 0))
        return true;
    return false;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector4f *v)
{
    float c1 = (x * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * y + v[1].x() * v[2].y() - v[2].x() * v[1].y()) / (v[0].x() * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * v[0].y() + v[1].x() * v[2].y() - v[2].x() * v[1].y());
    float c2 = (x * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * y + v[2].x() * v[0].y() - v[0].x() * v[2].y()) / (v[1].x() * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * v[1].y() + v[2].x() * v[0].y() - v[0].x() * v[2].y());
    float c3 = (x * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * y + v[0].x() * v[1].y() - v[1].x() * v[0].y()) / (v[2].x() * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * v[2].y() + v[0].x() * v[1].y() - v[1].x() * v[0].y());
    return {c1, c2, c3};
}

Rasterizer::Rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);

    clear(Buffers::Color | Buffers::Depth);
}

void Rasterizer::set_model(const Eigen::Matrix4f &m)
{
    model = m;
}

void Rasterizer::set_view(const Eigen::Matrix4f &v)
{
    view = v;
}

void Rasterizer::set_projection(const Eigen::Matrix4f &p)
{
    projection = p;
}

void Rasterizer::set_pixel(const Vector2i &point, const Eigen::Vector3f &color)
{
    //old index: auto ind = point.y() + point.x() * width;
    int ind = (height - point.y()) * width + point.x();
    frame_buf[ind] = color;
}

void Rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

PosBufId Rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);
    return {id};
}

IndBufId Rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);
    return {id};
}

void Rasterizer::draw(PosBufId pos_buffer, IndBufId ind_buffer, ColorBufId col_buffer)
{
    auto &buf = pos_buf[pos_buffer.pos_id];
    auto &ind = ind_buf[ind_buffer.ind_id];
    auto &col = col_buf[col_buffer.col_id];

    int f_id = 0;
    Matrix4f mvp = projection * view * model;
    for (Vector3i &f : ind)
    {
        // cout << "----------------------" << f_id << endl;
        Triangle t;
        for (int i = 0; i < 3; i++)
        {
            Vector4f vi = to_vector4(buf[f(i)]);
            // cout << "orig " << vi.transpose() << endl;
            // Model View Projection
            vi = mvp * vi; //ToDo: clip the coords if x < -w or x > w
            // cout << "mvp " << vi.transpose() << endl;
            vi.x() = 0.5 * width * (vi.x() + 1.0);
            vi.y() = 0.5 * height * (vi.y() + 1.0);
            t.v[i] = vi;
            // cout << "viewport " << t.v[i].transpose() << endl;
        }
        rasterize_triangle(t);
        f_id++;
    }
}

void Rasterizer::rasterize_triangle(const Triangle &t)
{
    auto &v = t.v;
    BBox bbox = t.bbox();
    // cout << "bbox " << bbox << endl;
    for (int y = bbox.min_y; y < bbox.max_y; y++)
    {
        for (int x = bbox.min_x; x < bbox.max_x; x++)
        {
            float px = x + 0.5, py = y + 0.5;
            if (insideTriangle(x, y, t.v))
            {
                float alpha, beta, gamma;
                std::tie(alpha, beta, gamma) = computeBarycentric2D(px, py, t.v);
                float w_interp = 1. / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interp = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interp *= w_interp;

                int ind = (height - 1 - y) * width + x;
                // printf("%d %d = %.4f depth_buf = %.4f\n", x, y, z_interp, depth_buf[ind]);
                if (z_interp < depth_buf[ind])
                {
                    depth_buf[ind] = z_interp;
                    Vector2i pnt(x, y);
                    Vector3f color(1.0, 1.0, 1.0);
                    set_pixel(pnt, color);
                }
            }
        }
    }
}