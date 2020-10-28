#pragma once

#include <eigen3/Eigen/Eigen>
#include <map>
#include <vector>
#include "primitives.h"

namespace rst
{
    enum class Buffers
    {
        Color = 1,
        Depth = 2
    };

    inline Buffers operator|(Buffers a, Buffers b)
    {
        return Buffers((int)a | (int)b);
    }

    inline Buffers operator&(Buffers a, Buffers b)
    {
        return Buffers((int)a & (int)b);
    }

    enum class Primitive
    {
        Line,
        Triangle
    };

    // Type safety
    struct PosBufId
    {

        int pos_id = 0;
    };
    struct IndBufId
    {
        int ind_id = 0;
    };
    struct ColorBufId
    {
        int col_id = 0;
    };

    class Rasterizer
    {
    public:
        Rasterizer(int w, int h);
        void set_model(const Eigen::Matrix4f &m);
        void set_view(const Eigen::Matrix4f &v);
        void set_projection(const Eigen::Matrix4f &p);

        PosBufId load_positions(const std::vector<Eigen::Vector3f> &positions);
        IndBufId load_indices(const std::vector<Eigen::Vector3i> &positions);
        void draw(PosBufId pos_id, IndBufId ind_id, ColorBufId col_id = {-1});

        void set_pixel(const Eigen::Vector2i &point, const Eigen::Vector3f &color);
        void clear(Buffers buff);

    private:
        void rasterize_triangle(const Triangle &t);

    private:
        Eigen::Matrix4f model;
        Eigen::Matrix4f view;
        Eigen::Matrix4f projection;

        std::map<int, std::vector<Eigen::Vector3f>> pos_buf;
        std::map<int, std::vector<Eigen::Vector3i>> ind_buf;
        std::map<int, std::vector<Eigen::Vector3f>> col_buf;

    public:
        std::vector<Eigen::Vector3f> frame_buf;
        std::vector<float> depth_buf;

        int width, height;

        int next_id = 0;
        int get_next_id() { return next_id++; }
    };
} // namespace rst