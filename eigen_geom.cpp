#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Eigen>
#include <iostream>

using namespace Eigen;
using namespace std;
int main()
{
    Matrix3d m = Matrix3d::Random();
    cout << "m\n"
         << m << endl;
    auto filter = m.array() < 0.1;
    cout << "filtered: " << filter.count() << endl;
    cout << filter << endl
         << endl;

    // Affine 3D
    Vector3d v0(0, 0, 0);
    Vector3d v1(0, 1, 0);
    Affine3d t;
    t.setIdentity();
    cout << t.matrix() << endl;
    t.pretranslate(v0 - v1);
    cout << t.matrix() << endl;
    return 0;
}