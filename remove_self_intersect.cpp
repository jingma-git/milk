#include <igl/copyleft/cgal/remesh_self_intersections.h>
#include <igl/copyleft/cgal/outer_hull.h>

#include <igl/remove_unreferenced.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>


#include <iostream>
#include <fstream>
using namespace std;

int main(int argc, char *argv[]) {
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  igl::readOBJ(argv[1], V, F);
  cout << V.rows() << ", " << F.rows() << endl;

  Eigen::MatrixXd SV, VV;
  Eigen::MatrixXi SF, FF, IF, G, J, flip;
  Eigen::VectorXi H,IM, UIM;
  igl::copyleft::cgal::RemeshSelfIntersectionsParam params;
  params.detect_only = false;
  igl::copyleft::cgal::remesh_self_intersections(V,F,params,VV,FF,IF,H,IM);
  std::for_each(FF.data(),FF.data()+FF.size(),[&IM](int & a){a=IM(a);});
  igl::remove_unreferenced(VV,FF,SV,SF,UIM);
  igl::copyleft::cgal::outer_hull_legacy(SV,SF,G,J,flip);


  igl::writeOBJ(argv[2], SV, G);
  cout << SV.rows() << ", " << G.rows() << endl;
}