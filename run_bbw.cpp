#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/readOBJ.h>
#include <igl/barycenter.h>
#include <igl/dihedral_angles.h>
#include <igl/writeMESH.h>
#include <igl/winding_number.h>

using namespace Eigen;
using namespace std;


// Input polygon
Eigen::MatrixXd V, TV;
Eigen::MatrixXi F, TT, TF;

// tetralization
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_3<K> Delaunay;
typedef K::Point_3 Point;
typedef Delaunay::Vertex_handle Vertex_handle;
typedef Delaunay::Cell_handle Cell_handle;

std::string data_dir = "/mnt/f/Dataset/RigNetv1/";

void gen_tet(){
      std::vector<Point> pts;
      for(int i=0; i<V.rows(); ++i){
        pts.push_back(Point(V(i, 0), V(i, 1), V(i, 2)));
      }

      Delaunay dt;
      dt.insert(pts.begin(), pts.end());

      TV.resize(dt.number_of_vertices(), 3);
      TT.resize(dt.number_of_cells(), 4);

      std::map<Vertex_handle, size_t> v2i;
      int count = 0;
      for (auto vit= dt.finite_vertices_begin(); vit!=dt.finite_vertices_end(); ++vit) {
          v2i.insert(std::make_pair(vit, count));
          TV(count, 0) = vit->point().x();
          TV(count, 1) = vit->point().y();
          TV(count, 2) = vit->point().z();
          ++count;
      }
      cout << "TV:" << count << endl;
      count = 0;
      for (Delaunay::Finite_cells_iterator fit = dt.finite_cells_begin(); fit != dt.finite_cells_end(); ++fit) {
          TT(count, 0) = v2i[fit->vertex(0)];
          TT(count, 1) = v2i[fit->vertex(1)];
          TT(count, 2) = v2i[fit->vertex(2)];
          TT(count, 3) = v2i[fit->vertex(3)];
          ++count;
      }
      cout << "TT:" << count << endl;
}

int main(int argc, char *argv[])
{
  // Load a surface mesh
  std::string model_id = argv[1];
  igl::readOBJ(data_dir + "obj/" + model_id + ".obj",V,F);
  cout << V.rows() << ", " << F.rows() << endl;

  // delaunay tetralization
  {
    // gen_tet();
    igl::copyleft::tetgen::tetrahedralize(V,F,"pq1.414Y", TV,TT,TF);
    cout << TV.rows() << ", " << TT.rows() << ", " <<  TF.rows() << endl;
    igl::writeMESH(data_dir + "tet/" + model_id + ".mesh",  V, TT, F);
  }
  
  if(false)
  {
    Eigen::MatrixXd BC;
    // Compute barycenters of all tets
    igl::barycenter(V,TT,BC);
    // Compute generalized winding number at all barycenters
    cout<<"Computing winding number over all "<<TT.rows()<<" tets..."<<endl;
    Eigen::VectorXd W;
    igl::winding_number(V,F,BC,W);

    // Extract interior tets
    Eigen::MatrixXi CT((W.array()>0.5).count(),4);
    {
      size_t k = 0;
      for(size_t t = 0;t<TT.rows();t++)
      {
        if(W(t)>0.5)
        {
          CT.row(k) = TT.row(t);
          k++;
        }
      }
    }
    // // find bounary facets of interior tets
    // Eigen::MatrixXi G;
    // igl::boundary_facets(CT,G);
    // G = G.rowwise().reverse().eval();
    // cout << "interior tets: " << CT.rows() << ", " << G.rows() << endl;
    cout << "interior tets: " << CT.rows() << endl;
    igl::writeMESH(data_dir + "tet/" + model_id + ".mesh",  V, CT, F);
  }
}

