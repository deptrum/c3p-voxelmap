#ifndef VOXEL_MAP_UTIL_HPP
#define VOXEL_MAP_UTIL_HPP
#include <openssl/md5.h>
#include <pcl/common/io.h>
#include <rosbag/bag.h>
#include <stdio.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <bitset>
#include <chrono>
#include <execution>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "common_lib.h"
#include "omp.h"

#define HASH_P 116101
#define MAX_N 10000000000

static int plane_id = 0;
static int merged_plane_idx = 0;
int voxel_nums = 0;
int merged_voxel_nums = 0;

// a point to plane matching structure
typedef struct ptpl {
  Eigen::Vector3d point;
  Eigen::Vector3d normal;
  Eigen::Vector3d center;
  Eigen::Matrix<double, 6, 6> plane_cov;
  double d;
  int layer;
} ptpl;

// 3D point with covariance
typedef struct pointWithCov {
  Eigen::Vector3d point;
  Eigen::Vector3d point_world;
  Eigen::Matrix3d cov;
} pointWithCov;

struct Plane {
  Eigen::Vector3d center;
  Eigen::Vector3d normal;
  Eigen::Vector3d y_normal;
  Eigen::Vector3d x_normal;
  Eigen::Matrix3d covariance;
  Eigen::Matrix<double, 6, 6> plane_cov;
  float radius = 0;
  float min_eigen_value = 1;
  float mid_eigen_value = 1;
  float max_eigen_value = 1;
  float d = 0;
  int points_size = 0;

  bool is_plane = false;
  bool is_init = false;
  int id = -1;
  int merge_plane_id = -1;
  // is_update and last_update_points_size are only for publish plane
  bool is_update = false;
  int last_update_points_size = 0;

  Plane() = default;
};

typedef struct PointFreeParams {
  Eigen::Matrix<double, 6, 1> sigma_p_pt;
  Eigen::Vector3d sigma_p;
  size_t point_nums;
  Eigen::Matrix<double, 6, 1> sigma_cov11_ppt;
  Eigen::Matrix<double, 6, 1> sigma_cov12_ppt;
  Eigen::Matrix<double, 6, 1> sigma_cov13_ppt;
  Eigen::Matrix<double, 6, 1> sigma_cov22_ppt;
  Eigen::Matrix<double, 6, 1> sigma_cov23_ppt;
  Eigen::Matrix<double, 6, 1> sigma_cov33_ppt;
  Eigen::Matrix3d sigma_cov_e1_pt;
  Eigen::Matrix3d sigma_cov_e2_pt;
  Eigen::Matrix3d sigma_cov_e3_pt;
  Eigen::Matrix<double, 6, 1> sigma_cov;

  void Initialize() {
    sigma_p_pt.setZero();
    sigma_p.setZero();
    point_nums = 0;
    sigma_cov11_ppt.setZero();
    sigma_cov12_ppt.setZero();
    sigma_cov13_ppt.setZero();
    sigma_cov22_ppt.setZero();
    sigma_cov23_ppt.setZero();
    sigma_cov33_ppt.setZero();
    sigma_cov_e1_pt.setZero();
    sigma_cov_e2_pt.setZero();
    sigma_cov_e3_pt.setZero();
    sigma_cov.setZero();
  }
} PointFreeParams;

class PointCluster {
 public:
  Eigen::Matrix3d P;
  Eigen::Vector3d v;
  int N;

  PointCluster() {
    P.setZero();
    v.setZero();
    N = 0;
  }

  void Clear() {
    P.setZero();
    v.setZero();
    N = 0;
  }
  void Push(const Eigen::Vector3d& vec) {
    N++;
    P += vec * vec.transpose();
    v += vec;
  }

  Eigen::Matrix3d Cov() {
    Eigen::Vector3d center = v / N;
    return P / N - center * center.transpose();
  }

  PointCluster operator+(const PointCluster& b) const {
    PointCluster res;

    res.P = this->P + b.P;
    res.v = this->v + b.v;
    res.N = this->N + b.N;

    return res;
  }

  PointCluster& operator+=(const PointCluster& sigv) {
    this->P += sigv.P;
    this->v += sigv.v;
    this->N += sigv.N;

    return *this;
  }
};

class VOXEL_LOC {
 public:
  int64_t x, y, z;

  VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOC& other) const {
    return (x == other.x && y == other.y && z == other.z);
  }
};

struct PlaneDesc {
 public:
  int64_t theta_bin;
  int64_t phi_bin;
  int64_t d_bin;
  int64_t x_bin;
  int64_t y_bin;

  PlaneDesc(const Plane* const plane,
            double merge_x_coord_diff_thresh,
            double merge_y_coord_diff_thresh) {
    double phi = std::asin(plane->normal[2]);
    double theta = std::atan2(plane->normal[1], plane->normal[0]);

    Eigen::Matrix3d R = (Eigen::AngleAxisd(theta, Vector3d::UnitZ()) *
                         Eigen::AngleAxisd(-phi, Vector3d::UnitY()))
                            .toRotationMatrix();

    if (phi < std::numeric_limits<double>::epsilon()) {
      phi += M_PI;
    }

    if (theta < std::numeric_limits<double>::epsilon()) {
      theta += 2 * M_PI;
    }
    phi *= 57.29578;
    theta *= 57.29578;

    float d = 10.f * plane->d;

    theta_bin = static_cast<int64_t>(phi / 10.f);
    phi_bin = static_cast<int64_t>(theta / 5.f);
    d_bin = static_cast<int64_t>(d / 10.f);

    Eigen::Vector3d coord_in_xy_plane = R * plane->center;

    x_bin = static_cast<int64_t>(coord_in_xy_plane(1) / merge_x_coord_diff_thresh);
    y_bin = static_cast<int64_t>(coord_in_xy_plane(2) / merge_y_coord_diff_thresh);
  }
  bool operator==(const PlaneDesc& other) const {
    return (theta_bin == other.theta_bin && phi_bin == other.phi_bin && d_bin == other.d_bin &&
            x_bin == other.x_bin && y_bin == other.y_bin);
  }
};

struct NodeLoc {
 public:
  int64_t x;
  int64_t y;
  int64_t z;
  int64_t leaf_num;
  NodeLoc() : x(0), y(0), z(0), leaf_num(0) {}
  NodeLoc(int64_t x_, int64_t y_, int64_t z_, int64_t leaf_num_) :
      x(x_),
      y(y_),
      z(z_),
      leaf_num(leaf_num_){};

  NodeLoc(const VOXEL_LOC& v, const std::vector<int64_t>& leaf_path) : x(v.x), y(v.y), z(v.z) {
    leaf_num = 0;

    if (!leaf_path.empty()) {
      for (int i = leaf_path.size() - 1; i >= 0; i--) {
        leaf_num = 10 * leaf_num + leaf_path[i] + 1;
      }
    }
  }

  bool operator==(const NodeLoc& other) const {
    return (x == other.x) && (y == other.y) && (z == other.z) && (leaf_num == other.leaf_num);
  }
};

// Hash value
namespace std {
template<>
struct hash<VOXEL_LOC> {
  int64_t operator()(const VOXEL_LOC& s) const {
    using std::hash;
    using std::size_t;
    return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
  }
};

template<>
struct hash<PlaneDesc> {
  int64_t operator()(const PlaneDesc& plane) const {
    using std::hash;
    using std::size_t;
    return ((((((((plane.theta_bin) * HASH_P) % MAX_N + (plane.phi_bin)) * HASH_P) % MAX_N +
               (plane.d_bin)) *
              HASH_P) %
                 MAX_N +
             (plane.x_bin)) *
            HASH_P) %
               MAX_N +
           (plane.y_bin);
  }
};

template<>
struct hash<NodeLoc> {
  std::size_t operator()(const NodeLoc& c) const {
    return ((((((c.z) * HASH_P) % MAX_N + (c.y)) * HASH_P) % MAX_N + (c.x)) * HASH_P) % MAX_N +
           c.leaf_num;
  }
};
}  // namespace std

struct UnionPlane {
  NodeLoc root_node_loc;
  std::unordered_set<NodeLoc> plane_bucket;
};

class OctoTree {
 public:
  std::vector<pointWithCov> temp_points_;  // all points in an octo tree
  std::vector<pointWithCov> new_points_;   // new points in an octo tree
  Plane* plane_ptr_;
  PointFreeParams* point_free_params_;
  int max_layer_;
  bool indoor_mode_;
  int layer_;
  int octo_state_;  // 0 is end of tree, 1 is not
  OctoTree* leaves_[8];
  double voxel_center_[3];  // x, y, z
  std::vector<int> layer_point_size_;
  float quater_length_;
  float planer_threshold_;
  int max_plane_update_threshold_;
  bool init_octo_;
  PointCluster point_cluster_;
  uint32_t last_update_frame_;
  OctoTree(int max_layer, int layer, std::vector<int> layer_point_size, float planer_threshold) :
      max_layer_(max_layer),
      layer_(layer),
      layer_point_size_(layer_point_size),
      planer_threshold_(planer_threshold) {
    temp_points_.clear();
    octo_state_ = 0;
    init_octo_ = false;
    max_plane_update_threshold_ = layer_point_size_[layer_];
    for (int i = 0; i < 8; i++) {
      leaves_[i] = nullptr;
    }
    point_free_params_ = nullptr;
    plane_ptr_ = new Plane;
    last_update_frame_ = 0;
  }

  ~OctoTree() {
    delete plane_ptr_;
    delete point_free_params_;
    for (size_t i = 0; i < 8; i++) {
      delete leaves_[i];
    }
  }

  int GetLeafNums() {
    int res = 0;
    for (size_t i = 0; i < 8; i++) {
      if (leaves_[i] != nullptr) {
        res += leaves_[i]->GetLeafNums();
      }
    }
    return res;
  }

  void MergeVoxelPlane(const OctoTree* const node) {
    point_cluster_ += node->point_cluster_;

    point_free_params_->sigma_p_pt += node->point_free_params_->sigma_p_pt;
    point_free_params_->sigma_p += node->point_free_params_->sigma_p;
    point_free_params_->point_nums += node->point_free_params_->point_nums;
    point_free_params_->sigma_cov11_ppt += node->point_free_params_->sigma_cov11_ppt;
    point_free_params_->sigma_cov12_ppt += node->point_free_params_->sigma_cov12_ppt;
    point_free_params_->sigma_cov13_ppt += node->point_free_params_->sigma_cov13_ppt;
    point_free_params_->sigma_cov22_ppt += node->point_free_params_->sigma_cov22_ppt;
    point_free_params_->sigma_cov23_ppt += node->point_free_params_->sigma_cov23_ppt;
    point_free_params_->sigma_cov33_ppt += node->point_free_params_->sigma_cov33_ppt;
    point_free_params_->sigma_cov_e1_pt += node->point_free_params_->sigma_cov_e1_pt;
    point_free_params_->sigma_cov_e2_pt += node->point_free_params_->sigma_cov_e2_pt;
    point_free_params_->sigma_cov_e3_pt += node->point_free_params_->sigma_cov_e3_pt;
    point_free_params_->sigma_cov += node->point_free_params_->sigma_cov;

    plane_ptr_->center = point_cluster_.v / point_cluster_.N;
    plane_ptr_->covariance = point_cluster_.Cov();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(plane_ptr_->covariance);

    Eigen::Vector3d eigen_value = solver.eigenvalues();
    Eigen::Matrix3d eigen_vector = solver.eigenvectors();

    plane_ptr_->normal = eigen_vector.col(0);

    plane_ptr_->min_eigen_value = eigen_value(0);
    plane_ptr_->mid_eigen_value = eigen_value(1);
    plane_ptr_->max_eigen_value = eigen_value(2);
    plane_ptr_->radius = std::sqrt(eigen_value(2));
    plane_ptr_->d = -(plane_ptr_->normal.dot(plane_ptr_->center));
    plane_ptr_->points_size = point_cluster_.N;

    std::vector<Eigen::Matrix3d> B(3);
    // Jacobian of the normal of plane w.r.t the world point
    for (int m = 0; m < 3; m++) {
      if (m != 0) {
        B[m] = (eigen_vector.col(m) * eigen_vector.col(0).transpose() +
                eigen_vector.col(0) * eigen_vector.col(m).transpose()) /
               (plane_ptr_->points_size * (eigen_value[0] - eigen_value[m]));
      } else {
        B[m].setZero();
      }
    }
    CalculatePointFreeVoxelPlaneCovariance(plane_ptr_->plane_cov,
                                           plane_ptr_->center,
                                           B,
                                           eigen_vector,
                                           0);
  }

  void UpdatePointFreeParams(const pointWithCov& pv) {
    const auto& pi = pv.point;
    const auto& cov = pv.cov;

    Eigen::Vector3d e1(1.0, 0.0, 0.0);
    Eigen::Vector3d e2(0.0, 1.0, 0.0);
    Eigen::Vector3d e3(0.0, 0.0, 1.0);

    point_free_params_->point_nums++;
    point_free_params_->sigma_p += pi;

    auto p_pt = pi * pi.transpose();
    point_free_params_->sigma_p_pt(0) += p_pt(0, 0);
    point_free_params_->sigma_p_pt(1) += p_pt(0, 1);
    point_free_params_->sigma_p_pt(2) += p_pt(0, 2);
    point_free_params_->sigma_p_pt(3) += p_pt(1, 1);
    point_free_params_->sigma_p_pt(4) += p_pt(1, 2);
    point_free_params_->sigma_p_pt(5) += p_pt(2, 2);

    Eigen::Matrix3d cov11_ppt = cov(0, 0) * pi * pi.transpose();
    point_free_params_->sigma_cov11_ppt(0) += cov11_ppt(0, 0);
    point_free_params_->sigma_cov11_ppt(1) += cov11_ppt(0, 1);
    point_free_params_->sigma_cov11_ppt(2) += cov11_ppt(0, 2);
    point_free_params_->sigma_cov11_ppt(3) += cov11_ppt(1, 1);
    point_free_params_->sigma_cov11_ppt(4) += cov11_ppt(1, 2);
    point_free_params_->sigma_cov11_ppt(5) += cov11_ppt(2, 2);

    Eigen::Matrix3d cov12_ppt = cov(0, 1) * pi * pi.transpose();
    point_free_params_->sigma_cov12_ppt(0) += cov12_ppt(0, 0);
    point_free_params_->sigma_cov12_ppt(1) += cov12_ppt(0, 1);
    point_free_params_->sigma_cov12_ppt(2) += cov12_ppt(0, 2);
    point_free_params_->sigma_cov12_ppt(3) += cov12_ppt(1, 1);
    point_free_params_->sigma_cov12_ppt(4) += cov12_ppt(1, 2);
    point_free_params_->sigma_cov12_ppt(5) += cov12_ppt(2, 2);

    Eigen::Matrix3d cov13_ppt = cov(0, 2) * pi * pi.transpose();
    point_free_params_->sigma_cov13_ppt(0) += cov13_ppt(0, 0);
    point_free_params_->sigma_cov13_ppt(1) += cov13_ppt(0, 1);
    point_free_params_->sigma_cov13_ppt(2) += cov13_ppt(0, 2);
    point_free_params_->sigma_cov13_ppt(3) += cov13_ppt(1, 1);
    point_free_params_->sigma_cov13_ppt(4) += cov13_ppt(1, 2);
    point_free_params_->sigma_cov13_ppt(5) += cov13_ppt(2, 2);

    Eigen::Matrix3d cov22_ppt = cov(1, 1) * pi * pi.transpose();
    point_free_params_->sigma_cov22_ppt(0) += cov22_ppt(0, 0);
    point_free_params_->sigma_cov22_ppt(1) += cov22_ppt(0, 1);
    point_free_params_->sigma_cov22_ppt(2) += cov22_ppt(0, 2);
    point_free_params_->sigma_cov22_ppt(3) += cov22_ppt(1, 1);
    point_free_params_->sigma_cov22_ppt(4) += cov22_ppt(1, 2);
    point_free_params_->sigma_cov22_ppt(5) += cov22_ppt(2, 2);

    Eigen::Matrix3d cov23_ppt = cov(1, 2) * pi * pi.transpose();
    point_free_params_->sigma_cov23_ppt(0) += cov23_ppt(0, 0);
    point_free_params_->sigma_cov23_ppt(1) += cov23_ppt(0, 1);
    point_free_params_->sigma_cov23_ppt(2) += cov23_ppt(0, 2);
    point_free_params_->sigma_cov23_ppt(3) += cov23_ppt(1, 1);
    point_free_params_->sigma_cov23_ppt(4) += cov23_ppt(1, 2);
    point_free_params_->sigma_cov23_ppt(5) += cov23_ppt(2, 2);

    Eigen::Matrix3d cov33_ppt = cov(2, 2) * pi * pi.transpose();
    point_free_params_->sigma_cov33_ppt(0) += cov33_ppt(0, 0);
    point_free_params_->sigma_cov33_ppt(1) += cov33_ppt(0, 1);
    point_free_params_->sigma_cov33_ppt(2) += cov33_ppt(0, 2);
    point_free_params_->sigma_cov33_ppt(3) += cov33_ppt(1, 1);
    point_free_params_->sigma_cov33_ppt(4) += cov33_ppt(1, 2);
    point_free_params_->sigma_cov33_ppt(5) += cov33_ppt(2, 2);

    point_free_params_->sigma_cov_e1_pt += cov * e1 * pi.transpose();
    point_free_params_->sigma_cov_e2_pt += cov * e2 * pi.transpose();
    point_free_params_->sigma_cov_e3_pt += cov * e3 * pi.transpose();

    point_free_params_->sigma_cov(0) += cov(0, 0);
    point_free_params_->sigma_cov(1) += cov(0, 1);
    point_free_params_->sigma_cov(2) += cov(0, 2);
    point_free_params_->sigma_cov(3) += cov(1, 1);
    point_free_params_->sigma_cov(4) += cov(1, 2);
    point_free_params_->sigma_cov(5) += cov(2, 2);
  }

  bool UpdateMergedPlanePointFreeParams(const pointWithCov& pv) {
    const auto& pi = pv.point;
    const auto& cov = pv.cov;

    auto p_pt = pi * pi.transpose();
    Eigen::Matrix<double, 6, 1> sigma_p_pt;
    sigma_p_pt(0) = point_free_params_->sigma_p_pt(0) + p_pt(0, 0);
    sigma_p_pt(1) = point_free_params_->sigma_p_pt(1) + p_pt(0, 1);
    sigma_p_pt(2) = point_free_params_->sigma_p_pt(2) + p_pt(0, 2);
    sigma_p_pt(3) = point_free_params_->sigma_p_pt(3) + p_pt(1, 1);
    sigma_p_pt(4) = point_free_params_->sigma_p_pt(4) + p_pt(1, 2);
    sigma_p_pt(5) = point_free_params_->sigma_p_pt(5) + p_pt(2, 2);

    Eigen::Vector3d sigma_p = point_free_params_->sigma_p + pi;

    Eigen::Matrix3d sigma_p_pt_mat;
    sigma_p_pt_mat(0, 0) = sigma_p_pt(0);
    sigma_p_pt_mat(0, 1) = sigma_p_pt(1);
    sigma_p_pt_mat(0, 2) = sigma_p_pt(2);
    sigma_p_pt_mat(1, 0) = sigma_p_pt(1);
    sigma_p_pt_mat(1, 1) = sigma_p_pt(3);
    sigma_p_pt_mat(1, 2) = sigma_p_pt(4);
    sigma_p_pt_mat(2, 0) = sigma_p_pt(2);
    sigma_p_pt_mat(2, 1) = sigma_p_pt(4);
    sigma_p_pt_mat(2, 2) = sigma_p_pt(5);

    int cur_point_size = point_free_params_->point_nums + 1;
    Eigen::Vector3d cur_center = sigma_p / cur_point_size;
    Eigen::Matrix3d cur_covariance = sigma_p_pt_mat / cur_point_size -
                                     cur_center * cur_center.transpose();

    Eigen::EigenSolver<Eigen::Matrix3d> es(cur_covariance);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evalsReal;
    evalsReal = evals.real();
    Eigen::Matrix3d::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    int evalsMid = 3 - evalsMin - evalsMax;
    Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
    Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
    Eigen::Vector3d evecMax = evecs.real().col(evalsMax);

    if (evalsReal(evalsMin) < planer_threshold_) {
      UpdatePointFreeParams(pv);

      return false;
    }

    return true;
  }

  void CalculatePointFreeVoxelPlaneCovariance(Eigen::Matrix<double, 6, 6>& plane_cov,
                                              const Eigen::Vector3d& center,
                                              const std::vector<Eigen::Matrix3d>& B,
                                              const Eigen::Matrix3d& J_n_pw,
                                              const int& min_eval_idx) {
    plane_cov.setZero();
    Eigen::Vector3d e1(1.0, 0.0, 0.0);
    Eigen::Vector3d e2(0.0, 1.0, 0.0);
    Eigen::Vector3d e3(0.0, 0.0, 1.0);

    Eigen::Matrix3d sigma_cov11_ppt_mat;
    sigma_cov11_ppt_mat(0, 0) = point_free_params_->sigma_cov11_ppt(0);
    sigma_cov11_ppt_mat(0, 1) = point_free_params_->sigma_cov11_ppt(1);
    sigma_cov11_ppt_mat(0, 2) = point_free_params_->sigma_cov11_ppt(2);
    sigma_cov11_ppt_mat(1, 0) = point_free_params_->sigma_cov11_ppt(1);
    sigma_cov11_ppt_mat(1, 1) = point_free_params_->sigma_cov11_ppt(3);
    sigma_cov11_ppt_mat(1, 2) = point_free_params_->sigma_cov11_ppt(4);
    sigma_cov11_ppt_mat(2, 0) = point_free_params_->sigma_cov11_ppt(2);
    sigma_cov11_ppt_mat(2, 1) = point_free_params_->sigma_cov11_ppt(4);
    sigma_cov11_ppt_mat(2, 2) = point_free_params_->sigma_cov11_ppt(5);

    Eigen::Matrix3d sigma_cov12_ppt_mat;
    sigma_cov12_ppt_mat(0, 0) = point_free_params_->sigma_cov12_ppt(0);
    sigma_cov12_ppt_mat(0, 1) = point_free_params_->sigma_cov12_ppt(1);
    sigma_cov12_ppt_mat(0, 2) = point_free_params_->sigma_cov12_ppt(2);
    sigma_cov12_ppt_mat(1, 0) = point_free_params_->sigma_cov12_ppt(1);
    sigma_cov12_ppt_mat(1, 1) = point_free_params_->sigma_cov12_ppt(3);
    sigma_cov12_ppt_mat(1, 2) = point_free_params_->sigma_cov12_ppt(4);
    sigma_cov12_ppt_mat(2, 0) = point_free_params_->sigma_cov12_ppt(2);
    sigma_cov12_ppt_mat(2, 1) = point_free_params_->sigma_cov12_ppt(4);
    sigma_cov12_ppt_mat(2, 2) = point_free_params_->sigma_cov12_ppt(5);

    Eigen::Matrix3d sigma_cov13_ppt_mat;
    sigma_cov13_ppt_mat(0, 0) = point_free_params_->sigma_cov13_ppt(0);
    sigma_cov13_ppt_mat(0, 1) = point_free_params_->sigma_cov13_ppt(1);
    sigma_cov13_ppt_mat(0, 2) = point_free_params_->sigma_cov13_ppt(2);
    sigma_cov13_ppt_mat(1, 0) = point_free_params_->sigma_cov13_ppt(1);
    sigma_cov13_ppt_mat(1, 1) = point_free_params_->sigma_cov13_ppt(3);
    sigma_cov13_ppt_mat(1, 2) = point_free_params_->sigma_cov13_ppt(4);
    sigma_cov13_ppt_mat(2, 0) = point_free_params_->sigma_cov13_ppt(2);
    sigma_cov13_ppt_mat(2, 1) = point_free_params_->sigma_cov13_ppt(4);
    sigma_cov13_ppt_mat(2, 2) = point_free_params_->sigma_cov13_ppt(5);

    Eigen::Matrix3d sigma_cov22_ppt_mat;
    sigma_cov22_ppt_mat(0, 0) = point_free_params_->sigma_cov22_ppt(0);
    sigma_cov22_ppt_mat(0, 1) = point_free_params_->sigma_cov22_ppt(1);
    sigma_cov22_ppt_mat(0, 2) = point_free_params_->sigma_cov22_ppt(2);
    sigma_cov22_ppt_mat(1, 0) = point_free_params_->sigma_cov22_ppt(1);
    sigma_cov22_ppt_mat(1, 1) = point_free_params_->sigma_cov22_ppt(3);
    sigma_cov22_ppt_mat(1, 2) = point_free_params_->sigma_cov22_ppt(4);
    sigma_cov22_ppt_mat(2, 0) = point_free_params_->sigma_cov22_ppt(2);
    sigma_cov22_ppt_mat(2, 1) = point_free_params_->sigma_cov22_ppt(4);
    sigma_cov22_ppt_mat(2, 2) = point_free_params_->sigma_cov22_ppt(5);

    Eigen::Matrix3d sigma_cov23_ppt_mat;
    sigma_cov23_ppt_mat(0, 0) = point_free_params_->sigma_cov23_ppt(0);
    sigma_cov23_ppt_mat(0, 1) = point_free_params_->sigma_cov23_ppt(1);
    sigma_cov23_ppt_mat(0, 2) = point_free_params_->sigma_cov23_ppt(2);
    sigma_cov23_ppt_mat(1, 0) = point_free_params_->sigma_cov23_ppt(1);
    sigma_cov23_ppt_mat(1, 1) = point_free_params_->sigma_cov23_ppt(3);
    sigma_cov23_ppt_mat(1, 2) = point_free_params_->sigma_cov23_ppt(4);
    sigma_cov23_ppt_mat(2, 0) = point_free_params_->sigma_cov23_ppt(2);
    sigma_cov23_ppt_mat(2, 1) = point_free_params_->sigma_cov23_ppt(4);
    sigma_cov23_ppt_mat(2, 2) = point_free_params_->sigma_cov23_ppt(5);

    Eigen::Matrix3d sigma_cov33_ppt_mat;
    sigma_cov33_ppt_mat(0, 0) = point_free_params_->sigma_cov33_ppt(0);
    sigma_cov33_ppt_mat(0, 1) = point_free_params_->sigma_cov33_ppt(1);
    sigma_cov33_ppt_mat(0, 2) = point_free_params_->sigma_cov33_ppt(2);
    sigma_cov33_ppt_mat(1, 0) = point_free_params_->sigma_cov33_ppt(1);
    sigma_cov33_ppt_mat(1, 1) = point_free_params_->sigma_cov33_ppt(3);
    sigma_cov33_ppt_mat(1, 2) = point_free_params_->sigma_cov33_ppt(4);
    sigma_cov33_ppt_mat(2, 0) = point_free_params_->sigma_cov33_ppt(2);
    sigma_cov33_ppt_mat(2, 1) = point_free_params_->sigma_cov33_ppt(4);
    sigma_cov33_ppt_mat(2, 2) = point_free_params_->sigma_cov33_ppt(5);

    Eigen::Matrix3d sigma_cov_mat;
    sigma_cov_mat(0, 0) = point_free_params_->sigma_cov(0);
    sigma_cov_mat(0, 1) = point_free_params_->sigma_cov(1);
    sigma_cov_mat(0, 2) = point_free_params_->sigma_cov(2);
    sigma_cov_mat(1, 0) = point_free_params_->sigma_cov(1);
    sigma_cov_mat(1, 1) = point_free_params_->sigma_cov(3);
    sigma_cov_mat(1, 2) = point_free_params_->sigma_cov(4);
    sigma_cov_mat(2, 0) = point_free_params_->sigma_cov(2);
    sigma_cov_mat(2, 1) = point_free_params_->sigma_cov(4);
    sigma_cov_mat(2, 2) = point_free_params_->sigma_cov(5);

    // Calculate plane_cov.block<3, 3>(0, 0)
    for (int i = 0; i < 3; i++) {
      if (i != min_eval_idx) {
        for (int j = i; j < 3; j++) {
          if (j != min_eval_idx) {
            const auto& B1 = B[i];
            const auto& B2 = B[j].transpose();

            double sigma_pt_B1_cov_B2t_p = (sigma_cov11_ppt_mat * B1 * e1 * e1.transpose() * B2)
                                               .trace() +
                                           (sigma_cov12_ppt_mat * B1 * e1 * e2.transpose() * B2)
                                               .trace() +
                                           (sigma_cov13_ppt_mat * B1 * e1 * e3.transpose() * B2)
                                               .trace() +
                                           (sigma_cov12_ppt_mat * B1 * e2 * e1.transpose() * B2)
                                               .trace() +
                                           (sigma_cov22_ppt_mat * B1 * e2 * e2.transpose() * B2)
                                               .trace() +
                                           (sigma_cov23_ppt_mat * B1 * e2 * e3.transpose() * B2)
                                               .trace() +
                                           (sigma_cov13_ppt_mat * B1 * e3 * e1.transpose() * B2)
                                               .trace() +
                                           (sigma_cov23_ppt_mat * B1 * e3 * e2.transpose() * B2)
                                               .trace() +
                                           (sigma_cov33_ppt_mat * B1 * e3 * e3.transpose() * B2)
                                               .trace();

            double e1t_B2_q = e1.transpose() * B2 * center;
            double e2t_B2_q = e2.transpose() * B2 * center;
            double e3t_B2_q = e3.transpose() * B2 * center;
            double e1t_B1_q = e1.transpose() * B1 * center;
            double e2t_B1_q = e2.transpose() * B1 * center;
            double e3t_B1_q = e3.transpose() * B1 * center;
            double sigma_pt_B1_cov_B2t_q = e1t_B2_q *
                                               (point_free_params_->sigma_cov_e1_pt * B1).trace() +
                                           e2t_B2_q *
                                               (point_free_params_->sigma_cov_e2_pt * B1).trace() +
                                           e3t_B2_q *
                                               (point_free_params_->sigma_cov_e3_pt * B1).trace();

            double sigma_qt_B1_cov_B2t_p = e1t_B1_q *
                                               (point_free_params_->sigma_cov_e1_pt * B2).trace() +
                                           e2t_B1_q *
                                               (point_free_params_->sigma_cov_e2_pt * B2).trace() +
                                           e3t_B1_q *
                                               (point_free_params_->sigma_cov_e3_pt * B2).trace();

            double sigma_qt_B1_cov_B2t_q = center.transpose() * B1 * sigma_cov_mat * B2 * center;

            plane_cov(i, j) = sigma_pt_B1_cov_B2t_p - sigma_pt_B1_cov_B2t_q -
                              sigma_qt_B1_cov_B2t_p + sigma_qt_B1_cov_B2t_q;
          }
        }
      }
    }
    for (size_t i = 0; i < 3; i++) {
      for (size_t j = i + 1; j < 3; j++) {
        plane_cov(j, i) = plane_cov(i, j);
      }
    }
    plane_cov.block<3, 3>(0, 0) = J_n_pw * plane_cov.block<3, 3>(0, 0).eval() * J_n_pw.transpose();

    // Calculate plane_cov.block<3, 3>(0, 3) and plane_cov.block<3, 3>(3, 0)
    for (int i = 0; i < 3; i++) {
      if (i != min_eval_idx) {
        const auto& Bk = B[i];

        Eigen::Vector3d pt_Bk_cov;
        Eigen::Vector3d qt_Bk_cov;

        pt_Bk_cov(0) = (point_free_params_->sigma_cov_e1_pt * Bk).trace();
        pt_Bk_cov(1) = (point_free_params_->sigma_cov_e2_pt * Bk).trace();
        pt_Bk_cov(2) = (point_free_params_->sigma_cov_e3_pt * Bk).trace();

        qt_Bk_cov = center.transpose() * Bk * sigma_cov_mat;
        plane_cov.block<1, 3>(i, 3) = pt_Bk_cov - qt_Bk_cov;
      }
    }
    plane_cov.block<3, 3>(0, 3) = J_n_pw * plane_cov.block<3, 3>(0, 3).eval() /
                                  point_free_params_->point_nums;
    plane_cov.block<3, 3>(3, 0) = plane_cov.block<3, 3>(0, 3).transpose();

    // Calculate plane_cov.block<3, 3>(3, 3)
    plane_cov.block<3, 3>(3, 3) = sigma_cov_mat /
                                  (point_free_params_->point_nums * point_free_params_->point_nums);
  }

  // check is plane , calc plane parameters including plane covariance
  void init_plane(const std::vector<pointWithCov>& points, Plane* plane) {
    plane->plane_cov = Eigen::Matrix<double, 6, 6>::Zero();
    plane->covariance = Eigen::Matrix3d::Zero();
    plane->center = Eigen::Vector3d::Zero();
    plane->normal = Eigen::Vector3d::Zero();
    plane->radius = 0;

    if (!init_octo_) {
      plane->points_size = points.size();
      for (auto pv : points) {
        plane->covariance += pv.point * pv.point.transpose();
        plane->center += pv.point;
      }
      plane->center = plane->center / plane->points_size;
      plane->covariance = plane->covariance / plane->points_size -
                          plane->center * plane->center.transpose();
    } else {
      Eigen::Matrix3d sigma_p_pt_mat;
      sigma_p_pt_mat(0, 0) = point_free_params_->sigma_p_pt(0);
      sigma_p_pt_mat(0, 1) = point_free_params_->sigma_p_pt(1);
      sigma_p_pt_mat(0, 2) = point_free_params_->sigma_p_pt(2);
      sigma_p_pt_mat(1, 0) = point_free_params_->sigma_p_pt(1);
      sigma_p_pt_mat(1, 1) = point_free_params_->sigma_p_pt(3);
      sigma_p_pt_mat(1, 2) = point_free_params_->sigma_p_pt(4);
      sigma_p_pt_mat(2, 0) = point_free_params_->sigma_p_pt(2);
      sigma_p_pt_mat(2, 1) = point_free_params_->sigma_p_pt(4);
      sigma_p_pt_mat(2, 2) = point_free_params_->sigma_p_pt(5);
      plane->points_size = point_free_params_->point_nums;
      plane->center = point_free_params_->sigma_p / plane->points_size;
      plane->covariance = sigma_p_pt_mat / plane->points_size -
                          plane->center * plane->center.transpose();
    }

    Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evalsReal;
    evalsReal = evals.real();
    Eigen::Matrix3d::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    int evalsMid = 3 - evalsMin - evalsMax;
    Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
    Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
    Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
    // plane covariance calculation
    Eigen::Matrix3d J_Q;
    J_Q << 1.0 / plane->points_size, 0, 0, 0, 1.0 / plane->points_size, 0, 0, 0,
        1.0 / plane->points_size;

    if (evalsReal(evalsMin) < planer_threshold_) {
      if (!init_octo_) {
        point_free_params_ = new PointFreeParams();
        point_free_params_->Initialize();
        for (const auto& pv : temp_points_) {
          UpdatePointFreeParams(pv);
        }
      }
      std::vector<Eigen::Matrix3d> B(3);
      // Jacobian of the normal of plane w.r.t the world point
      auto J_n_pw = evecs.real();
      std::vector<Eigen::Matrix3d> F_aux(3);
      for (int m = 0; m < 3; m++) {
        if (m != (int) evalsMin) {
          B[m] = (evecs.real().col(m) * evecs.real().col(evalsMin).transpose() +
                  evecs.real().col(evalsMin) * evecs.real().col(m).transpose()) /
                 (plane->points_size * (evalsReal[evalsMin] - evalsReal[m]));
        } else {
          B[m].setZero();
        }
      }
      CalculatePointFreeVoxelPlaneCovariance(plane->plane_cov,
                                             plane->center,
                                             B,
                                             J_n_pw,
                                             (int) evalsMin);

      plane->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
          evecs.real()(2, evalsMin);
      plane->y_normal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid),
          evecs.real()(2, evalsMid);
      plane->x_normal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax),
          evecs.real()(2, evalsMax);
      plane->min_eigen_value = evalsReal(evalsMin);
      plane->mid_eigen_value = evalsReal(evalsMid);
      plane->max_eigen_value = evalsReal(evalsMax);
      plane->radius = sqrt(evalsReal(evalsMax));
      plane->d = -(plane->normal(0) * plane->center(0) + plane->normal(1) * plane->center(1) +
                   plane->normal(2) * plane->center(2));

      plane->is_plane = true;
      if (plane->last_update_points_size == 0) {
        plane->last_update_points_size = plane->points_size;
        plane->is_update = true;
      } else if (plane->points_size - plane->last_update_points_size > 100) {
        plane->last_update_points_size = plane->points_size;
        plane->is_update = true;
      }

      if (!plane->is_init) {
        plane->id = plane_id;
        plane_id++;
        plane->is_init = true;
      }
    } else {
      if (!plane->is_init) {
        plane->id = plane_id;
        plane_id++;
        plane->is_init = true;
      }
      if (plane->last_update_points_size == 0) {
        plane->last_update_points_size = plane->points_size;
        plane->is_update = true;
      } else if (plane->points_size - plane->last_update_points_size > 100) {
        plane->last_update_points_size = plane->points_size;
        plane->is_update = true;
      }
      plane->is_plane = false;
      plane->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
          evecs.real()(2, evalsMin);
      plane->y_normal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid),
          evecs.real()(2, evalsMid);
      plane->x_normal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax),
          evecs.real()(2, evalsMax);
      plane->min_eigen_value = evalsReal(evalsMin);
      plane->mid_eigen_value = evalsReal(evalsMid);
      plane->max_eigen_value = evalsReal(evalsMax);
      plane->radius = sqrt(evalsReal(evalsMax));
      plane->d = -(plane->normal(0) * plane->center(0) + plane->normal(1) * plane->center(1) +
                   plane->normal(2) * plane->center(2));
    }
  }

  void UpdatePlaneCov() {
    plane_ptr_->plane_cov = Eigen::Matrix<double, 6, 6>::Zero();
    plane_ptr_->covariance = Eigen::Matrix3d::Zero();
    plane_ptr_->center = Eigen::Vector3d::Zero();
    plane_ptr_->normal = Eigen::Vector3d::Zero();
    plane_ptr_->radius = 0;

    Eigen::Matrix3d sigma_p_pt_mat;
    sigma_p_pt_mat(0, 0) = point_free_params_->sigma_p_pt(0);
    sigma_p_pt_mat(0, 1) = point_free_params_->sigma_p_pt(1);
    sigma_p_pt_mat(0, 2) = point_free_params_->sigma_p_pt(2);
    sigma_p_pt_mat(1, 0) = point_free_params_->sigma_p_pt(1);
    sigma_p_pt_mat(1, 1) = point_free_params_->sigma_p_pt(3);
    sigma_p_pt_mat(1, 2) = point_free_params_->sigma_p_pt(4);
    sigma_p_pt_mat(2, 0) = point_free_params_->sigma_p_pt(2);
    sigma_p_pt_mat(2, 1) = point_free_params_->sigma_p_pt(4);
    sigma_p_pt_mat(2, 2) = point_free_params_->sigma_p_pt(5);
    plane_ptr_->points_size = point_free_params_->point_nums;
    plane_ptr_->center = point_free_params_->sigma_p / plane_ptr_->points_size;
    plane_ptr_->covariance = sigma_p_pt_mat / plane_ptr_->points_size -
                             plane_ptr_->center * plane_ptr_->center.transpose();

    Eigen::EigenSolver<Eigen::Matrix3d> es(plane_ptr_->covariance);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evalsReal;
    evalsReal = evals.real();
    Eigen::Matrix3d::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    int evalsMid = 3 - evalsMin - evalsMax;
    Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
    Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
    Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
    // plane covariance calculation
    Eigen::Matrix3d J_Q;
    J_Q << 1.0 / plane_ptr_->points_size, 0, 0, 0, 1.0 / plane_ptr_->points_size, 0, 0, 0,
        1.0 / plane_ptr_->points_size;

    std::vector<Eigen::Matrix3d> B(3);
    // Jacobian of the normal of plane w.r.t the world point
    auto J_n_pw = evecs.real();
    std::vector<Eigen::Matrix3d> F_aux(3);
    for (int m = 0; m < 3; m++) {
      if (m != (int) evalsMin) {
        B[m] = (evecs.real().col(m) * evecs.real().col(evalsMin).transpose() +
                evecs.real().col(evalsMin) * evecs.real().col(m).transpose()) /
               (plane_ptr_->points_size * (evalsReal[evalsMin] - evalsReal[m]));
      } else {
        B[m].setZero();
      }
    }
    CalculatePointFreeVoxelPlaneCovariance(plane_ptr_->plane_cov,
                                           plane_ptr_->center,
                                           B,
                                           J_n_pw,
                                           (int) evalsMin);

    plane_ptr_->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
        evecs.real()(2, evalsMin);
    plane_ptr_->y_normal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid),
        evecs.real()(2, evalsMid);
    plane_ptr_->x_normal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax),
        evecs.real()(2, evalsMax);
    plane_ptr_->min_eigen_value = evalsReal(evalsMin);
    plane_ptr_->mid_eigen_value = evalsReal(evalsMid);
    plane_ptr_->max_eigen_value = evalsReal(evalsMax);
    plane_ptr_->radius = sqrt(evalsReal(evalsMax));
    plane_ptr_->d = -(plane_ptr_->normal(0) * plane_ptr_->center(0) +
                      plane_ptr_->normal(1) * plane_ptr_->center(1) +
                      plane_ptr_->normal(2) * plane_ptr_->center(2));

    plane_ptr_->is_plane = true;
    if (plane_ptr_->last_update_points_size == 0) {
      plane_ptr_->last_update_points_size = plane_ptr_->points_size;
      plane_ptr_->is_update = true;
    } else if (plane_ptr_->points_size - plane_ptr_->last_update_points_size > 100) {
      plane_ptr_->last_update_points_size = plane_ptr_->points_size;
      plane_ptr_->is_update = true;
    }
  }

  void init_octo_tree(std::vector<size_t>& leaf_path, const VOXEL_LOC& voxel_loc) {
    if (temp_points_.size() > max_plane_update_threshold_) {
      init_plane(temp_points_, plane_ptr_);
      if (plane_ptr_->is_plane == true) {
        octo_state_ = 0;
      } else {
        if (layer_ < max_layer_) {
          octo_state_ = 1;
          cut_octo_tree(leaf_path, voxel_loc);
        } else {
          point_free_params_ = new PointFreeParams();
          point_free_params_->Initialize();
          for (const auto& pv : temp_points_) {
            UpdatePointFreeParams(pv);
          }
        }
      }
      std::vector<pointWithCov>().swap(temp_points_);
      init_octo_ = true;
    }
  }

  void cut_octo_tree(std::vector<size_t>& leaf_path, const VOXEL_LOC& voxel_loc) {
    if (layer_ == max_layer_) {
      octo_state_ = 0;
      return;
    }
    for (size_t i = 0; i < temp_points_.size(); i++) {
      int xyz[3] = {0, 0, 0};
      if (temp_points_[i].point[0] > voxel_center_[0]) {
        xyz[0] = 1;
      }
      if (temp_points_[i].point[1] > voxel_center_[1]) {
        xyz[1] = 1;
      }
      if (temp_points_[i].point[2] > voxel_center_[2]) {
        xyz[2] = 1;
      }
      int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
      if (leaves_[leafnum] == nullptr) {
        leaves_[leafnum] = new OctoTree(max_layer_,
                                        layer_ + 1,
                                        layer_point_size_,
                                        planer_threshold_);
        voxel_nums++;
        leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
        leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
        leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
        leaves_[leafnum]->quater_length_ = quater_length_ / 2;
        leaf_path.push_back(leafnum);
        std::bitset<32> leaf_id;
        leaf_id.reset();
        for (size_t i = 0; i < leaf_path.size(); i++) {
          size_t bit_idx = 4 * i + 1;
          if (i == leaf_path.size() - 1) {
            leaf_id[bit_idx] = 1;
          }
          size_t leaf_idx = leaf_path[i];
          leaf_id[bit_idx + 1] = leaf_idx & 1;
          leaf_id[bit_idx + 2] = (leaf_idx >> 1) & 1;
          leaf_id[bit_idx + 3] = (leaf_idx >> 2) & 1;
        }
        leaf_path.pop_back();
      }
      leaves_[leafnum]->temp_points_.push_back(temp_points_[i]);
    }
    for (size_t leafnum = 0; leafnum < 8; ++leafnum) {
      if (leaves_[leafnum]) {
        if (leaves_[leafnum]->temp_points_.size() > leaves_[leafnum]->max_plane_update_threshold_) {
          leaf_path.push_back(leafnum);
          leaves_[leafnum]->init_octo_tree(leaf_path, voxel_loc);
          leaf_path.pop_back();
        }
      }
    }
  }

  void UpdateOctoTree(bool& plane_is_update,
                      std::vector<int64_t>& plane_leaf_path,
                      bool& node_need_split,
                      const pointWithCov& pv,
                      std::vector<size_t>& leaf_path,
                      const VOXEL_LOC& voxel_loc,
                      uint32_t frame_num) {
    last_update_frame_ = frame_num;
    if (!init_octo_) {
      point_cluster_.Push(pv.point);
      temp_points_.push_back(pv);
      if (temp_points_.size() > max_plane_update_threshold_) {
        init_octo_tree(leaf_path, voxel_loc);
      }
    } else {
      if (plane_ptr_->is_plane) {
        point_cluster_.Push(pv.point);
        if (plane_ptr_->merge_plane_id >= 0) {
          node_need_split = UpdateMergedPlanePointFreeParams(pv);
          UpdatePlaneCov();
        } else {
          UpdatePointFreeParams(pv);
          init_plane(temp_points_, plane_ptr_);
          if (plane_ptr_->is_plane) {
            plane_is_update = true;
          }
        }
      } else {
        if (layer_ < max_layer_) {
          if (temp_points_.size() != 0) {
            std::vector<pointWithCov>().swap(temp_points_);
          }
          if (new_points_.size() != 0) {
            std::vector<pointWithCov>().swap(new_points_);
          }
          int xyz[3] = {0, 0, 0};
          if (pv.point[0] > voxel_center_[0]) {
            xyz[0] = 1;
          }
          if (pv.point[1] > voxel_center_[1]) {
            xyz[1] = 1;
          }
          if (pv.point[2] > voxel_center_[2]) {
            xyz[2] = 1;
          }
          int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];

          leaf_path.push_back(leafnum);
          if (leaves_[leafnum] != nullptr) {
            plane_leaf_path.push_back(leafnum);
            leaves_[leafnum]->UpdateOctoTree(plane_is_update,
                                             plane_leaf_path,
                                             node_need_split,
                                             pv,
                                             leaf_path,
                                             voxel_loc,
                                             frame_num);
          } else {
            leaves_[leafnum] = new OctoTree(max_layer_,
                                            layer_ + 1,
                                            layer_point_size_,
                                            planer_threshold_);
            voxel_nums++;
            leaves_[leafnum]->layer_point_size_ = layer_point_size_;
            leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] +
                                                 (2 * xyz[0] - 1) * quater_length_;
            leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] +
                                                 (2 * xyz[1] - 1) * quater_length_;
            leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] +
                                                 (2 * xyz[2] - 1) * quater_length_;
            leaves_[leafnum]->quater_length_ = quater_length_ / 2;

            std::bitset<32> leaf_id;
            leaf_id.reset();
            for (size_t i = 0; i < leaf_path.size(); i++) {
              size_t bit_idx = 4 * i + 1;
              if (i == leaf_path.size() - 1) {
                leaf_id[bit_idx] = 1;
              }
              size_t leaf_idx = leaf_path[i];
              leaf_id[bit_idx + 1] = leaf_idx & 1;
              leaf_id[bit_idx + 2] = (leaf_idx >> 1) & 1;
              leaf_id[bit_idx + 3] = (leaf_idx >> 2) & 1;
            }

            plane_leaf_path.push_back(leafnum);
            leaves_[leafnum]->UpdateOctoTree(plane_is_update,
                                             plane_leaf_path,
                                             node_need_split,
                                             pv,
                                             leaf_path,
                                             voxel_loc,
                                             frame_num);
          }
          leaf_path.pop_back();
        } else {
          point_cluster_.Push(pv.point);
          UpdatePointFreeParams(pv);
          init_plane(temp_points_, plane_ptr_);
        }
      }
    }
  }

  void MergeNode(OctoTree* merged_node, int64_t leaf_num, bool need_delete) {
    int64_t leaf_layer_num = leaf_num / 10;
    int64_t leaf_idx = leaf_num % 10 - 1;

    if (leaf_layer_num == 0) {
      if (need_delete) {
        merged_voxel_nums++;
        merged_voxel_nums += leaves_[leaf_idx]->GetLeafNums();
        delete leaves_[leaf_idx];
      }
      leaves_[leaf_idx] = merged_node;
      return;
    } else {
      leaves_[leaf_idx]->MergeNode(merged_node, leaf_layer_num, need_delete);
    }
  }

  void SplitMergedNode(std::vector<size_t>& leaf_path,
                       const VOXEL_LOC& voxel_loc,
                       const pointWithCov& pv,
                       uint32_t frame_num,
                       int64_t leaf_num,
                       int layer) {
    int64_t leaf_layer_num = leaf_num / 10;
    int64_t leaf_idx = leaf_num % 10 - 1;

    int xyz[3] = {0, 0, 0};
    if (pv.point[0] > voxel_center_[0]) {
      xyz[0] = 1;
    }
    if (pv.point[1] > voxel_center_[1]) {
      xyz[1] = 1;
    }
    if (pv.point[2] > voxel_center_[2]) {
      xyz[2] = 1;
    }
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];

    if (leaf_layer_num == 0) {
      OctoTree* node = new OctoTree(leaves_[leaf_idx]->max_layer_,
                                    layer,
                                    leaves_[leaf_idx]->layer_point_size_,
                                    leaves_[leaf_idx]->planer_threshold_);
      voxel_nums++;
      leaves_[leaf_idx] = node;

      leaves_[leaf_idx]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
      leaves_[leaf_idx]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
      leaves_[leaf_idx]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
      leaves_[leaf_idx]->quater_length_ = quater_length_ / 2;

      leaves_[leaf_idx]->last_update_frame_ = frame_num;
      leaves_[leaf_idx]->temp_points_.push_back(pv);
      leaves_[leaf_idx]->point_cluster_.Push(pv.point);

      leaf_path.push_back(leaf_idx);
      std::bitset<32> leaf_id;
      leaf_id.reset();
      for (size_t i = 0; i < leaf_path.size(); i++) {
        size_t bit_idx = 4 * i + 1;
        if (i == leaf_path.size() - 1) {
          leaf_id[bit_idx] = 1;
        }
        size_t leaf_idx = leaf_path[i];
        leaf_id[bit_idx + 1] = leaf_idx & 1;
        leaf_id[bit_idx + 2] = (leaf_idx >> 1) & 1;
        leaf_id[bit_idx + 3] = (leaf_idx >> 2) & 1;
      }
      return;
    } else {
      leaf_path.push_back(leaf_idx);
      leaves_[leaf_idx]
          ->SplitMergedNode(leaf_path, voxel_loc, pv, frame_num, leaf_layer_num, layer + 1);
    }
  }
};

void mapJet(double v, double vmin, double vmax, uint8_t& r, uint8_t& g, uint8_t& b) {
  r = 255;
  g = 255;
  b = 255;

  if (v < vmin) {
    v = vmin;
  }

  if (v > vmax) {
    v = vmax;
  }

  double dr, dg, db;

  if (v < 0.1242) {
    db = 0.504 + ((1. - 0.504) / 0.1242) * v;
    dg = dr = 0.;
  } else if (v < 0.3747) {
    db = 1.;
    dr = 0.;
    dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
  } else if (v < 0.6253) {
    db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
    dg = 1.;
    dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
  } else if (v < 0.8758) {
    db = 0.;
    dr = 1.;
    dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
  } else {
    db = 0.;
    dg = 0.;
    dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
  }

  r = (uint8_t)(255 * dr);
  g = (uint8_t)(255 * dg);
  b = (uint8_t)(255 * db);
}

bool CheckIfMerge(std::pair<Eigen::Vector3d, Eigen::Vector3d>& target_plane_coeff,
                  PointCluster& target_cluster,
                  const Plane* const candidate,
                  const PointCluster& candidate_cluster,
                  double merge_theta_thresh,
                  double merge_dist_thresh,
                  double merge_cov_min_eigen_val_thresh,
                  double merge_x_coord_diff_thresh,
                  double merge_y_coord_diff_thresh) {
  double length = (target_plane_coeff.second - candidate->center).norm();
  double dist = length * (target_plane_coeff.second - candidate->center)
                             .normalized()
                             .dot(target_plane_coeff.first);

  const auto& candidate_normal = candidate->normal;
  if (std::abs(target_plane_coeff.first.normalized().dot(candidate_normal.normalized())) <
      std::cos(merge_theta_thresh)) {
    return false;
  }

  if (std::abs(dist) > merge_dist_thresh) {
    return false;
  }

  PointCluster plane_merged = target_cluster + candidate_cluster;

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(plane_merged.Cov());

  if (solver.eigenvalues()[0] < merge_cov_min_eigen_val_thresh && solver.eigenvalues()[0] > 0) {
    target_cluster = plane_merged;

    target_plane_coeff.first = solver.eigenvectors().col(0);
    target_plane_coeff.second = target_cluster.v / target_cluster.N;

    return true;
  }

  return false;
}

OctoTree* GetVoxelPlaneNodeRecursive(OctoTree* current_node, int64_t leaf_num) {
  int64_t leaf_layer_num = leaf_num / 10;
  int64_t leaf_idx = leaf_num % 10 - 1;

  if (leaf_layer_num == 0) {
    return current_node->leaves_[leaf_idx];
  }

  return GetVoxelPlaneNodeRecursive(current_node->leaves_[leaf_idx], leaf_layer_num);
}

OctoTree* GetVoxelPlaneNode(std::unordered_map<VOXEL_LOC, OctoTree*>& feat_map,
                            const NodeLoc& node_loc) {
  VOXEL_LOC voxel_loc(node_loc.x, node_loc.y, node_loc.z);

  if (node_loc.leaf_num == 0) {
    return feat_map[voxel_loc];
  } else {
    return GetVoxelPlaneNodeRecursive(feat_map[voxel_loc], node_loc.leaf_num);
  }
}

void MergeVoxelPlaneNode(OctoTree* root_node,
                         std::unordered_map<VOXEL_LOC, OctoTree*>& feat_map,
                         const NodeLoc& node_loc,
                         bool need_delete) {
  VOXEL_LOC voxel_loc(node_loc.x, node_loc.y, node_loc.z);

  if (node_loc.leaf_num == 0) {
    if (need_delete) {
      merged_voxel_nums++;
      merged_voxel_nums += feat_map[voxel_loc]->GetLeafNums();
      delete feat_map[voxel_loc];
    }
    feat_map[voxel_loc] = root_node;
  } else {
    feat_map[voxel_loc]->MergeNode(root_node, node_loc.leaf_num, need_delete);
  }
}

void UpdatePlaneHashMap(std::unordered_map<VOXEL_LOC, OctoTree*>& feat_map,
                        std::unordered_map<PlaneDesc, UnionPlane>& plane_map,
                        std::unordered_map<int, std::unordered_set<NodeLoc>>& merged_table,
                        const VOXEL_LOC& voxel_loc,
                        const std::vector<int64_t>& leaf_path,
                        double merge_theta_thresh,
                        double merge_dist_thresh,
                        double merge_cov_min_eigen_val_thresh,
                        double merge_x_coord_diff_thresh,
                        double merge_y_coord_diff_thresh) {
  std::unordered_map<NodeLoc, PlaneDesc> unique_node_table;

  NodeLoc node_loc(voxel_loc, leaf_path);

  const OctoTree* const current_node = GetVoxelPlaneNode(feat_map, node_loc);

  if (!current_node->plane_ptr_->is_plane) {
    return;
  }

  PlaneDesc plane_loc(current_node->plane_ptr_,
                      merge_x_coord_diff_thresh,
                      merge_y_coord_diff_thresh);
  auto plane_loc_it = plane_map.find(plane_loc);

  if (plane_loc_it != plane_map.end()) {
    auto& union_plane_bucket = plane_loc_it->second.plane_bucket;

    if (union_plane_bucket.count(node_loc) == 0) {
      union_plane_bucket.insert(node_loc);

      if (union_plane_bucket.size() > 4) {
        bool has_initialized = false;
        {
          for (const auto& plane_i : union_plane_bucket) {
            if (has_initialized && plane_i == plane_loc_it->second.root_node_loc) {
              continue;
            }

            const OctoTree* const p1 = GetVoxelPlaneNode(feat_map, plane_i);
            if (!p1 || !p1->plane_ptr_->is_plane) {
              continue;
            }

            if (!has_initialized) {
              plane_loc_it->second.root_node_loc = plane_i;
              has_initialized = true;
              continue;
            }

            const OctoTree* const p2 = GetVoxelPlaneNode(feat_map,
                                                         plane_loc_it->second.root_node_loc);

            if (!p2 || p1->point_cluster_.N > p2->point_cluster_.N) {
              plane_loc_it->second.root_node_loc = plane_i;
            }
          }
        }

        if (!has_initialized) {
          return;
        }

        const auto& root_node_loc = plane_loc_it->second.root_node_loc;

        OctoTree* root_plane_content = GetVoxelPlaneNode(feat_map, root_node_loc);

        if (!root_plane_content) {
          return;
        }

        PointCluster merged_plane = root_plane_content->point_cluster_;
        std::pair<Eigen::Vector3d, Eigen::Vector3d> merged_plane_coeff = std::make_pair(
            root_plane_content->plane_ptr_->normal,
            root_plane_content->plane_ptr_->center);

        for (auto vit = union_plane_bucket.begin(); vit != union_plane_bucket.end(); ++vit) {
          const auto& node_loc_cur = *vit;

          if (node_loc_cur == root_node_loc) {
            continue;
          }

          int node_cur_merged_id = -1;
          {
            OctoTree* voxel_plane_cur = GetVoxelPlaneNode(feat_map, node_loc_cur);
            if (!voxel_plane_cur || !voxel_plane_cur->plane_ptr_->is_plane ||
                voxel_plane_cur == root_plane_content) {
              continue;
            }

            if (!CheckIfMerge(merged_plane_coeff,
                              merged_plane,
                              voxel_plane_cur->plane_ptr_,
                              voxel_plane_cur->point_cluster_,
                              merge_theta_thresh,
                              merge_dist_thresh,
                              merge_cov_min_eigen_val_thresh,
                              merge_x_coord_diff_thresh,
                              merge_y_coord_diff_thresh)) {
              continue;
            }

            if (root_plane_content->plane_ptr_->merge_plane_id < 0) {
              root_plane_content->plane_ptr_->merge_plane_id = merged_plane_idx;
              merged_plane_idx++;
            }

            node_cur_merged_id = voxel_plane_cur->plane_ptr_->merge_plane_id;

            root_plane_content->MergeVoxelPlane(voxel_plane_cur);
          }

          merged_table[root_plane_content->plane_ptr_->merge_plane_id].insert(root_node_loc);
          merged_table[root_plane_content->plane_ptr_->merge_plane_id].insert(node_loc_cur);

          if (node_cur_merged_id >= 0) {
            auto& nodes_with_id = merged_table[node_cur_merged_id];
            for (const auto& node_loc : nodes_with_id) {
              MergeVoxelPlaneNode(root_plane_content, feat_map, node_loc, node_loc == node_loc_cur);

              merged_table[root_plane_content->plane_ptr_->merge_plane_id].insert(node_loc);
            }
            std::unordered_set<NodeLoc>().swap(nodes_with_id);
            merged_table.erase(node_cur_merged_id);
          } else {
            MergeVoxelPlaneNode(root_plane_content, feat_map, node_loc_cur, true);
          }
        }

        std::unordered_set<NodeLoc>().swap(union_plane_bucket);
        plane_map.erase(plane_loc_it);

        PlaneDesc root_plane_loc(root_plane_content->plane_ptr_,
                                 merge_x_coord_diff_thresh,
                                 merge_y_coord_diff_thresh);
        auto root_plane_loc_it = plane_map.find(root_plane_loc);
        if (root_plane_loc_it != plane_map.end()) {
          root_plane_loc_it->second.plane_bucket.insert(root_node_loc);
        } else {
          UnionPlane union_plane;
          union_plane.plane_bucket.insert(root_node_loc);
          union_plane.root_node_loc = root_node_loc;
          plane_map[root_plane_loc] = union_plane;
        }
      }
    }
  } else {
    UnionPlane union_plane;
    union_plane.plane_bucket.insert(node_loc);
    union_plane.root_node_loc = node_loc;
    plane_map[plane_loc] = union_plane;
  }
}

void buildVoxelMap(bool enable_voxel_merging,
                   const std::vector<pointWithCov>& input_points,
                   const float voxel_size,
                   const int max_layer,
                   const std::vector<int>& layer_point_size,
                   const float planer_threshold,
                   std::unordered_map<VOXEL_LOC, OctoTree*>& feat_map,
                   std::unordered_map<PlaneDesc, UnionPlane>& plane_map,
                   std::unordered_map<int, std::unordered_set<NodeLoc>>& merged_table,
                   double merge_theta_thresh,
                   double merge_dist_thresh,
                   double merge_cov_min_eigen_val_thresh,
                   double merge_x_coord_diff_thresh,
                   double merge_y_coord_diff_thresh) {
  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++) {
    const pointWithCov& p_v = input_points[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_v.point[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t) loc_xyz[0], (int64_t) loc_xyz[1], (int64_t) loc_xyz[2]);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      feat_map[position]->temp_points_.push_back(p_v);

      feat_map[position]->point_cluster_.Push(p_v.point);
    } else {
      OctoTree* octo_tree = new OctoTree(max_layer, 0, layer_point_size, planer_threshold);
      voxel_nums++;
      feat_map[position] = octo_tree;
      feat_map[position]->quater_length_ = voxel_size / 4;
      feat_map[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      feat_map[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      feat_map[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
      feat_map[position]->temp_points_.push_back(p_v);
      feat_map[position]->layer_point_size_ = layer_point_size;
      std::bitset<32> leaf_id;
      leaf_id.reset();
      leaf_id[0] = 1;
      feat_map[position]->point_cluster_.Push(p_v.point);
    }
  }
  for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter) {
    std::vector<size_t> leaf_path;
    iter->second->init_octo_tree(leaf_path, iter->first);

    if (enable_voxel_merging && iter->second->plane_ptr_->is_plane) {
      UpdatePlaneHashMap(feat_map,
                         plane_map,
                         merged_table,
                         iter->first,
                         std::vector<int64_t>(),
                         merge_theta_thresh,
                         merge_dist_thresh,
                         merge_cov_min_eigen_val_thresh,
                         merge_x_coord_diff_thresh,
                         merge_y_coord_diff_thresh);
    }
  }
}

void updateVoxelMap(bool enable_voxel_merging,
                    const std::vector<pointWithCov>& input_points,
                    const float voxel_size,
                    const int max_layer,
                    const std::vector<int>& layer_point_size,
                    const float planer_threshold,
                    std::unordered_map<VOXEL_LOC, OctoTree*>& feat_map,
                    std::unordered_map<PlaneDesc, UnionPlane>& plane_map,
                    std::unordered_map<int, std::unordered_set<NodeLoc>>& merged_table,
                    uint32_t frame_num,
                    double merge_theta_thresh,
                    double merge_dist_thresh,
                    double merge_cov_min_eigen_val_thresh,
                    double merge_x_coord_diff_thresh,
                    double merge_y_coord_diff_thresh) {
  uint plsize = input_points.size();

  for (uint i = 0; i < plsize; i++) {
    // Check plane of voxel is update
    bool plane_is_update = false;
    // Node path in voxel
    std::vector<int64_t> plane_node_loc;

    const pointWithCov p_v = input_points[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_v.point[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t) loc_xyz[0], (int64_t) loc_xyz[1], (int64_t) loc_xyz[2]);

    bool node_need_split = false;
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      std::vector<size_t> leaf_path;
      feat_map[position]->UpdateOctoTree(plane_is_update,
                                         plane_node_loc,
                                         node_need_split,
                                         p_v,
                                         leaf_path,
                                         position,
                                         frame_num);
    } else {
      OctoTree* octo_tree = new OctoTree(max_layer, 0, layer_point_size, planer_threshold);
      voxel_nums++;
      feat_map[position] = octo_tree;
      feat_map[position]->quater_length_ = voxel_size / 4;
      feat_map[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      feat_map[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      feat_map[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
      std::bitset<32> leaf_id;
      leaf_id.reset();
      leaf_id[0] = 1;
      std::vector<size_t> leaf_path;
      feat_map[position]->UpdateOctoTree(plane_is_update,
                                         plane_node_loc,
                                         node_need_split,
                                         p_v,
                                         leaf_path,
                                         position,
                                         frame_num);
    }

    if (enable_voxel_merging) {
      if (plane_is_update) {
        UpdatePlaneHashMap(feat_map,
                           plane_map,
                           merged_table,
                           position,
                           plane_node_loc,
                           merge_theta_thresh,
                           merge_dist_thresh,
                           merge_cov_min_eigen_val_thresh,
                           merge_x_coord_diff_thresh,
                           merge_y_coord_diff_thresh);
      }
    }
  }
}

void transformLidar(const StatesGroup& state,
                    const shared_ptr<ImuProcess>& p_imu,
                    const PointCloudXYZI::Ptr& input_cloud,
                    pcl::PointCloud<pcl::PointXYZI>::Ptr& trans_cloud) {
  trans_cloud->clear();
  for (size_t i = 0; i < input_cloud->size(); i++) {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    // p = p_imu->Lid_rot_to_IMU * p + p_imu->Lid_offset_to_IMU;
    p = state.rot_end * p + state.pos_end;
    pcl::PointXYZI pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    trans_cloud->points.push_back(pi);
  }
}

void build_single_residual(const pointWithCov& pv,
                           const OctoTree* current_octo,
                           const int current_layer,
                           const int max_layer,
                           const double sigma_num,
                           bool& is_sucess,
                           double& prob,
                           ptpl& single_ptpl) {
  double radius_k = 3;
  Eigen::Vector3d p_w = pv.point_world;
  if (current_octo->plane_ptr_->is_plane) {
    Plane& plane = *current_octo->plane_ptr_;
    Eigen::Vector3d p_world_to_center = p_w - plane.center;
    double proj_x = p_world_to_center.dot(plane.x_normal);
    double proj_y = p_world_to_center.dot(plane.y_normal);
    float dis_to_plane = fabs(plane.normal(0) * p_w(0) + plane.normal(1) * p_w(1) +
                              plane.normal(2) * p_w(2) + plane.d);
    float dis_to_center = (plane.center(0) - p_w(0)) * (plane.center(0) - p_w(0)) +
                          (plane.center(1) - p_w(1)) * (plane.center(1) - p_w(1)) +
                          (plane.center(2) - p_w(2)) * (plane.center(2) - p_w(2));
    float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);

    if (range_dis <= radius_k * plane.radius) {
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = p_w - plane.center;
      J_nq.block<1, 3>(0, 3) = -plane.normal;
      double sigma_l = J_nq * plane.plane_cov * J_nq.transpose();
      sigma_l += plane.normal.transpose() * pv.cov * plane.normal;
      if (dis_to_plane < sigma_num * sqrt(sigma_l)) {
        is_sucess = true;
        double this_prob = 1.0 / (sqrt(sigma_l)) *
                           exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
        if (this_prob > prob) {
          prob = this_prob;
          single_ptpl.point = pv.point;
          single_ptpl.plane_cov = plane.plane_cov;
          single_ptpl.normal = plane.normal;
          single_ptpl.center = plane.center;
          single_ptpl.d = plane.d;
          single_ptpl.layer = current_layer;
        }
        return;
      } else {
        // is_sucess = false;
        return;
      }
    } else {
      // is_sucess = false;
      return;
    }
  } else {
    if (current_layer < max_layer) {
      for (size_t leafnum = 0; leafnum < 8; leafnum++) {
        if (current_octo->leaves_[leafnum] != nullptr) {
          OctoTree* leaf_octo = current_octo->leaves_[leafnum];
          build_single_residual(pv,
                                leaf_octo,
                                current_layer + 1,
                                max_layer,
                                sigma_num,
                                is_sucess,
                                prob,
                                single_ptpl);
        }
      }
      return;
    } else {
      // is_sucess = false;
      return;
    }
  }
}

void GetUpdatePlane(const OctoTree* current_octo,
                    const int pub_max_voxel_layer,
                    std::vector<Plane>& plane_list) {
  if (current_octo->layer_ > pub_max_voxel_layer) {
    return;
  }
  if (current_octo->plane_ptr_->is_update) {
    plane_list.push_back(*current_octo->plane_ptr_);
  }
  if (current_octo->layer_ < current_octo->max_layer_) {
    if (!current_octo->plane_ptr_->is_plane) {
      for (size_t i = 0; i < 8; i++) {
        if (current_octo->leaves_[i] != nullptr) {
          GetUpdatePlane(current_octo->leaves_[i], pub_max_voxel_layer, plane_list);
        }
      }
    }
  }
  return;
}

// void BuildResidualListTBB(const unordered_map<VOXEL_LOC, OctoTree *>
// &voxel_map,
//                           const double voxel_size, const double sigma_num,
//                           const int max_layer,
//                           const std::vector<pointWithCov> &pv_list,
//                           std::vector<ptpl> &ptpl_list,
//                           std::vector<Eigen::Vector3d> &non_match) {
//   std::mutex mylock;
//   ptpl_list.clear();
//   std::vector<ptpl> all_ptpl_list(pv_list.size());
//   std::vector<bool> useful_ptpl(pv_list.size());
//   std::vector<size_t> index(pv_list.size());
//   for (size_t i = 0; i < index.size(); ++i) {
//     index[i] = i;
//     useful_ptpl[i] = false;
//   }
//   std::for_each(
//       std::execution::par_unseq, index.begin(), index.end(),
//       [&](const size_t &i) {
//         pointWithCov pv = pv_list[i];
//         float loc_xyz[3];
//         for (int j = 0; j < 3; j++) {
//           loc_xyz[j] = pv.point_world[j] / voxel_size;
//           if (loc_xyz[j] < 0) {
//             loc_xyz[j] -= 1.0;
//           }
//         }
//         VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
//                            (int64_t)loc_xyz[2]);
//         auto iter = voxel_map.find(position);
//         if (iter != voxel_map.end()) {
//           OctoTree *current_octo = iter->second;
//           ptpl single_ptpl;
//           bool is_sucess = false;
//           double prob = 0;
//           build_single_residual(pv, current_octo, 0, max_layer, sigma_num,
//                                 is_sucess, prob, single_ptpl);
//           if (!is_sucess) {
//             VOXEL_LOC near_position = position;
//             if (loc_xyz[0] > (current_octo->voxel_center_[0] +
//                               current_octo->quater_length_)) {
//               near_position.x = near_position.x + 1;
//             } else if (loc_xyz[0] < (current_octo->voxel_center_[0] -
//                                      current_octo->quater_length_)) {
//               near_position.x = near_position.x - 1;
//             }
//             if (loc_xyz[1] > (current_octo->voxel_center_[1] +
//                               current_octo->quater_length_)) {
//               near_position.y = near_position.y + 1;
//             } else if (loc_xyz[1] < (current_octo->voxel_center_[1] -
//                                      current_octo->quater_length_)) {
//               near_position.y = near_position.y - 1;
//             }
//             if (loc_xyz[2] > (current_octo->voxel_center_[2] +
//                               current_octo->quater_length_)) {
//               near_position.z = near_position.z + 1;
//             } else if (loc_xyz[2] < (current_octo->voxel_center_[2] -
//                                      current_octo->quater_length_)) {
//               near_position.z = near_position.z - 1;
//             }
//             auto iter_near = voxel_map.find(near_position);
//             if (iter_near != voxel_map.end()) {
//               build_single_residual(pv, iter_near->second, 0, max_layer,
//                                     sigma_num, is_sucess, prob, single_ptpl);
//             }
//           }
//           if (is_sucess) {

//             mylock.lock();
//             useful_ptpl[i] = true;
//             all_ptpl_list[i] = single_ptpl;
//             mylock.unlock();
//           } else {
//             mylock.lock();
//             useful_ptpl[i] = false;
//             mylock.unlock();
//           }
//         }
//       });
//   for (size_t i = 0; i < useful_ptpl.size(); i++) {
//     if (useful_ptpl[i]) {
//       ptpl_list.push_back(all_ptpl_list[i]);
//     }
//   }
// }

void BuildResidualListOMP(const unordered_map<VOXEL_LOC, OctoTree*>& voxel_map,
                          const double voxel_size,
                          const double sigma_num,
                          const int max_layer,
                          const std::vector<pointWithCov>& pv_list,
                          std::vector<ptpl>& ptpl_list,
                          std::vector<Eigen::Vector3d>& non_match) {
  std::mutex mylock;
  ptpl_list.clear();
  std::vector<ptpl> all_ptpl_list(pv_list.size());
  std::vector<bool> useful_ptpl(pv_list.size());
  std::vector<size_t> index(pv_list.size());
  for (size_t i = 0; i < index.size(); ++i) {
    index[i] = i;
    useful_ptpl[i] = false;
  }
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
  for (int i = 0; i < index.size(); i++) {
    pointWithCov pv = pv_list[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = pv.point_world[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t) loc_xyz[0], (int64_t) loc_xyz[1], (int64_t) loc_xyz[2]);
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end()) {
      OctoTree* current_octo = iter->second;
      ptpl single_ptpl;
      bool is_sucess = false;
      double prob = 0;
      build_single_residual(pv,
                            current_octo,
                            0,
                            max_layer,
                            sigma_num,
                            is_sucess,
                            prob,
                            single_ptpl);
      if (!is_sucess) {
        VOXEL_LOC near_position = position;
        if (loc_xyz[0] > (current_octo->voxel_center_[0] + current_octo->quater_length_)) {
          near_position.x = near_position.x + 1;
        } else if (loc_xyz[0] < (current_octo->voxel_center_[0] - current_octo->quater_length_)) {
          near_position.x = near_position.x - 1;
        }
        if (loc_xyz[1] > (current_octo->voxel_center_[1] + current_octo->quater_length_)) {
          near_position.y = near_position.y + 1;
        } else if (loc_xyz[1] < (current_octo->voxel_center_[1] - current_octo->quater_length_)) {
          near_position.y = near_position.y - 1;
        }
        if (loc_xyz[2] > (current_octo->voxel_center_[2] + current_octo->quater_length_)) {
          near_position.z = near_position.z + 1;
        } else if (loc_xyz[2] < (current_octo->voxel_center_[2] - current_octo->quater_length_)) {
          near_position.z = near_position.z - 1;
        }
        auto iter_near = voxel_map.find(near_position);
        if (iter_near != voxel_map.end()) {
          build_single_residual(pv,
                                iter_near->second,
                                0,
                                max_layer,
                                sigma_num,
                                is_sucess,
                                prob,
                                single_ptpl);
        }
      }
      if (is_sucess) {
        mylock.lock();
        useful_ptpl[i] = true;
        all_ptpl_list[i] = single_ptpl;
        mylock.unlock();
      } else {
        mylock.lock();
        useful_ptpl[i] = false;
        mylock.unlock();
      }
    }
  }
  for (size_t i = 0; i < useful_ptpl.size(); i++) {
    if (useful_ptpl[i]) {
      ptpl_list.push_back(all_ptpl_list[i]);
    }
  }
}

void BuildResidualListNormal(const unordered_map<VOXEL_LOC, OctoTree*>& voxel_map,
                             const double voxel_size,
                             const double sigma_num,
                             const int max_layer,
                             const std::vector<pointWithCov>& pv_list,
                             std::vector<ptpl>& ptpl_list,
                             std::vector<Eigen::Vector3d>& non_match) {
  ptpl_list.clear();
  std::vector<size_t> index(pv_list.size());
  for (size_t i = 0; i < pv_list.size(); ++i) {
    pointWithCov pv = pv_list[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = pv.point_world[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t) loc_xyz[0], (int64_t) loc_xyz[1], (int64_t) loc_xyz[2]);
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end()) {
      OctoTree* current_octo = iter->second;
      ptpl single_ptpl;
      bool is_sucess = false;
      double prob = 0;
      build_single_residual(pv,
                            current_octo,
                            0,
                            max_layer,
                            sigma_num,
                            is_sucess,
                            prob,
                            single_ptpl);

      if (!is_sucess) {
        VOXEL_LOC near_position = position;
        if (loc_xyz[0] > (current_octo->voxel_center_[0] + current_octo->quater_length_)) {
          near_position.x = near_position.x + 1;
        } else if (loc_xyz[0] < (current_octo->voxel_center_[0] - current_octo->quater_length_)) {
          near_position.x = near_position.x - 1;
        }
        if (loc_xyz[1] > (current_octo->voxel_center_[1] + current_octo->quater_length_)) {
          near_position.y = near_position.y + 1;
        } else if (loc_xyz[1] < (current_octo->voxel_center_[1] - current_octo->quater_length_)) {
          near_position.y = near_position.y - 1;
        }
        if (loc_xyz[2] > (current_octo->voxel_center_[2] + current_octo->quater_length_)) {
          near_position.z = near_position.z + 1;
        } else if (loc_xyz[2] < (current_octo->voxel_center_[2] - current_octo->quater_length_)) {
          near_position.z = near_position.z - 1;
        }
        auto iter_near = voxel_map.find(near_position);
        if (iter_near != voxel_map.end()) {
          build_single_residual(pv,
                                iter_near->second,
                                0,
                                max_layer,
                                sigma_num,
                                is_sucess,
                                prob,
                                single_ptpl);
        }
      }
      if (is_sucess) {
        ptpl_list.push_back(single_ptpl);
      } else {
        non_match.push_back(pv.point_world);
      }
    }
  }
}

void CalcVectQuation(const Eigen::Vector3d& x_vec,
                     const Eigen::Vector3d& y_vec,
                     const Eigen::Vector3d& z_vec,
                     geometry_msgs::Quaternion& q) {
  Eigen::Matrix3d rot;
  rot << x_vec(0), x_vec(1), x_vec(2), y_vec(0), y_vec(1), y_vec(2), z_vec(0), z_vec(1), z_vec(2);
  Eigen::Matrix3d rotation = rot.transpose();
  Eigen::Quaterniond eq(rotation);
  q.w = eq.w();
  q.x = eq.x();
  q.y = eq.y();
  q.z = eq.z();
}

void CalcQuation(const Eigen::Vector3d& vec, const int axis, geometry_msgs::Quaternion& q) {
  Eigen::Vector3d x_body = vec;
  Eigen::Vector3d y_body(1, 1, 0);
  if (x_body(2) != 0) {
    y_body(2) = -(y_body(0) * x_body(0) + y_body(1) * x_body(1)) / x_body(2);
  } else {
    if (x_body(1) != 0) {
      y_body(1) = -(y_body(0) * x_body(0)) / x_body(1);
    } else {
      y_body(0) = 0;
    }
  }
  y_body.normalize();
  Eigen::Vector3d z_body = x_body.cross(y_body);
  Eigen::Matrix3d rot;

  rot << x_body(0), x_body(1), x_body(2), y_body(0), y_body(1), y_body(2), z_body(0), z_body(1),
      z_body(2);
  Eigen::Matrix3d rotation = rot.transpose();
  if (axis == 2) {
    Eigen::Matrix3d rot_inc;
    rot_inc << 0, 0, 1, 0, 1, 0, -1, 0, 0;
    rotation = rotation * rot_inc;
  }
  Eigen::Quaterniond eq(rotation);
  q.w = eq.w();
  q.x = eq.x();
  q.y = eq.y();
  q.z = eq.z();
}

void pubSinglePlane(visualization_msgs::MarkerArray& plane_pub,
                    const std::string plane_ns,
                    const Plane& single_plane,
                    const float alpha,
                    const Eigen::Vector3d rgb) {
  visualization_msgs::Marker plane;
  plane.header.frame_id = "camera_init";
  plane.header.stamp = ros::Time();
  plane.ns = plane_ns;
  plane.id = single_plane.id;
  plane.type = visualization_msgs::Marker::CYLINDER;
  plane.action = visualization_msgs::Marker::ADD;
  plane.pose.position.x = single_plane.center[0];
  plane.pose.position.y = single_plane.center[1];
  plane.pose.position.z = single_plane.center[2];
  geometry_msgs::Quaternion q;
  CalcVectQuation(single_plane.x_normal, single_plane.y_normal, single_plane.normal, q);
  plane.pose.orientation = q;
  plane.scale.x = 3 * sqrt(single_plane.max_eigen_value);
  plane.scale.y = 3 * sqrt(single_plane.mid_eigen_value);
  plane.scale.z = 2 * sqrt(single_plane.min_eigen_value);
  plane.color.a = alpha;
  plane.color.r = rgb(0);
  plane.color.g = rgb(1);
  plane.color.b = rgb(2);
  plane.lifetime = ros::Duration();
  plane_pub.markers.push_back(plane);
}

void pubNoPlaneMap(const std::unordered_map<VOXEL_LOC, OctoTree*>& feat_map,
                   const ros::Publisher& plane_map_pub) {
  int id = 0;
  ros::Rate loop(500);
  float use_alpha = 0.8;
  visualization_msgs::MarkerArray voxel_plane;
  voxel_plane.markers.reserve(1000000);
  for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++) {
    if (!iter->second->plane_ptr_->is_plane) {
      for (uint i = 0; i < 8; i++) {
        if (iter->second->leaves_[i] != nullptr) {
          OctoTree* temp_octo_tree = iter->second->leaves_[i];
          if (!temp_octo_tree->plane_ptr_->is_plane) {
            for (uint j = 0; j < 8; j++) {
              if (temp_octo_tree->leaves_[j] != nullptr) {
                if (!temp_octo_tree->leaves_[j]->plane_ptr_->is_plane) {
                  Eigen::Vector3d plane_rgb(1, 1, 1);
                  pubSinglePlane(voxel_plane,
                                 "no_plane",
                                 *(temp_octo_tree->leaves_[j]->plane_ptr_),
                                 use_alpha,
                                 plane_rgb);
                }
              }
            }
          }
        }
      }
    }
  }
  plane_map_pub.publish(voxel_plane);
  loop.sleep();
}

void pubVoxelMap(const std::unordered_map<VOXEL_LOC, OctoTree*>& voxel_map,
                 const int pub_max_voxel_layer,
                 const ros::Publisher& plane_map_pub) {
  double max_trace = 0.25;
  double pow_num = 0.2;
  ros::Rate loop(500);
  float use_alpha = 0.8;
  visualization_msgs::MarkerArray voxel_plane;
  voxel_plane.markers.reserve(1000000);
  std::vector<Plane> pub_plane_list;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    GetUpdatePlane(iter->second, pub_max_voxel_layer, pub_plane_list);
  }
  for (size_t i = 0; i < pub_plane_list.size(); i++) {
    V3D plane_cov = pub_plane_list[i].plane_cov.block<3, 3>(0, 0).diagonal();
    double trace = plane_cov.sum();
    if (trace >= max_trace) {
      trace = max_trace;
    }
    trace = trace * (1.0 / max_trace);
    trace = pow(trace, pow_num);
    uint8_t r, g, b;
    mapJet(trace, 0, 1, r, g, b);
    Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
    double alpha;
    if (pub_plane_list[i].is_plane) {
      alpha = use_alpha;
    } else {
      alpha = 0;
    }
    pubSinglePlane(voxel_plane, "plane", pub_plane_list[i], alpha, plane_rgb);
  }
  plane_map_pub.publish(voxel_plane);
  loop.sleep();
}

void pubPlaneMap(const std::unordered_map<VOXEL_LOC, OctoTree*>& feat_map,
                 const ros::Publisher& plane_map_pub) {
  OctoTree* current_octo = nullptr;

  double max_trace = 0.25;
  double pow_num = 0.2;
  ros::Rate loop(500);
  float use_alpha = 1.0;
  visualization_msgs::MarkerArray voxel_plane;
  voxel_plane.markers.reserve(1000000);

  for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++) {
    if (iter->second->plane_ptr_->is_update) {
      Eigen::Vector3d normal_rgb(0.0, 1.0, 0.0);

      V3D plane_cov = iter->second->plane_ptr_->plane_cov.block<3, 3>(0, 0).diagonal();
      double trace = plane_cov.sum();
      if (trace >= max_trace) {
        trace = max_trace;
      }
      trace = trace * (1.0 / max_trace);
      trace = pow(trace, pow_num);
      uint8_t r, g, b;
      mapJet(trace, 0, 1, r, g, b);
      Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
      // Eigen::Vector3d plane_rgb(1, 0, 0);
      float alpha = 0.0;
      if (iter->second->plane_ptr_->is_plane) {
        alpha = use_alpha;
      } else {
        // std::cout << "delete plane" << std::endl;
      }
      pubSinglePlane(voxel_plane, "plane", *(iter->second->plane_ptr_), alpha, plane_rgb);

      iter->second->plane_ptr_->is_update = false;
    } else {
      for (uint i = 0; i < 8; i++) {
        if (iter->second->leaves_[i] != nullptr) {
          if (iter->second->leaves_[i]->plane_ptr_->is_update) {
            Eigen::Vector3d normal_rgb(0.0, 1.0, 0.0);

            V3D plane_cov = iter->second->leaves_[i]
                                ->plane_ptr_->plane_cov.block<3, 3>(0, 0)
                                .diagonal();
            double trace = plane_cov.sum();
            if (trace >= max_trace) {
              trace = max_trace;
            }
            trace = trace * (1.0 / max_trace);
            // trace = (max_trace - trace) / max_trace;
            trace = pow(trace, pow_num);
            uint8_t r, g, b;
            mapJet(trace, 0, 1, r, g, b);
            Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
            plane_rgb << 0, 1, 0;
            // fabs(iter->second->leaves_[i]->plane_ptr_->normal[0]),
            //     fabs(iter->second->leaves_[i]->plane_ptr_->normal[1]),
            //     fabs(iter->second->leaves_[i]->plane_ptr_->normal[2]);
            float alpha = 0.0;
            if (iter->second->leaves_[i]->plane_ptr_->is_plane) {
              alpha = use_alpha;
            } else {
              // std::cout << "delete plane" << std::endl;
            }
            pubSinglePlane(voxel_plane,
                           "plane",
                           *(iter->second->leaves_[i]->plane_ptr_),
                           alpha,
                           plane_rgb);
            // loop.sleep();
            iter->second->leaves_[i]->plane_ptr_->is_update = false;
            // loop.sleep();
          } else {
            OctoTree* temp_octo_tree = iter->second->leaves_[i];
            for (uint j = 0; j < 8; j++) {
              if (temp_octo_tree->leaves_[j] != nullptr) {
                if (temp_octo_tree->leaves_[j]->octo_state_ == 0 &&
                    temp_octo_tree->leaves_[j]->plane_ptr_->is_update) {
                  if (temp_octo_tree->leaves_[j]->plane_ptr_->is_plane) {
                    // std::cout << "subsubplane" << std::endl;
                    Eigen::Vector3d normal_rgb(0.0, 1.0, 0.0);
                    V3D plane_cov = temp_octo_tree->leaves_[j]
                                        ->plane_ptr_->plane_cov.block<3, 3>(0, 0)
                                        .diagonal();
                    double trace = plane_cov.sum();
                    if (trace >= max_trace) {
                      trace = max_trace;
                    }
                    trace = trace * (1.0 / max_trace);
                    // trace = (max_trace - trace) / max_trace;
                    trace = pow(trace, pow_num);
                    uint8_t r, g, b;
                    mapJet(trace, 0, 1, r, g, b);
                    Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
                    plane_rgb << 0, 0, 1;
                    float alpha = 0.0;
                    if (temp_octo_tree->leaves_[j]->plane_ptr_->is_plane) {
                      alpha = use_alpha;
                    }

                    pubSinglePlane(voxel_plane,
                                   "plane",
                                   *(temp_octo_tree->leaves_[j]->plane_ptr_),
                                   alpha,
                                   plane_rgb);
                    // loop.sleep();
                    temp_octo_tree->leaves_[j]->plane_ptr_->is_update = false;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  plane_map_pub.publish(voxel_plane);
  // plane_map_pub.publish(voxel_norm);
  loop.sleep();
  // cout << "[Map Info] Plane counts:" << plane_count
  //      << " Sub Plane counts:" << sub_plane_count
  //      << " Sub Sub Plane counts:" << sub_sub_plane_count << endl;
  // cout << "[Map Info] Update plane counts:" << update_count
  //      << "total size: " << feat_map.size() << endl;
}

std::tuple<int, int, int> GenerateBoundaryRGB(size_t idx) {
  std::hash<size_t> hasher;
  std::mt19937 rng(hasher(idx));
  std::uniform_int_distribution<int> dist(0, 255);

  int r = dist(rng);
  int g = dist(rng);
  int b = dist(rng);

  return std::make_tuple(r, g, b);
}

void SetVoxelPlaneVisualizationInfo(visualization_msgs::MarkerArray& plane_pub,
                                    const Plane* const single_plane,
                                    const Eigen::Vector3d& plane_position,
                                    double plane_scale,
                                    int viz_plane_id) {
  visualization_msgs::Marker plane;
  plane.header.frame_id = "camera_init";
  plane.header.stamp = ros::Time();
  plane.ns = "merged voxel";
  plane.id = viz_plane_id;
  plane.type = visualization_msgs::Marker::CYLINDER;
  plane.action = visualization_msgs::Marker::ADD;

  auto color = GenerateBoundaryRGB(single_plane->merge_plane_id);
  int r = std::get<0>(color);
  int g = std::get<1>(color);
  int b = std::get<2>(color);

  Eigen::Vector3f plane_rgb(r / 256.f, g / 256.f, b / 256.f);

  plane.color.r = plane_rgb(0);
  plane.color.g = plane_rgb(1);
  plane.color.b = plane_rgb(2);

  plane.pose.position.x = plane_position[0];
  plane.pose.position.y = plane_position[1];
  plane.pose.position.z = plane_position[2];

  const auto& plane_normal = single_plane->normal.normalized();
  Eigen::Vector3d base(0, 0, 1);
  size_t max_index = 0;
  plane_normal.cwiseAbs().maxCoeff(&max_index);
  if (max_index == 2) {
    base << 1.f, 0.f, 0.f;
  }
  Eigen::Vector3d x_normal = (plane_normal.cross(base)).normalized();
  Eigen::Vector3d y_normal = plane_normal.cross(x_normal);

  geometry_msgs::Quaternion q;
  CalcVectQuation(x_normal, y_normal, plane_normal, q);

  plane.pose.orientation = q;
  plane.scale.x = plane_scale;
  plane.scale.y = plane_scale;
  plane.scale.z = 0.01;
  plane.color.a = 0.8;

  plane_pub.markers.push_back(plane);
}

void PublishMergedVoxelPlane(
    std::unordered_map<VOXEL_LOC, OctoTree*>& feat_map,
    const std::unordered_map<int, std::unordered_set<NodeLoc>>& merged_voxels,
    const ros::Publisher& plane_map_pub,
    double voxel_size) {
  if (merged_voxels.empty()) {
    return;
  }

  visualization_msgs::MarkerArray voxel_plane;
  voxel_plane.markers.reserve(1000000);

  int viz_plane_id = 0;

  for (const auto& plane_i : merged_voxels) {
    if (plane_i.second.empty()) {
      continue;
    }

    auto node_loc_it = *(plane_i.second.begin());

    const OctoTree* const current_node = GetVoxelPlaneNode(feat_map, node_loc_it);

    auto current_plane = current_node->plane_ptr_;

    for (const auto& content_loc : plane_i.second) {
      VOXEL_LOC v(content_loc.x, content_loc.y, content_loc.z);

      double plane_scale = voxel_size;

      Eigen::Vector3d viz_center;
      viz_center << (0.5f + v.x) * plane_scale, (0.5f + v.y) * plane_scale,
          (0.5f + v.z) * plane_scale;

      Eigen::Vector3d n = current_plane->normal;

      double t = n.dot(current_plane->center) - n.dot(viz_center);

      Eigen::Vector3d plane_pose = viz_center + t * n;

      if (content_loc.leaf_num > 0) {
        plane_scale *= 0.5f;
      }

      SetVoxelPlaneVisualizationInfo(voxel_plane,
                                     current_plane,
                                     plane_pose,
                                     plane_scale,
                                     viz_plane_id);

      viz_plane_id++;
    }
  }
  plane_map_pub.publish(voxel_plane);
}

void calcBodyCov(Eigen::Vector3d& pb,
                 const float range_inc,
                 const float degree_inc,
                 Eigen::Matrix3d& cov) {
  float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
  float range_var = range_inc * range_inc;
  Eigen::Matrix2d direction_var;
  direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0, pow(sin(DEG2RAD(degree_inc)), 2);
  Eigen::Vector3d direction(pb);
  direction.normalize();
  Eigen::Matrix3d direction_hat;
  direction_hat << 0, -direction(2), direction(1), direction(2), 0, -direction(0), -direction(1),
      direction(0), 0;
  Eigen::Vector3d base_vector1(1, 1, -(direction(0) + direction(1)) / direction(2));
  base_vector1.normalize();
  Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
  base_vector2.normalize();
  Eigen::Matrix<double, 3, 2> N;
  N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1), base_vector1(2),
      base_vector2(2);
  Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
  cov = direction * range_var * direction.transpose() + A * direction_var * A.transpose();
};

#endif