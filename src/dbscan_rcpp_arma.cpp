// Title: Implementation of DBSCAN in C++ of DBSCAN
// Author: Mariana Castaneda-Guzman
// Last Updated: 4/14/20223

// Brief description: This code is a C++ implementation of the DBSCAN
// (Density-Based Spatial Clustering of Applications with Noise) algorithm.
// DBSCAN is a clustering algorithm that groups together points that are closely
// packed together while marking points that are not in a cluster as noise.


#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;


#include <vector>
#include <string>

// Point Class Description:
//
// The Point class represents a point in a n-dimensional space. Each point can
// be assigned a cluster and a label. The cluster variable indicates the cluster
// to which the point belongs, and the label variable is a string that can be
// used to describe the type of the point (e.g., "core", "border", or "noise").
// The id variable is an integer that uniquely identifies the point. The
// neighbors vector stores the indices of neighboring points of the current
// point.
//
// The Point class provides public member functions to get and set the cluster,
// label, and id variables, as well as functions to add a neighboring
// point to the neighbors vector, get the size of the neighborhood, and get the
// indices of neighboring points.
//
// The isCore() member function checks whether the current point is a "core"
// point, which is defined as a point with at least minPts neighboring points
// within a certain radius eps.
//
// The hasNeighbors() member function checks whether the current point has any
// neighboring points.
//
// The toString() and toString_n() member functions return a string
// representation of the point, which includes the cluster, label, id, and
// neighbors (for toString_n()).

class Point {
private:
  std::vector<int> neighbors;
  int id;

  
public:
  int cluster;
  std::string label;
  
  Point() {
    cluster = 0;
    label = "noise";
  }
  
  Point(int aCluster, std::string aLabel) {
    cluster = aCluster;
    label = aLabel;
  }
  
  void setCluster(int c) {
    cluster = c;
  }
  
  int getCluster() const {
    return cluster;
  }
  
  void setLabel(std::string l) {
    label = l;
  }
  
  std::string getLabel() const {
    return label;
  }
  
  void setId(int aId) {
    id = aId;
  }
  
  int getId() const {
    return id;
  }
  
  void addNeighbor(int n) {
    neighbors.push_back(n);
  }
  
  int getNeighborhoodSize() const {
    return neighbors.size();
  }
  
  std::vector<int> getNeighbors() const {
    return neighbors;
  }
  
  bool isCore() const {
    if (label == "core") {
      return true;
    }
    return false;
  }
  
  bool hasNeighbors() const {
    if (neighbors.size() > 0) {
      return true;
    } else {
      return false;
    }
  }
  
  std::string toString() const {
    return "Cluster: " + std::to_string(cluster) + ", Label: " + label;
  }
  
  std::string toString_n() const {
    std::string r = "ID: " + std::to_string(id) + ", Cluster: " + std::to_string(cluster) + ", Label: " + label + ", Neighbors: ";
    for (int item : neighbors) {
      r += std::to_string(item);
      r += " ";
    }
    return r;
  }
};


// Functions 

// std::vector<int> getNeighborhood(std::vector<Point>& points_dbscan, std::vector<int> v, unsigned int ctr);
std::vector<int> getNeighborhood(std::vector<Point>& points_dbscan, std::vector<int> v, unsigned int ctr);
std::vector< std::vector<double> > dfToVecVec(arma::mat& df);
double euclidean_distance(const std::vector<double>& pointA, const std::vector<double>& pointB);
double mahalanobis_distance(const std::vector<double>& pointA, const std::vector<double>& pointB, const arma::mat& cov_mat);


// [[Rcpp::export]]
arma::mat dbscan_train(arma::mat& df, double eps, int minPts,
                       std::string dis_metric = "euclidean"){
  
  std::vector< std::vector<double> > predictors = dfToVecVec(df);

  std::vector<Point> points_dbscan;
  
  arma::mat cov_mat;
  if (dis_metric == "mahalanobis") {
    cov_mat = arma::cov(df);
  }
  
  // Check if dis_metric is valid
  if (dis_metric != "euclidean" && dis_metric != "mahalanobis") {
    Rcpp::stop("Invalid distance metric.");
  }
  
  for(unsigned int i = 0; i < predictors.size(); i++){
    std::vector< std::vector<double> > temp;
    temp = predictors;
    
    temp.erase(temp.begin() + i);
    
    Point p;
    
    p.setId(i);
    
    for(unsigned int j = 0; j < temp.size(); j++){
      double d = eps + 1;
      
      if(dis_metric == "euclidean"){
        d = euclidean_distance(predictors[i], temp[j]);
      }else if( dis_metric == "mahalanobis"){
        d = mahalanobis_distance(predictors[i], predictors[j], cov_mat);
      }
      
      if (d <= eps) {
        int index = j < i ? j : j + 1;
        p.addNeighbor(index);
      }
      
    }
    
    points_dbscan.push_back(p);
  }
  
  
  // Label the points
  for(Point& item: points_dbscan){
    if(item.hasNeighbors() && item.getNeighborhoodSize() >= (minPts-1)){
      item.setLabel("core");
    }
  }
  
  // assign cores and labels
  
  int cluster = 1;
  
  for(Point& item : points_dbscan){
    
    if(item.isCore()){
      if(item.getCluster() == 0){
        
        item.setCluster(cluster);
        
       std::vector<int> v;
       std::vector<int> n = getNeighborhood(points_dbscan, v, item.getId());
        
        // Remove duplicates
        sort(n.begin(), n.end());
        n.erase(unique(n.begin(), n.end()), n.end());
        
        for(int h : n){
          // Points can only belong to one neighborhood, so if point has already
          // been determine to belong to another neighbor then we can not
          // change it
          if(points_dbscan[h].getCluster() == 0){
            points_dbscan[h].setCluster(item.getCluster());
          }
          
          if(!points_dbscan[h].isCore()){
            points_dbscan[h].setLabel("border");
          }
          
        }
        
        cluster++;
        
      }
    }
  }
  
 std::vector<int> ids;
 std::vector<int> clusters;
 std::vector<std::string> labels;
  
  for(Point item:points_dbscan){
    ids.push_back(item.getId());
    clusters.push_back(item.getCluster());
    labels.push_back(item.getLabel());
  }
  
  
  arma::mat final_mat(points_dbscan.size(), 3);
  for (unsigned int i = 0; i < points_dbscan.size(); i++) {
    final_mat(i, 0) = points_dbscan[i].getId();
    final_mat(i, 1) = points_dbscan[i].getCluster();
    final_mat(i, 2) = points_dbscan[i].getLabel() == "core" ? 1 : 0;
  }
  
  return final_mat;
  
}


// [[Rcpp::export]]
std::vector<int> dbscan_projection(arma::mat& new_data,
                                   arma::mat& train_data, 
                                   std::vector<int> dbscan_cores, 
                                   double eps, 
                                   std::string dis_metric = "euclidean"){
  
  std::vector<Point> points_dbscan_projection;
  
  std::vector< std::vector<double> > train_df = dfToVecVec(train_data);
  std::vector< std::vector<double> > new_df = dfToVecVec(new_data);
  
  
  arma::mat cov_mat;
  if (dis_metric == "mahalanobis") {
    cov_mat = arma::cov(train_data);
  }
  
  // Check if dis_metric is valid
  if (dis_metric != "euclidean" && dis_metric != "mahalanobis") {
    Rcpp::stop("Invalid distance metric.");
  }
  
  // iterate through all the cluster 
  for(unsigned int i = 0; i < new_df.size(); i++){
    Point p;
    
    p.setId(i);
    p.setCluster(0);
    
    unsigned int j = 0;
    while(p.getCluster() == 0 && j < dbscan_cores.size()){
      
      double d = eps + 1;
      
      if(dis_metric == "euclidean"){
        d = euclidean_distance(new_df[i], train_df[j]);
      }else if( dis_metric == "mahalanobis"){
        d = mahalanobis_distance(new_df[i], train_df[j], cov_mat);
      }      
      
      if(d <= eps){
        // assign cluster
        p.setCluster(dbscan_cores[j]);
      }
      
      j++;
    }
    
    points_dbscan_projection.push_back(p);
  }
  
  
  std::vector<int> clusters;
  
  for(Point item: points_dbscan_projection){
    clusters.push_back(item.getCluster());
  }
  
  
  return clusters; 
  
}


std::vector<int> getNeighborhood(std::vector<Point>& points_dbscan, 
                                 std::vector<int> v, unsigned int ctr){
  
  if(!points_dbscan[ctr].hasNeighbors()){
    return v;
  }else{
    std::vector<int> n = points_dbscan[ctr].getNeighbors();
    
    for(unsigned int i = 0; i < n.size(); i ++){
      
      
      // This checks if the neighbor is already on the list, and if so it does
      // not add it
      if(count(v.begin(), v.end(), n[i]) < 1) {
        if(!points_dbscan[n[i]].isCore()){
          v.push_back(n[i]);
        }else{
          v.push_back(n[i]);
          v = getNeighborhood(points_dbscan, v, n[i]);
        }
      }
      
    }
    
    
    return v;
    
  }
}



// [[Rcpp::export]]
std::vector< std::vector<double> > dfToVecVec(arma::mat& df) {
  int n = df.n_rows;
  int m = df.n_cols;
  std::vector< std::vector<double> > vecvec(n, std::vector<double>(m));
  
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      vecvec[i][j] = df(i,j);
    }
  }
  
  return vecvec;
}

// [[Rcpp::export]]
double euclidean_distance(const std::vector<double>& pointA, 
                          const std::vector<double>& pointB){
  
  double distance = 0.0;
  int n = pointA.size();
  
  for(int i = 0; i < n; i++){
    distance += pow(pointB[i] - pointA[i], 2);
  }
  
  return sqrt(distance);
}


// [[Rcpp::export]]
double mahalanobis_distance(const std::vector<double>& pointA,
                            const std::vector<double>& pointB,
                            const arma::mat& cov_mat) {

  int n = pointA.size();

  // convert std::vector to arma::colvec
  arma::colvec vecA(n);
  arma::colvec vecB(n);
  for(int i = 0; i < n; i++){
    vecA(i) = pointA[i];
    vecB(i) = pointB[i];
  }

  // calculate Mahalanobis distance
  arma::colvec diff = vecB - vecA;
  double distance = as_scalar(diff.t() * inv(cov_mat) * diff);
  distance = sqrt(distance);

  return distance;
}



/*** R
# library(factoextra)
# data("multishapes")
# 
# og_data <- multishapes[, 1:2]
# 
# plot(og_data)
# 
# rows_t <- sample(1:nrow(og_data), nrow(og_data)*0.7)
# df_train <- og_data[rows_t, ]
# 
# df_dbscan <- dbscan_train(df = as.matrix(og_data), eps = 0.15, minPts = 5,
#                           dis_metric = "euclidean")
# 
# p <- plot_dbscan(mod_data = og_data, mod = df_dbscan)
# p
# 
# p + labs(title = "DBSCAN with Euclidean Distance")


# df_dbscan <- dbscan_train(df = as.matrix(og_data), eps = 0.15, minPts = 7,
#                           dis_metric = "mahalanobis")
# 
# table(df_dbscan[, 2])
# 
# p2 <- plot_dbscan(mod_data = og_data, mod = df_dbscan)
# p2
# 
# p2 + labs(title = "DBSCAN with Mahalanobis Distance")

*/

