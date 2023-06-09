// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// dbscan_train
arma::mat dbscan_train(arma::mat& df, double eps, int minPts, std::string dis_metric);
RcppExport SEXP _Marble_dbscan_train(SEXP dfSEXP, SEXP epsSEXP, SEXP minPtsSEXP, SEXP dis_metricSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type df(dfSEXP);
    Rcpp::traits::input_parameter< double >::type eps(epsSEXP);
    Rcpp::traits::input_parameter< int >::type minPts(minPtsSEXP);
    Rcpp::traits::input_parameter< std::string >::type dis_metric(dis_metricSEXP);
    rcpp_result_gen = Rcpp::wrap(dbscan_train(df, eps, minPts, dis_metric));
    return rcpp_result_gen;
END_RCPP
}
// dbscan_projection
std::vector<int> dbscan_projection(arma::mat& new_data, arma::mat& train_data, std::vector<int> dbscan_cores, double eps, std::string dis_metric);
RcppExport SEXP _Marble_dbscan_projection(SEXP new_dataSEXP, SEXP train_dataSEXP, SEXP dbscan_coresSEXP, SEXP epsSEXP, SEXP dis_metricSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type new_data(new_dataSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type train_data(train_dataSEXP);
    Rcpp::traits::input_parameter< std::vector<int> >::type dbscan_cores(dbscan_coresSEXP);
    Rcpp::traits::input_parameter< double >::type eps(epsSEXP);
    Rcpp::traits::input_parameter< std::string >::type dis_metric(dis_metricSEXP);
    rcpp_result_gen = Rcpp::wrap(dbscan_projection(new_data, train_data, dbscan_cores, eps, dis_metric));
    return rcpp_result_gen;
END_RCPP
}
// dfToVecVec
std::vector< std::vector<double> > dfToVecVec(arma::mat& df);
RcppExport SEXP _Marble_dfToVecVec(SEXP dfSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type df(dfSEXP);
    rcpp_result_gen = Rcpp::wrap(dfToVecVec(df));
    return rcpp_result_gen;
END_RCPP
}
// euclidean_distance
double euclidean_distance(const std::vector<double>& pointA, const std::vector<double>& pointB);
RcppExport SEXP _Marble_euclidean_distance(SEXP pointASEXP, SEXP pointBSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<double>& >::type pointA(pointASEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type pointB(pointBSEXP);
    rcpp_result_gen = Rcpp::wrap(euclidean_distance(pointA, pointB));
    return rcpp_result_gen;
END_RCPP
}
// mahalanobis_distance
double mahalanobis_distance(const std::vector<double>& pointA, const std::vector<double>& pointB, const arma::mat& cov_mat);
RcppExport SEXP _Marble_mahalanobis_distance(SEXP pointASEXP, SEXP pointBSEXP, SEXP cov_matSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<double>& >::type pointA(pointASEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type pointB(pointBSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type cov_mat(cov_matSEXP);
    rcpp_result_gen = Rcpp::wrap(mahalanobis_distance(pointA, pointB, cov_mat));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_Marble_dbscan_train", (DL_FUNC) &_Marble_dbscan_train, 4},
    {"_Marble_dbscan_projection", (DL_FUNC) &_Marble_dbscan_projection, 5},
    {"_Marble_dfToVecVec", (DL_FUNC) &_Marble_dfToVecVec, 1},
    {"_Marble_euclidean_distance", (DL_FUNC) &_Marble_euclidean_distance, 2},
    {"_Marble_mahalanobis_distance", (DL_FUNC) &_Marble_mahalanobis_distance, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_Marble(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
