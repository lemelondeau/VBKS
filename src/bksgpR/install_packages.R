
pkgs = c("Rcpp", "RcppEigen", "BH", "numDeriv", "foreach", "doParallel", "optimx", "optimr", "optparse","nloptr")
install.packages(pkgs)

pkgs = c("cowplot","devtools","corrplot")
install.packages(pkgs)

# install.packages("StanHeaders", version="2.17.1")
packageurl <- "http://cran.r-project.org/src/contrib/Archive/StanHeaders/StanHeaders_2.17.2.tar.gz"
install.packages(packageurl, repos=NULL, type="source")

source('./env/env_maplecgpu.R')

source('./bks_install_bksgpR.R')
