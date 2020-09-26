require(Rcpp)
require(RcppEigen)
require(StanHeaders)
library(foreach)
library(doParallel)
library(optimx)
library(optimr)

libs=c('BH','Rcpp','RcppEigen','StanHeaders')
home_dir=Sys.getenv("HOME")
flibs=paste('-I"', home_dir, '/miniconda3/lib/R/library/',libs,'/include"',sep='')
Sys.setenv("PKG_CXXFLAGS"=paste(flibs,collapse = ' '))

folderpath='/R_bks'
filepath=''
packagepath = '/bksgpR'
home='~/BKS/src'
home2=paste(home,folderpath,sep='')
setwd(home2)
