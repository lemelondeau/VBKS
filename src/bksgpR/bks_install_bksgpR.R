if ("bksgpR" %in% rownames(installed.packages()) == TRUE) {
  remove.packages("bksgpR")
}
if (dir.exists(paste(home2, packagepath, sep=''))) {
  unlink(paste(home2, packagepath, sep=''), recursive = TRUE)
}
Rcpp.package.skeleton("bksgpR", example_code = FALSE, cpp_files = c(paste(home2, "/bks_logml.cc", sep='')), attributes = TRUE, force = TRUE)
tar('bksgpR.tar.gz', 'bksgpR')
install.packages('bksgpR.tar.gz',repos=NULL)

#if installing on windows machine
#install_local('./bksgpR')
