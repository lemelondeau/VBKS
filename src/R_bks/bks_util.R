start_save_file=function(file_str)
{
  setEPS()
  postscript(paste(file_str,'.eps',sep=''))
}
end_save_file=function()
{
  dev.off()
}

#' normalize features
#'
#' @param x a matrix of feature vectors (column)
#'
#' @return normalized feature
#' @export
#'
#' @examples
normalize_col=function(x)
{
  #normalize the inputs
  x_mu = apply(x, 2, mean)
  x_sd = apply(x, 2, sd)
  x_sd[ x_sd == 0 ] = 1
  xx = (x - matrix(x_mu, nrow(x), ncol(x), byrow = TRUE))/
    matrix(x_sd, nrow(x), ncol(x), byrow = TRUE)
  return(xx)
}



make_random_name <- function(key='random',n=1, len=8)
{
  randomString <- c(1:n)                  # initialize vector
  for (i in 1:n)
  {
    randomString[i] <- paste(sample(c(0:9, letters, LETTERS),
                                    len, replace=TRUE),
                             collapse="")
  }
  randomString=paste(key,randomString,'.csv', sep='')
  return(randomString)
}

get_hyper_from_stop<-function(file_str)
{
  aa=read.csv(file=file_str,header=TRUE)
  return(aa$hyper[-1])
}


get_hyper_from_file<-function(file_str)
{
  aa=read.csv(file=file_str,header=TRUE)
  return(aa$x)
}

decode_kernel=function(kern_type)
{
  k_vec=as.numeric(strsplit(as.character(kern_type), "")[[1]])
  root=k_vec[1]
  seq=NULL
  op=NULL

  if (length(k_vec)>1)
  {
    seq=k_vec[seq(3,length(k_vec),2)]
    op=k_vec[seq(2,length(k_vec),2)]
  }
  list(root=root,seq=seq,op=op)
}

init_hyp=function (root, sequence=NULL,op=NULL, init_theta=NULL)
{


  if(is.null(init_theta))
  {
    theta1=log(c(10,1)) #SE
    theta2=log(c(1,20,1)) #RQ
    theta3=log(20) #LIN
    theta4=log(c(4,15,1)) #PER
  } else {
    theta1=log(init_theta[1:2]) #SE
    theta2=log(init_theta[3:5]) #RQ
    theta3=log(init_theta[6]) #LIN
    theta4=log(init_theta[7:9]) #PER
  }

  thetas=list(theta1,theta2,theta3,theta4)
  theta = thetas[[root]]

  if (!is.null(sequence))
  {
    for (i in sequence[seq(1,length(sequence))])
    {
      theta = c(theta,thetas[[i]])
    }
  }
  theta
}

compose_kernel=function (root,seq=NULL,op=NULL)
{
  #SE, RQ, Lin, Per
  hyper_num=c(2,3,1,3);
  hypnum=sum(hyper_num[c(root,seq)])
  kernel=list(root=root,seq=c(seq,-1),op=c(op,-1))
  hyper=c(0,log(rep(1,hypnum)),log(2))
  list(hyper=hyper,kernel=kernel)
}

init_hyper_random = function(x, y, seed, init_kern_theta_manual)
{
  set.seed(seed)
  xlim=5
  per_lim = min(40, (max(x)-min(x))/4)
  se_len=xlim*runif(1)+init_kern_theta_manual[1]
  se_sig=init_kern_theta_manual[2]
  rq_a=init_kern_theta_manual[3]
  rq_len=xlim*runif(1)+init_kern_theta_manual[4]
  rq_sig=init_kern_theta_manual[5]
  lin_len=xlim*runif(1)+init_kern_theta_manual[6]
  per_len=xlim*runif(1)+init_kern_theta_manual[7]
  per_p=per_lim*runif(1)+1
  per_sig=init_kern_theta_manual[9]
  init_kern_theta=c(se_len,se_sig, #SE
                    rq_a, rq_len,rq_sig, #RQ
                    lin_len, #LIN
                    per_len,per_p,per_sig)#PER
  return(init_kern_theta)
}
# results_folder
#    k1_bxx_sxxx_uxxx_rxx_xxxxxxxx
#        kern1_xxxxxxxx.csv_iter
#    k2_bxx_sxxx_uxxx_rxx_xxxxxxxx
#        kern2_xxxxxxxx.csv_iter
#    k3_bxx_sxxx_uxxx_rxx_xxxxxxxx
#        kern3_xxxxxxxx.csv_iter
#    ...
#    iterx_best_kernel (optional)
# Extract local VLB for specified kernels, or all kernels if kerns is unspecified
# If exclude_prev_best is true, the local VLB of kernels that were the best of previous iterations (and therefore have been expanded already) will be excluded
extract_local_info=function (results_folder, kerns=NULL, exclude_prev_best=FALSE)
{
  spec={}
  kernels={}
  L_i={}
  times={}

  if (exclude_prev_best) {
    exclude=c()
    for (file in list.files(path=results_folder, pattern='iter_[1-9]*_results', full.names=TRUE)) {
      con <- file(file,"r")
      exclude = c(exclude, readLines(con,n=1))
      close(con)
    }
  }

  for (case in list.files(path = results_folder, pattern = "k[1-9]*_.*", full.names = FALSE))
  {
    case_kern=substring(unlist(strsplit(case, '_'))[1],2)
    if ((!is.null(kerns) && !as.numeric(case_kern) %in% kerns) || (exclude_prev_best && case_kern %in% exclude)) {
      next
    }
    kern=as.numeric(case_kern)
    for (file in list.files(paste(results_folder,case,sep='/'), full.names = FALSE)) {
      if (grepl("_iter", file))
      {
        f = file(paste(results_folder,case,file,sep='/'), 'r')
        L_i_val=-Inf
        while (TRUE)
        {
          line = readLines(f, n = 1)
          if (length(line) == 0)
          {
            break
          }
          splt = unlist(strsplit(line,split='_', fixed=TRUE))
          L_i_val=max(L_i_val, as.numeric(splt[length(splt)]))
          # L_i_val = as.numeric(splt[length(splt)])
        }
        kernels = c(kernels, kern)
        L_i = c(L_i, L_i_val)
        close(f)
      }

      # Extract time information
      if (grepl("_cfg", file))
      {
        lines=readLines(paste(results_folder,case,file,sep='/'))
        times=c(times, unlist(strsplit(lines[grep("time", lines)], ','))[2])
      }

    }
  }
  spec$kernels=kernels
  spec$L_i=L_i
  spec$times=times
  spec$size_k=length(kernels)
  return(spec)

}
extract_local_info_by_epoch = function(results_folder, step, kerns=NULL, exclude_prev_best=FALSE){
  spec={}
  kernels={}
  L_i={}
  times={}

  if (exclude_prev_best) {
    exclude=c()
    for (file in list.files(path=results_folder, pattern='iter_[1-9]*_results', full.names=TRUE)) {
      con <- file(file,"r")
      exclude = c(exclude, readLines(con,n=1))
      close(con)
    }
  }

  # folders for each kernel
  all_kernel_folder = list.dirs(path = results_folder,
                                full.names = TRUE,
                                recursive = FALSE)
  L_i = {}
  kernels = {}
  times = {}
  for (j in (1:length(all_kernel_folder))) {

    rmse_file = list.files(all_kernel_folder[j], '*csv_rmse', full.names = TRUE)[1]#fullname
    iter_filename=list.files(all_kernel_folder[j], '*csv_iter', full.names = TRUE)[1]#fullname
    iter_filename2=list.files(all_kernel_folder[j], '*csv_iter', full.names = FALSE)[1]
    cfg_filename = list.files(all_kernel_folder[j], '*_cfg', full.names = TRUE,recursive =TRUE)[1]
    case_kern = substring(unlist(strsplit(iter_filename2, '_'))[1], 5)
    
    if(file.exists(cfg_filename)){

      d=read.csv(cfg_filename,header = TRUE)
      # print(dim(d))
      curr_time = d[dim(d)[1],2]
      times = c(times, curr_time)
      kern = as.numeric(case_kern)
      kernels = c(kernels, kern)
      elbos_seq = read.csv(iter_filename, header = FALSE,stringsAsFactors=FALSE)
      elbos_line = elbos_seq[step,]
      splt = unlist(strsplit(elbos_line, split = '_', fixed = TRUE))
      L_i_val = as.numeric(splt[length(splt)])
      L_i = c(L_i, L_i_val)
   }
 }

  spec$kernels=kernels
  spec$L_i=L_i
  spec$times=times
  spec$size_k=length(kernels)
  return(spec)
}

extract_local_info_by_batch = function(results_folder, step, kerns=NULL, exclude_prev_best=FALSE){
  spec={}
  kernels={}
  L_i={}
  times={}

  if (exclude_prev_best) {
    exclude=c()
    for (file in list.files(path=results_folder, pattern='iter_[1-9]*_results', full.names=TRUE)) {
      con <- file(file,"r")
      exclude = c(exclude, readLines(con,n=1))
      close(con)
    }
  }

  # folders for each kernel
  all_kernel_folder = list.dirs(path = results_folder,
                                full.names = TRUE,
                                recursive = FALSE)
  L_i = {}
  kernels = {}
  times = {}
  for (j in (1:length(all_kernel_folder))) {

    rmse_file = list.files(all_kernel_folder[j], '*csv_rmse', full.names = TRUE)[1]#fullname
    iter_filename=list.files(all_kernel_folder[j], '*csv_iter_batch_', full.names = TRUE)[1]#fullname
    iter_filename2=list.files(all_kernel_folder[j], '*csv_iter_batch_', full.names = FALSE)[1]
    cfg_filename = list.files(all_kernel_folder[j], '*_cfg', full.names = TRUE,recursive =TRUE)[1]
    case_kern = substring(unlist(strsplit(iter_filename2, '_'))[1], 5)
    
    if(file.exists(cfg_filename)){

      d=read.csv(cfg_filename,header = TRUE)
      # print(dim(d))
      curr_time = d[dim(d)[1],2]
      times = c(times, curr_time)
      kern = as.numeric(case_kern)
      kernels = c(kernels, kern)
      elbos_seq = read.csv(iter_filename, header = FALSE,stringsAsFactors=FALSE)
      elbos_line = elbos_seq[step,]
      splt = unlist(strsplit(elbos_line, split = '_', fixed = TRUE))
      splt = unlist(strsplit(splt, split='-',fixed=TRUE))
      L_i_val = as.numeric(splt[length(splt)])
      L_i = c(L_i, - L_i_val)
   }
 }

  spec$kernels=kernels
  spec$L_i=L_i
  spec$times=times
  spec$size_k=length(kernels)
  return(spec)
}

write_VLB_table_latex = function(results_folder, file_name, u)
{
  vlb_mat = matrix(nrow=4, ncol=9, data="-")
  spec=extract_local_info(results_folder)
  for (i in 1:length(spec$kernels)) {
    kern=as.numeric(strsplit(as.character(spec$kernels[i]), "")[[1]])

    if (length(kern) == 3) {
      k1 = kern[1]
      k2 = kern[3]
      op = kern[2]
      vlb_mat[k1, 1 + (op - 1)* 4 + k2] = spec$L_i[i]
    } else {
      k1 = kern[1]
      vlb_mat[k1, 1] = spec$L_i[i]
    }
  }

  rows = c("SE", "RQ", "LIN", "PER")
  sink(paste(file_name, ".txt", sep=""))
  cat("\\begin{table}\n")
  cat("\\centering\n")
  cat("\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|}\n")
  cat("\\hline\n")
  cat("& - & +SE & +RQ & +LIN & +PER & xSE & xRQ & xLIN & xPER \\\\ \n")
  cat("\\hline \n")
  cat("$|u|$ =", u, "& & & & & & & & & \\\\ \n")
  for (i in 1:nrow(vlb_mat)) {
    cat(rows[i], "& ")
    for (j in 1:(ncol(vlb_mat)-1)) {
      cat(vlb_mat[i,j], "& ")
    }
    cat(vlb_mat[i, ncol(vlb_mat)], "\\\\ \n")
  }
  cat("\\hline \n")
  cat("\\end{tabular} \n")
  cat("\\end{table} \n")
  sink()
}

soft_max_from_samples=function(samples) {
  n_samples=length(samples[1,])
  n_kernels=length(samples[,1])

  soft_maxes=matrix(nrow=n_kernels,ncol=n_samples)

  for (i in seq(1, n_samples)) {
    denom = 0
    for (j in seq(1, n_kernels)) {
      denom = denom + exp(samples[j,i])
    }
    for (j in seq(1, n_kernels)) {
      soft_maxes[j,i] = exp(samples[j,i]) / denom
    }
  }

  return (rowMeans(soft_maxes))
}

# settings of the form su512sb100ss200r55mi200m3sm3ssg100mig20smb32
parse_settings=function(settings) {
  settings_split = strsplit(settings, "[a-z]+")[[1]]
  set = {}
  set$sizeu = strtoi(settings_split[2])
  set$sizeb = strtoi(settings_split[3])
  set$sizes = strtoi(settings_split[4])
  set$rnum = strtoi(settings_split[5])
  set$maxit = strtoi(settings_split[6])
  set$mode = strtoi(settings_split[7])
  set$smode = strtoi(settings_split[8])
  set$sizesglobal = strtoi(settings_split[9])
  set$maxitglobal = strtoi(settings_split[10])
  set$sizemb = strtoi(settings_split[11])
  set$sgdmode = strtoi(settings_split[12])
  set$nstarts = strtoi(settings_split[13])
  set$nummb = strtoi(settings_split[14])
  return (set)
}
