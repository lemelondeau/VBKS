# Computes equation 34 (with partial derivatives) using pre-computed values for L_i
# hyper is a vector containing:
#   mean_g (size_k)
#   Choleskey factor (lower triangular) of cov_g (size_k(size_k+1)/2)
compute_L_global<-function(hyper,spec,samplesize=100,seed=1,outfile='random.csv')
{
  # For each sample of g:
  #   Compute inner term (bks_global_inner)
  #   take the mean of all samples
  # calculate KL
  # save to file, return
  parts1=matrix(nrow=length(hyper)+1,ncol=samplesize)
  parts2=matrix(nrow=length(hyper)+1,ncol=samplesize)
  size_k=spec$size_k
  #samples=matrix(nrow=size_k,ncol=samplesize)
  set.seed(seed)
  pre_samples=matrix(rnorm((size_k)*samplesize),nrow=(size_k),ncol=samplesize)

  comb <- function(x, ...) {
    mapply('cbind', x, ..., SIMPLIFY=FALSE)
  }

  parts <- foreach (s = seq(1:samplesize), .packages = c('bksgpR', 'foreach', 'doParallel'), .combine = 'comb', .multicombine = TRUE) %dopar%
    #  parts <- foreach (s = seq(1:samplesize), .packages = c('bksgpR', 'foreach'), .combine = 'comb', .multicombine = TRUE)
  {
    pre_sample_g=pre_samples[,s]

    ret_s=bks_global_inner(hyper, spec$L_i, size_k, pre_sample_g)

    inner=head(ret_s,n=-size_k)
    sample=tail(ret_s,size_k)
    list(inner, sample)
  }

  inner = parts[[1]]
  samples <<- parts[[2]]

  inner_expectation=apply(inner,1,mean)
  kl_g=bks_global_kl_g(hyper,spec$hyper_p,size_k)
  L_global=inner_expectation-kl_g

  #if (is.nan(L_global[1]) || is.na(L_global[1]) || is.infinite(L_global[1])) {
  #  cat('nan\n')
  #}
  # ret=data.frame(L_global=L_global,inner=inner, kl_g=kl_g,hyper=c(L_global[1],hyper))
  ret=data.frame(L_global=L_global)
  if(!is.infinite(L_global[1]) && !is.na(L_global[1]))
  {
    #save(samples, parts1, parts2, file='log.rdat')
    write.csv(file=paste(outfile,'_hyper',sep=''),x=hyper)
  }
  cat('compute_L_global:',L_global[1],'\n')
  write.csv(file=outfile,x=ret)
  #save(file='samples.rdat',parts1,parts2,samples)
  return(L_global)
}

objective_global=function(hyper, spec, samplesize=100,seed=1,outfile='random.csv')
{
  cat('obj:',hyper[1:5],'\n')

  if(file.exists(outfile))
  {
    ret_data=read.csv(file=outfile)
    ret=ret_data$L_global[1]
    intfile=paste(outfile,'_',round(ret_data$L_global[1]),sep='')
    file.remove(outfile)
    # file.rename(outfile,intfile)
  }else{
    #save(file=paste(outfile,'_env',sep=''),hyper,spec,dat,blocksize,samplesize,seed,outfile)
    L_global=compute_L_global(hyper,spec,samplesize,seed,outfile)
    ret=L_global[1]
    iterfile=paste(outfile,'_iter',sep='')
    write(file=iterfile,paste(outfile,'_',round(ret),sep=''),append=TRUE)
  }

  return(ret)
}

gradient_global=function(hyper,spec,samplesize=100,seed=1,outfile='random.csv')
{
  cat('grad:',hyper[1:5],'\n')
  if(file.exists(outfile))
  {
    ret_data=read.csv(file=outfile)
    ret=ret_data$L_global[-1]
    intfile=paste(outfile,'_',round(ret_data$L_global[1]),sep='')
    # file.rename(outfile,intfile)
    file.remove(outfile)
    
  }else{
    L_global=compute_L_global(hyper,spec,samplesize,seed,outfile)
    ret=lpgrad[-1]
  }

  return(ret)
}


optimize_with_restart_global=function(hyper, spec,samplesize=100,seed=1,outfile='random.csv',max_restart=10,max_iter=50)
{
  restart_num=1
  remain_iter=max_iter
  rst=list()
  ret={}
  while(restart_num<=max_restart & remain_iter>0){

    rst=tryCatch(
      out.optim <- optim(hyper, objective_global, gradient_global, method = 'L-BFGS-B',
                         control=list(trace=1,fnscale=-1,maxit=remain_iter,ndeps=1e-3),
                         spec=spec, samplesize, seed, outfile),
      error=function(e)
      {
        print(e)
        cat('restarting ', restart_num, ' time\n')
        if(file.exists(outfile)){
          file.rename(outfile,paste(outfile,'_error',sep=''))
        }
        return(NULL)
      }
    )
    if(is.null(rst))
    {
      hyper=get_hyper_from_file(paste(outfile,'_hyper',sep=''))
      restart_num=restart_num+1
      rst=list()
    } else {
      cat('converged before max iterations\n')
      # print(rst)
      hyper=rst$par
    }
    aa=read.table(paste(outfile,'_iter',sep=''))
    remain_iter=max(0,max_iter-length(aa$V1))
    cat('remain iteration:', remain_iter,'\n')
  }
  ret$out=rst
  ret$samples=samples
  return(ret)
}
