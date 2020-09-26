source('bks_load_all_global.R')

option_list = list(
  make_option(c("-e", "--envfile"), type="character", default='env_maplecgpu.R',
              help="Environment File", metavar='env_file'),

  make_option(c("-s", "--sizes"), type="integer", default=5000,
              help="size_s (sample size)", metavar="size_s"),

  make_option(c("-r", "--rnum"), type="integer", default=55,
              help="random seed", metavar="rnum"),

  make_option(c("-m", "--maxit"), type="integer", default=100,
              help="max number of iterations", metavar="maxit"),

  make_option(c("-w", "--wkdir"), type="character", default='./',
              help="directory to temporarily store experiment files", metavar="wkdir"),

  make_option(c("-f", "--resdir"), type="character", default="PxLxR1000v6_sm4_su16_smb32_aaai/nmb8/",
              help="results folder name", metavar="resdir"),

  make_option(c("-i", "--iter"), type="integer", default=1,
              help="iteration", metavar="iteration"),

  make_option(c("-k", "--kernels"), type="character", default=NULL,
              help="Kernels to calculate global VLB on", metavar="kernels"),

  make_option(c("-t", "--settings"), type="character", default=NULL,
              help="settings", metavar="settings"),
  make_option(c("-n", "--nkerns"), type="integer", default=0,
              help="nkerns", metavar="nkerns"),

  make_option(c("-p", "--priorK"), type="integer", default=10,
              help="variance of p(g)", metavar="priorK"),

  make_option(c("-o", "--epoch"), type="integer", default=1,
              help=" epoch", metavar="epoch"),

  make_option(c("-a", "--file_str"), type="character", default='all',
              help="file_str", metavar="file_str")
);

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

env_file = opt$envfile
results_folder = opt$resdir
wkdir = opt$wkdir
id = opt$iter
priorK = opt$priorK / 20
nkerns = opt$nkerns
epoch = opt$epoch
pk_file_str = opt$file_str
if (is.null(opt$settings)) {
  size_s = opt$sizes
  rnum = opt$rnum
  maxit = opt$maxit
} else {
  print("settings found")
  set = parse_settings(opt$settings)
  size_s = set$sizesglobal
  rnum = set$rnum
  maxit = set$maxitglobal
}

print(paste("size_s =", size_s))
print(paste("rnum =", rnum))
print(paste("maxit =", maxit))

# Exclude the best kernels of previous iterations: this is to prevent the possibility of expanding the same kernels again and again
exclude_prev_best = FALSE

if (is.null(opt$resdir)) {
  stop("Results folder argument missing\n")
}

kerns = NULL
if (!is.null(opt$kernels)) {
  kerns = strtoi(strsplit(opt$kernels,split = ' ')[[1]])
}


source(paste('./env/',env_file,sep=''))

directory = wkdir

working_folder= results_folder
working_folder=paste(directory,working_folder, sep='/')

if(!dir.exists(working_folder))
{
  dir.create(working_folder, recursive = TRUE)
}
# only consider the first n kernels, if nkern=0, consider all kernels
if(nkerns > 0){
   elbos = read.csv(paste(working_folder, '/elbos_ordered.csv', sep = ''),sep="\t", header =TRUE)
   # X1, X2/kernels, L_i
   kerns = elbos$kernels[1:nkerns]
   # kerns = elbos$X1[1:nkerns]

   print(kerns)
}
browser()
spec=extract_local_info(results_folder, kerns=kerns, exclude_prev_best=exclude_prev_best)
# spec = extract_local_info_by_batch(results_folder, epoch, kerns=kerns, exclude_prev_best=exclude_prev_best)
# spec = extract_local_info_by_epoch(results_folder, epoch, kerns=kerns, exclude_prev_best=exclude_prev_best)

portion = strtoi(strsplit(results_folder, split='/nmb')[[1]][2])
spec$L_i = spec$L_i/(portion/4)
## Parallelise
# registerDoParallel()
registerDoParallel(cores = 20)
size_k = spec$size_k
m_qg=rep(0, size_k)
m_pg=rep(0, size_k)
print(spec)
k_qg=diag(1,size_k,size_k)
L_qg=t(chol(k_qg))
L_qg[row(L_qg) == col(L_qg)] = log(L_qg[row(L_qg) == col(L_qg)]) #To enforce positive diagonals
L_qg=L_qg[lower.tri(L_qg,diag=TRUE)]

k_pg=diag(priorK,size_k,size_k)
L_pg=t(chol(k_pg))
L_pg[row(L_pg) == col(L_pg)] = log(L_pg[row(L_pg) == col(L_pg)]) #To enforce positive diagonals
L_pg=L_pg[lower.tri(L_pg,diag=TRUE)]

hyper=c(m_qg,L_qg)
spec$hyper_p=c(m_pg,L_pg)
if(nkerns > 0){
  pk_file_str = paste( "top",nkerns,sep='')
}
# file_str=make_random_name(key=paste('L_global_iter_',id,'_',sep=''),len = 8)
file_str=paste(pk_file_str,"_norm","_max_s",size_s,"_pg",priorK,".csv")
log_folder = paste(working_folder ,'logs',sep='/')
if(!dir.exists(log_folder))
  {
  dir.create(log_folder, recursive = TRUE)
  }

file_str=paste(log_folder, file_str, sep='/')
cat('file string: ', file_str,'\n')

# browser()

#--------
start_time = Sys.time()
#optim with restart
res = optimize_with_restart_global(hyper=hyper, spec=spec,samplesize=size_s,
                           seed=rnum, outfile = file_str,
                           max_restart=1000, max_iter = maxit)
end_time = Sys.time()
diff = difftime(end_time, start_time, units='min')

out = res$out
samples=res$samples
# Something is wrong with out!
# Instead of determining best kernels by q(g) means, use samples to find mean of p(k|g)
sm=soft_max_from_samples(samples)
cat(diff, '\n')
cfg=c(results_folder,
      size_s,
      rnum,
      maxit,
      out$value,
      out$par,
      sm,
      spec$kernels,
      spec$L_i,
      spec$times,
      spec$size_k,
      spec$hyper_p,
      id,
      diff)
df_cfg=data.frame(cbind(cfg))
ind=seq(1,length(hyper))
rownames(df_cfg)=c('results_folder',
                   'sample_size',
                   'seed',
                   'max_iter',
                   'global_VLB',
                   paste('hyp_q',seq(1,length(out$par)),sep='_'),
                   paste('soft_max',seq(1,length(sm)),sep='_'),
                   paste('kernel',seq(1,length(spec$kernels)),sep='_'),
                   paste('local_VLB',seq(1,length(spec$L_i)),sep='_'),
                   paste('local_time_taken',seq(1,length(spec$times)),sep='_'),
                   'num_kernels',
                   paste('hyp_p',seq(1,length(spec$hyper_p)),sep='_'),
                   'iteration',
                   'time')
write.csv(file=paste(file_str,'_cfg',sep=''),x=df_cfg)

# Store ordering of kernels in descending order
df2=data.frame(spec$kernels, sm, fix.empty.names = FALSE)
df2=df2[order(-df2[,2]),]



write.table(df2,paste(working_folder, pk_file_str,"_norm","_max_s",size_s,"_pg",priorK,".csv", sep=''),sep="\t",row.names=FALSE, col.names =FALSE)
# write.table(df2,paste(working_folder, "/iter_", id, "_top",nkerns,"_norm",epoch,"_max_results_s",size_s,"_pg",priorK,"recheck.csv", sep=''),sep="\t",row.names=FALSE, col.names =FALSE)

#-------
objs = rbind(spec$kernels, spec$L_i)
df2=data.frame(t(objs), fix.empty.names = FALSE)
# write.table(df2,paste(working_folder, '/elbos.csv', sep = ''),sep="\t",row.names=FALSE, col.names =FALSE)
ordered=df2[order(spec$L_i,decreasing=TRUE),]
# write.table(ordered,paste(working_folder, '/elbos_ordered.csv', sep = ''),sep="\t",row.names=FALSE, col.names =TRUE)


