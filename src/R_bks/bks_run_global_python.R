# setwd('/Users/tong/Dropbox/CODE/vaksGP/src/bks/')
source('bks_load_all_global.R')

env_file = 'env_maplecgpu.R'
source(paste('./env/',env_file,sep=''))
rnum=50
maxit=100
# size_sample=1000
# prior_pg=1


# read settings and filenames
filename <- read.delim("../filename.txt", header=FALSE,stringsAsFactors=FALSE)
kern_info_file <- paste('../',filename$V1[1],sep='')
prob_result_file <- filename$V1[2]
working_folder = paste('../', filename$V1[3],sep='')
size_sample <- as.numeric(filename$V1[4])
prior_pg <- as.numeric(filename$V1[5])

# read from kern_info_file
spec_temp = read.csv(kern_info_file, stringsAsFactors=FALSE)
spec = list()
spec$kernels = spec_temp$kernels
spec$L_i = spec_temp$L_i
spec$size_k = length(spec$kernels)


# Parallelize
# registerDoParallel()
registerDoParallel(cores = 12)
size_k = spec$size_k
print(size_k)
m_qg=rep(0, size_k)
m_pg=rep(0, size_k)
print(spec)
k_qg=diag(1,size_k,size_k)
L_qg=t(chol(k_qg))
L_qg[row(L_qg) == col(L_qg)] = log(L_qg[row(L_qg) == col(L_qg)]) #To enforce positive diagonals
L_qg=L_qg[lower.tri(L_qg,diag=TRUE)]

k_pg=diag(prior_pg,size_k,size_k)
L_pg=t(chol(k_pg))
L_pg[row(L_pg) == col(L_pg)] = log(L_pg[row(L_pg) == col(L_pg)]) #To enforce positive diagonals
L_pg=L_pg[lower.tri(L_pg,diag=TRUE)]

hyper=c(m_qg,L_qg)
spec$hyper_p=c(m_pg,L_pg)

# file_str=make_random_name(key=paste('L_global_iter_','1','_',sep=''),len = 8)
log_folder = paste(working_folder ,'logs',sep='/')
if(!dir.exists(log_folder))
  {
  dir.create(log_folder, recursive = TRUE)
  }

file_str=paste(log_folder, prob_result_file, sep='/')
cat('file string: ', file_str,'\n')

# browser()
# spec: "kernels" "L_i"    "size_k"  "hyper_p"
#--------
start_time = Sys.time()
#optim with restart
res = optimize_with_restart_global(hyper=hyper, spec=spec,samplesize=size_sample,
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
cfg=c(size_sample,
      rnum,
      maxit,
      out$value,
      out$par,
      sm,
      spec$kernels,
      spec$L_i,
      spec$size_k,
      spec$hyper_p,
      diff)
df_cfg=data.frame(cbind(cfg))
ind=seq(1,length(hyper))
rownames(df_cfg)=c('sample_size',
                   'seed',
                   'max_iter',
                   'global_VLB',
                   paste('hyp_q',seq(1,length(out$par)),sep='_'),
                   paste('soft_max',seq(1,length(sm)),sep='_'),
                   paste('kernel',seq(1,length(spec$kernels)),sep='_'),
                   paste('local_VLB',seq(1,length(spec$L_i)),sep='_'),
                   'num_kernels',
                   paste('hyp_p',seq(1,length(spec$hyper_p)),sep='_'),
                   'time')
write.csv(file=paste(file_str,'_cfg',sep=''),x=df_cfg)

# Store ordering of kernels in descending order
df2=data.frame(spec$kernels, sm, spec$L_i, fix.empty.names = FALSE)
df2=df2[order(-df2[,2]),]

write.table(df2,paste(working_folder, 'ana_res/', prob_result_file, sep=''),sep=",",row.names=FALSE, col.names =FALSE)
#-------
stopImplicitCluster()