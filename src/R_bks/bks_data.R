#data_name='vol'
#data_name='per5k'
#data_name='PxL1k'
#data_name='per150'
#data_name='rbf200'
#data_name='airline144'
#data_name='solar402'
#data_name='COGT7668'
#data_name='NO2GT7709'
#data_name='NOxGT7712'
#data_name='mauna545'
#data_name='solar402'
#data_name='occ'
#data_name='spd'

load_data = function(dataset) {
  #y=t(matrix(toy$dat$Data1))
  load(paste(home,'/datasets/',dataset,'.rdat',sep=''))

  #y=t(matrix(toy$dat$Data2))
  y=t(normalize_col(matrix(toy$dat$Data1)))
  ##dataset
  x=t(matrix(toy$dat$xi))
  #y=t(matrix(toy$dat$Data2))
  #y=t(matrix(toy$dat$Data1))
  x=t(matrix(toy$dat$xi-mean(toy$dat$xi)))
  
  dat=list(x=x,y=y)

  #vol
  #sup_st=0.1
  #sup_en=0.15

  init=read.csv('bks_init.csv')
  ii=which(init$dataset==dataset)
  if(length(ii)==0)
  {
    #PxL1k
    sup_st=0.15
    sup_en=0.25
    init_kern_theta=NULL
    cat('no initial values for ',dataset,'\n')
  }else{
    cat('found setting for initial values for ',dataset,'\n')
    sup_st=init$sup_st[ii]
    sup_en=init$sup_en[ii]
    init_kern_theta=c(init$se_len[ii],init$se_sig[ii], #SE
                      init$rq_a[ii], init$rq_len[ii],init$rq_sig[ii], #RQ
                      init$lin_len[ii], #LIN
                      init$per_len[ii],init$per_p[ii],init$per_sig[ii])#PER
  }


  start_save_file(dataset)
  plot(x,y,xlab='Input',ylab='Output',type='l')
  end_save_file()

  ret = list('x'= x, 'y' = y, 'sup_st' = sup_st, 'sup_en' = sup_en, 'init_kern_theta' = init_kern_theta)
}


get_data_to_use <- function(dat,num_mb,size_mb,size_u,support_mode, dataseed){
  x = dat$x
  y = dat$y
  test_data =NULL
  # if num_mb = 0, no separate testing data
  # if num_mb = inf, there are separate data, but use all other data for  training
  if (num_mb>0){
    datasize_total=dim(x)[2]
    datasize_to_use=num_mb*size_mb


    # Check whether exceed boundary
    # make sure no overlapping (train and test)
    test_percentage = 0.2 
    max_test = 10000
    test_size = min(max_test, round(datasize_total*test_percentage))


    if (datasize_to_use > (datasize_total-test_size)){
      cat('Number of minibatch too big, using all the data.')
      datasize_to_use = datasize_total-test_size
    }
      

    if(support_mode==4){
      # TODO: this is only for smode=4
      # shuffle but not shuffle the inducing points
      # inducing points are in the most beginning
      ind = (size_u+1):datasize_total
      set.seed(dataseed)
      shuffled = sample(ind)
      # shuffled = ind  # test unshuffled data
      inducing_ind = 1:size_u
      data_to_use_ind = c(inducing_ind, shuffled[1:(datasize_to_use-size_u)])
      #  training data
      x = dat$x[, data_to_use_ind]
      x = matrix(x, dim(dat$x)[1],datasize_to_use)
      y = dat$y[, data_to_use_ind]
      y = matrix(y, 1, datasize_to_use)
      # test data
      
      test_data = NULL
      test_ind = shuffled[(datasize_total-size_u-test_size+1):(datasize_total-size_u)]
      test_data$x = t(matrix(dat$x[, test_ind]))
      test_data$y = matrix(dat$y[, test_ind])

      dat$x=x
      dat$y=y
    }else{
      set.seed(dataseed)
      shuffled = sample(datasize_total)
      # shuffled = seq(1, datasize_total, 1)  # test unshuffled data
      data_to_use_ind = shuffled[1:(datasize_to_use)]
      #  training data
      x = dat$x[, data_to_use_ind]
      y = dat$y[, data_to_use_ind]
      # ordering (cannot use order(data_to_use_ind), dat$x might not in order)
      ordered_ind = order(x)
      x = x[ordered_ind]
      x = t(matrix(x))
      y = y[ordered_ind]
      y = t(matrix(y))
      # cat(dim(x))
      # test data
      
      test_data = NULL
      test_ind = shuffled[(datasize_total-test_size+1):(datasize_total)]
      test_data$x = t(matrix(dat$x[, test_ind]))
      test_data$y = matrix(dat$y[, test_ind])

      # print(test_data$x[, 1:20])
      dat$x=x
      dat$y=y
    }

  }
  results=list("x"=x, "y"=y, "dat"=dat, "test_data"=test_data)
  return(results)
}

get_data_to_use_swiss <- function(dat,num_mb,size_mb,size_u,support_mode, dataseed){
  # choose consecutive subset
  x = dat$x
  y = dat$y
  # for swissgrid only
  ind_sparse = seq(1,dim(x)[2],4)
  x=t(matrix(x[, ind_sparse]))
  y=t(matrix(y[, ind_sparse]))
  dat$x=x
  dat$y=y
  # the final training size: original_size/4 - test_size not (original_size-test_size)/4
  
  test_data =NULL
  # if num_mb = 0, no separate testing data
  # if num_mb = inf, there are separate data, but use all other data for  training
  if (num_mb>0){
    datasize_total=dim(x)[2]
    datasize_to_use=num_mb*size_mb


    # Check whether exceed boundary
    # make sure no overlapping (train and test)
    test_percentage = 0.2 
    max_test = 10000
    test_size = min(max_test, round(datasize_total*test_percentage))


    if (datasize_to_use > (datasize_total-test_size)){
      cat('Number of minibatch too big, using all the data.')
      datasize_to_use = datasize_total-test_size
    }
      

    if(support_mode==4){
      # TODO: this is only for smode=4
      # shuffle but not shuffle the inducing points
      # inducing points are in the most beginning
      ind = (size_u+1):datasize_total
      set.seed(dataseed)
      shuffled = sample(ind)
      inducing_ind = 1:size_u
      #  training data
      
      
      train_candi_ind = shuffled[1:(datasize_total-size_u-test_size)]
      train_data_candi_x = dat$x[, train_candi_ind]
      train_data_candi_y = dat$y[, train_candi_ind]
      ordered_ind = order(train_data_candi_x)
      train_data_candi_x = train_data_candi_x[ordered_ind]
      train_data_candi_y = train_data_candi_y[ordered_ind]
      # data_to_use_ind = c(inducing_ind, shuffled[1:(datasize_to_use-size_u)])
      x = c(dat$x[, inducing_ind], train_data_candi_x[1:(datasize_to_use-size_u)])
      x = matrix(x, dim(dat$x)[1],datasize_to_use)
      y = c(dat$y[, inducing_ind], train_data_candi_y[1:(datasize_to_use-size_u)])
      y = matrix(y, 1, datasize_to_use)


      # test data
      # TODO: selecting test data across the whole dataset while choosing training data consecutively is not good
      test_data = NULL
      test_ind = shuffled[(datasize_total-size_u-test_size+1):(datasize_total-size_u)]
      test_data$x = t(matrix(dat$x[, test_ind]))
      test_data$y = matrix(dat$y[, test_ind])

      dat$x=x
      dat$y=y
    }else{
      set.seed(dataseed)
      shuffled = sample(datasize_total)
      # shuffled = seq(1, datasize_total, 1)  # test unshuffled data
      # data_to_use_ind = shuffled[1:(datasize_to_use)]
      #  training data
      train_candi_ind = shuffled[1:(datasize_total-test_size)]
      train_data_candi_x = dat$x[, train_candi_ind]
      train_data_candi_y = dat$y[, train_candi_ind]
      ordered_ind = order(train_data_candi_x)
      train_data_candi_x = train_data_candi_x[ordered_ind]
      train_data_candi_y = train_data_candi_y[ordered_ind]
      x = train_data_candi_x[1:datasize_to_use]
      y = train_data_candi_y[1:datasize_to_use]
      # ordering (cannot use order(data_to_use_ind), dat$x might not in order)
      x = t(matrix(x))
      y = t(matrix(y))
      # cat(dim(x))
      # test data
      # TODO: selecting test data across the whole dataset while choosing training data consecutively is not good
      test_data = NULL
      test_ind = shuffled[(datasize_total-test_size+1):(datasize_total)]
      test_data$x = t(matrix(dat$x[, test_ind]))
      test_data$y = matrix(dat$y[, test_ind])

      # print(test_data$x[, 1:20])
      dat$x=x
      dat$y=y
    }

  }
  results=list("x"=x, "y"=y, "dat"=dat, "test_data"=test_data)
  return(results)
}