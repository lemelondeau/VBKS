//compute covariance of composed kernels
#include <Rcpp.h>
#include <stan/math.hpp>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <fstream>
#define OUTPUT_LOG 0


//using namespace Eigen;
using Eigen::Matrix;
using Eigen::Dynamic;

//return a functor: cov_comp(init,op,seq)
struct cov_comp {
  const int init_;
  const Matrix<int, Dynamic, 1> op_;
  const Matrix<int, Dynamic, 1> seq_;

  //Kernels
  static const int SE = 1;   //Squared Exponential
  static const int RQ = 2;   //Rational Quadratic
  static const int LIN = 3;  //Linear
  static const int PER = 4;  //Periodic

  //Operators
  static const int SUM = 1;
  static const int PROD = 2;

  cov_comp(const int& init, const Matrix<int, Dynamic, 1> &op, const Matrix<int, Dynamic, 1> &seq) : init_(init),op_(op),seq_(seq) { }

  template <typename T>
  T operator()(const Matrix<T, Dynamic, 1>& theta,const Matrix<double, Dynamic, 1>& xi,const Matrix<double, Dynamic, 1>& xj) const {
    T k = 0;    //Final kernel eval
    int i = 0;  //theta index

    //Compute root kernel value
    if (init_ == SE) {
      T ell = theta[i++];
      T sigma = theta[i++];
      k += squared_exponential(xi, xj, ell, sigma);
    } else if (init_ == RQ) {
      T alpha = theta[i++];
      T ell = theta[i++];
      T sigma = theta[i++];
      k += rational_quadratic(xi, xj, alpha, ell, sigma);
    } else if (init_ == LIN) {
      T ell = theta[i++];
      k += linear(xi, xj, ell);
    } else if (init_ == PER) {
      T ell = theta[i++];
      T p = theta[i++];
      T sigma = theta[i++];
      k += periodic(xi, xj, ell, p, sigma);
    }

    //Compute rest of kernel values in seq_
    for (int j = 0; j < seq_.rows(); j++) {
      T k2 = 0;
      int seq = seq_(j);

      if (seq == SE) {
        T ell = theta[i++];
        T sigma = theta[i++];
        k2 += squared_exponential(xi, xj, ell, sigma);
      } else if (seq == RQ) {
        T alpha = theta[i++];
        T ell = theta[i++];
        T sigma = theta[i++];
        k2 += rational_quadratic(xi, xj, alpha, ell, sigma);
      } else if (seq == LIN) {
        T ell = theta[i++];
        k2 += linear(xi, xj, ell);
      } else if (seq == PER) {
        T ell = theta[i++];
        T p = theta[i++];
        T sigma = theta[i++];
        k2 += periodic(xi, xj, ell, p, sigma);
      } else {
        break;
      }

      if (op_(j) == SUM) {
        k += k2;
      } else if (op_(j) == PROD) {
        k *= k2;
      } else {
        break;
      }
    }

    return k;
  }

  template <typename T>
  T squared_exponential(const Matrix<double, Dynamic, 1>& xi,const Matrix<double, Dynamic, 1>& xj, T ell, T sigma) const {
    return exp(2*sigma)*exp(-0.5*stan::math::squared_distance(xi,xj)/exp(2*ell));
  }

  //RQ kernel from http://www.cs.toronto.edu/~duvenaud/cookbook/
  template <typename T>
  T rational_quadratic(const Matrix<double, Dynamic, 1>& xi,const Matrix<double, Dynamic, 1>& xj, T alpha, T ell, T sigma) const {
    //using std::pow;
    return exp(2*sigma)*pow((1+0.5*(stan::math::squared_distance(xi,xj)/(exp(alpha)*exp(2*ell)))),(-exp(alpha)));
  }

  //Linear kernel from covfunc/cov/covLINiso.m
  template <typename T>
  T linear(const Matrix<double, Dynamic, 1>& xi,const Matrix<double, Dynamic, 1>& xj, T ell) const {
    int dnum = xi.rows();
    Matrix<T, Dynamic, Dynamic> P(dnum, dnum);
    for (int i = 0; i < dnum; i++) {
      P(i,i) = exp(2*ell);
    }

    return xi.transpose()*P.inverse()*xj;
  }

  //Periodic kernel from http://www.cs.toronto.edu/~duvenaud/cookbook/
  template <typename T>
  T periodic(const Matrix<double, Dynamic, 1>& xi,const Matrix<double, Dynamic, 1>& xj, T ell, T p, T sigma) const {
    //using std::pow;
    return exp(2*sigma)*exp(-2*pow(stan::math::sin(stan::math::pi() * stan::math::distance(xi,xj)/exp(p)),2)/exp(2*ell));
  }
};

using namespace Rcpp;

// [[Rcpp::export]]
NumericVector test_per_kernel(NumericVector hyp_,NumericVector xi_,NumericVector xj_)
{
  NumericVector ret_(hyp_.size() + 1);
  using Eigen::Matrix;
  using Eigen::Dynamic;
  Matrix<int, Dynamic, 1> op(1);
  Matrix<int, Dynamic, 1> seq(1);
  Matrix<double, Dynamic, 1> xi(1);
  Matrix<double, Dynamic, 1> xj(1);
  Matrix<stan::math::var, Dynamic, 1> theta(hyp_.size());
  stan::math::var obj=0;
  for(int i=0; i<xi_.size();i++)
  {
    xi[i]=xi_[i];
    xj[i]=xj_[i];
  }
  for(int i=0; i<hyp_.size(); i++)
  {
    theta[i]=hyp_[i];
  }

  int init=4;
  op[0]=-1;
  seq[0]=-1;
  cov_comp cov(init,op,seq);

  obj=cov(theta,xi,xj);

  ret_[0]=obj.val();
  obj.grad();
  for(int i=0; i<hyp_.size(); i++)
  {
    ret_[i+1]=theta[i].adj();
  }
  std::cout<<"hello world"<<std::endl;

  // Memory is allocated on a global stack
  stan::math::recover_memory();
  stan::math::ChainableStack::memalloc_.free_all();

  return ret_;
}

// [[Rcpp::export]]
NumericVector bks_logml_close(NumericVector vec_vi_par, NumericVector vec_vi_par0,
                              NumericVector tnum, NumericVector unum) {
  NumericVector ret_(vec_vi_par.size() + 1);
  using Eigen::Matrix;
  using Eigen::Dynamic;
  int size_t=tnum[0];
  int size_u=unum[0];
  stan::math::var logml=0;
  Matrix<stan::math::var, Dynamic, 1> t_m(size_t);
  Matrix<stan::math::var, Dynamic, 1> t_k(size_t);
  Matrix<stan::math::var, Dynamic, 1> u_m(size_u);
  Matrix<stan::math::var, Dynamic, 1> L_diag(size_u);
  Matrix<stan::math::var, Dynamic, Dynamic> L(size_u,size_u);
  Matrix<double, Dynamic, 1> t_m0(size_t);
  Matrix<double, Dynamic, 1> t_k0(size_t);
  Matrix<double, Dynamic, 1> u_m0(size_u);
  //Matrix<double, Dynamic, Dynamic> L0(size_u,size_u);
  //initialize structure

  for(int i=0; i<size_t; i++)
  {
    t_m[i]=vec_vi_par[i];
    t_k[i]=vec_vi_par[size_t+i];//the orignal variance are in logscale
    t_m0[i]=vec_vi_par0[i];
    t_k0[i]=vec_vi_par0[size_t+i];
  }

  //for(int i=0; i<size_u; i++)
  //{
  //  u_m[i]=vec_vi_par[size_t*2+i];
  //  u_m0[i]=vec_vi_par0[size_t*2+i];
  //}

  L.fill(0);
  //L0.fill(0);
  int ind=size_t*2+size_u;
  for(int i=0; i<size_u; i++)
  {
    for(int j=i; j<size_u; j++)
    {
      if (i==j) {
        L_diag[i] = vec_vi_par[ind];
        L(i,j) = stan::math::exp(L_diag[i]);
      } else {
        L(i,j)=vec_vi_par[ind];
      }
      //L0(i,j)=vec_vi_par0[ind];
      ind++;
    }
  }

  //KLD
  logml+=-0.5*sum(stan::math::elt_divide(stan::math::exp(2*t_k),stan::math::exp(2*t_k0)));
  logml+=-0.5*stan::math::dot_product(stan::math::elt_divide(stan::math::subtract(t_m0,t_m),stan::math::exp(2*t_k0)),stan::math::subtract(t_m0,t_m));
  //logml+=-0.5*log(stan::math::prod(stan::math::exp(t_k0))/stan::math::prod(stan::math::exp(t_k)));
  logml+=-0.5*(2*stan::math::sum(t_k0)-2*stan::math::sum(t_k));
  logml+=size_t;
  //Entropy
  //logml+=-0.5*log(2*stan::math::pi())-0.5-0.5*stan::math::log(stan::math::pow(stan::math::prod(stan::math::diagonal(L)),2));
  logml+=0.5*size_u*(1+2*stan::math::pi());
  //logml+=0.5*stan::math::log(stan::math::pow(stan::math::prod(stan::math::diagonal(L)),2));
  logml+=0.5*stan::math::sum(stan::math::log(stan::math::elt_multiply(stan::math::diagonal(L),stan::math::diagonal(L))));
  //std::cout<< "logml:"<<logml<<std::endl;

  ret_[0]=logml.val();
  logml.grad();
  ind=1;
  for(int i=0; i<size_t; i++)
  {
    ret_[ind++]=t_m[i].adj();
  }
  for(int i=0; i<size_t; i++)
  {
    ret_[ind++]=t_k[i].adj();
  }
  for(int i=0; i<size_u; i++)
  {
    ret_[ind++]=0;
  }
  for(int i=0; i<size_u; i++)
  {
    for(int j=i; j<size_u; j++)
    {
      if (i==j) {
        ret_[ind++]=L_diag[i].adj();
      } else {
        ret_[ind++]=L(i,j).adj();
      }
    }
  }

  // Memory is allocated on a global stack
  stan::math::recover_memory();
  stan::math::ChainableStack::memalloc_.free_all();

  return ret_;
}

// [[Rcpp::export]]
NumericVector bks_logml_s(NumericVector vec_vi_par, NumericMatrix mat_xu,
                          NumericVector init_, NumericVector seq_, NumericVector op_,
                          NumericVector vec_sample_t,  NumericVector vec_sample_u) {
  NumericVector ret_(vec_vi_par.size() + 1);




  using Eigen::Matrix;
  using Eigen::Dynamic;
  //size of hyperparameter
  int size_t=vec_sample_t.size();
  //size of support
  int size_u=vec_sample_u.size();
  //dimension
  int dim_x=mat_xu.nrow();
  //std::cout <<"size of support:" << size_u <<std::endl
  //          << "size of hyper:" << size_t<<std::endl
  //          << "size of block:" << size_b<<std::endl
  //          << "dim of input:" << dim_x <<std::endl;
  //
  stan::math::var logml=0;
  Matrix<stan::math::var, Dynamic, 1> t_s(size_t);
  stan::math::var t_noise;
  stan::math::var t_mean;
  Matrix<stan::math::var, Dynamic, 1> t_theta(size_t-2);

  Matrix<stan::math::var, Dynamic, 1> u_s(size_u);

  Matrix<stan::math::var, Dynamic, 1> t_m(size_t);
  Matrix<stan::math::var, Dynamic, 1> t_k(size_t);
  Matrix<stan::math::var, Dynamic, 1> u_m(size_u);
  Matrix<stan::math::var, Dynamic, 1> L_diag(size_u);
  Matrix<stan::math::var, Dynamic, Dynamic> L(size_u,size_u);
  Matrix<stan::math::var, Dynamic, Dynamic> sig_uu(size_u,size_u);

  Matrix<double, Dynamic, 1> sample_t(size_t);
  Matrix<double, Dynamic, 1> sample_u(size_u);
  Matrix<double, Dynamic, Dynamic> xu(dim_x,size_u);
  Matrix<int, Dynamic, 1> op(op_.size());
  Matrix<int, Dynamic, 1> seq(seq_.size());
  int init;
  //initialization
  init=init_[0];
  for(int i=0; i<op_.size(); i++)
  {
    op[i]=op_[i];
    seq[i]=seq_[i];
  }

  for(int i=0; i<size_t; i++)
  {
    t_m[i]=vec_vi_par[i];
    t_k[i]=vec_vi_par[size_t+i];
    sample_t[i]=vec_sample_t[i];
  }

  for(int i=0; i<size_u; i++)
  {
    u_m[i]=vec_vi_par[size_t*2+i];
    sample_u[i]=vec_sample_u[i];
  }

  L.fill(0);
  int ind=size_t*2+size_u;
  for(int i=0; i<size_u; i++)
  {
    for(int j=i; j<size_u; j++)
    {
      if (i==j) {
        L_diag[i] = vec_vi_par[ind++];
        L(i,j) = stan::math::exp(L_diag[i]);
      } else {
        L(i,j)=vec_vi_par[ind++];
      }
    }
  }

  for(int i=0; i<dim_x; i++)
  {
    for(int j=0; j<size_u; j++)
    {
      xu(i,j)=mat_xu(i,j);
    }
  }

  t_s=t_m+stan::math::elt_multiply(stan::math::exp(t_k),sample_t);
  u_s=u_m+stan::math::multiply(L,sample_u);

  cov_comp cov(init,op,seq);

  t_mean=t_s[0];
  t_noise=t_s[size_t-1];
  for(int i=1; i<size_t-1; i++) {
    t_theta[i-1]=t_s[i];
  }

  //compute sig_uu
  for(int i=0; i<size_u; i++){
    for(int j=i; j<size_u; j++){
      Matrix<double, Dynamic, 1> xi(dim_x);
      Matrix<double, Dynamic, 1> xj(dim_x);
      for(int k=0; k<dim_x; k++){
        xi(k)=xu(k,i);
        xj(k)=xu(k,j);
      }
      sig_uu(i,j)=cov(t_theta,xi,xj);
      if(i!=j)
      {
        sig_uu(j,i)=sig_uu(i,j);
      } else {
        sig_uu(i,j)=sig_uu(i,j)+0.0001;
      }
    }
  }

  Matrix<stan::math::var, Dynamic, Dynamic> Lu(size_u,size_u);
  Matrix<stan::math::var, Dynamic, 1> alphau(size_u);

  Lu=stan::math::cholesky_decompose(sig_uu);
  alphau=stan::math::mdivide_left_tri_low(Lu,u_s);

  logml+=-0.5*stan::math::dot_self(alphau);
  //std::cout<<"1:"<<logml<<std::endl;
  logml+=-0.5*stan::math::log_determinant(sig_uu);
  //std::cout<<"2:"<<-0.5*stan::math::log_determinant(sig_uu)<<std::endl;
  logml+=-0.5*size_u*log(2*stan::math::pi());
  //std::cout<<"3:"<<-0.5*size_u*log(2*stan::math::pi())<<std::endl;

  ret_[0]=logml.val();
  logml.grad();
  ind=1;
  for(int i=0; i<size_t; i++)
  {
    ret_[ind++]=t_m[i].adj();
  }
  for(int i=0; i<size_t; i++)
  {
    ret_[ind++]=t_k[i].adj();
  }
  for(int i=0; i<size_u; i++)
  {
    ret_[ind++]=u_m[i].adj();
  }
  for(int i=0; i<size_u; i++)
  {
    for(int j=i; j<size_u; j++)
    {
      if (i==j) {
        ret_[ind++]=L_diag[i].adj();
      } else {
        ret_[ind++]=L(i,j).adj();
      }
    }
  }

  // Memory is allocated on a global stack
  stan::math::recover_memory();
  stan::math::ChainableStack::memalloc_.free_all();

  return ret_;
}

// [[Rcpp::export]]
NumericVector bks_logml_sb(NumericVector vec_vi_par, NumericMatrix mat_xu,
                           NumericVector init_, NumericVector seq_, NumericVector op_,
                           NumericVector vec_sample_t,  NumericVector vec_sample_u,
                           NumericVector vec_yb, NumericMatrix mat_xb) {
  //output
  NumericVector ret_(vec_vi_par.size() + 1);
  using Eigen::Matrix;
  using Eigen::Dynamic;
  //size of hyperparameter
  int size_t=vec_sample_t.size();
  //size of support
  int size_u=vec_sample_u.size();
  //data points
  int size_b=vec_yb.size();
  //dimension
  int dim_x=mat_xb.nrow();
  //std::cout <<"size of support:" << size_u <<std::endl
  //          << "size of hyper:" << size_t<<std::endl
  //          << "size of block:" << size_b<<std::endl
  //          << "dim of input:" << dim_x <<std::endl;
  //
  stan::math::var logml=0;
  Matrix<stan::math::var, Dynamic, 1> t_s(size_t);
  stan::math::var t_noise;
  stan::math::var t_mean;
  Matrix<stan::math::var, Dynamic, 1> t_theta(size_t-2);

  Matrix<stan::math::var, Dynamic, 1> u_s(size_u);

  Matrix<stan::math::var, Dynamic, 1> t_m(size_t);
  Matrix<stan::math::var, Dynamic, 1> t_k(size_t);
  Matrix<stan::math::var, Dynamic, 1> u_m(size_u);
  Matrix<stan::math::var, Dynamic, 1> L_diag(size_u);
  Matrix<stan::math::var, Dynamic, Dynamic> L(size_u,size_u);
  Matrix<stan::math::var, Dynamic, Dynamic> sig_uu(size_u,size_u);

  Matrix<stan::math::var, Dynamic, 1> fb_m(size_b);
  Matrix<stan::math::var, Dynamic, Dynamic> fb_k(size_b,size_b);
  Matrix<stan::math::var, Dynamic, Dynamic> sig_ufb(size_u,size_b);
  Matrix<stan::math::var, Dynamic, Dynamic> sig_fbfb(size_b,size_b);

  Matrix<double, Dynamic, 1> sample_t(size_t);
  Matrix<double, Dynamic, 1> sample_u(size_u);
  Matrix<double, Dynamic, Dynamic> xu(dim_x,size_u);
  Matrix<double, Dynamic, Dynamic> xb(dim_x,size_b);
  Matrix<double, Dynamic, 1> yb(size_b);
  Matrix<int, Dynamic, 1> op(op_.size());
  Matrix<int, Dynamic, 1> seq(seq_.size());
  int init;
  //initialization
  init=init_[0];
  for(int i=0; i<op_.size(); i++)
  {
    op[i]=op_[i];
    seq[i]=seq_[i];
  }

  //std::cout<<init<<std::endl<<op<<std::endl<<seq<<std::endl;


  for(int i=0; i<size_t; i++)
  {
    t_m[i]=vec_vi_par[i];
    t_k[i]=vec_vi_par[size_t+i];
    sample_t[i]=vec_sample_t[i];
  }

  // std::cout<<"mean of hyper:"<<t_m.transpose()<<std::endl
  //           <<"log sd of hyper:"<<t_k.transpose()<<std::endl
  //           << "sample of std norm:"<<sample_t.transpose()<<std::endl;


  for(int i=0; i<size_u; i++)
  {
    u_m[i]=vec_vi_par[size_t*2+i];
    sample_u[i]=vec_sample_u[i];
  }

  //std::cout<<u_m<<std::endl<<sample_u<<std::endl;


  L.fill(0);
  int ind=size_t*2+size_u;
  for(int i=0; i<size_u; i++)
  {
    for(int j=i; j<size_u; j++)
    {
      if (i==j) {
        L_diag[i] = vec_vi_par[ind++];
        L(i,j) = stan::math::exp(L_diag[i]);
      } else {
        L(i,j)=vec_vi_par[ind++];
      }
    }
  }
  //std::cout.precision(5);
  //std::cout<<L<<std::endl<<L.rows()<<std::endl<<L.cols()<<std::endl;

  for(int i=0; i<dim_x; i++)
  {
    for(int j=0; j<size_u; j++)
    {
      xu(i,j)=mat_xu(i,j);
    }
  }

  for(int j=0; j<size_b; j++)
  {
    yb[j]=vec_yb[j];
    for(int i=0; i<dim_x; i++)
    {
      xb(i,j)=mat_xb(i,j);
    }
  }

  //std::cout<<xu.cols()<<","<<xu.rows()<<","<<xb.cols()<<","<<xb.rows()<<","<<yb.size()<<std::endl;

  //compute theta^(s) and u^s
  t_s=t_m+stan::math::elt_multiply(stan::math::exp(t_k),sample_t);
  u_s=u_m+stan::math::multiply(L,sample_u);
  //std::cout<<"sample of hyper:"<<t_s.transpose()<<std::endl
  //         <<"sample of u:"<<u_s.transpose()<<std::endl;

  cov_comp cov(init,op,seq);

  t_mean=t_s[0];
  t_noise=t_s[size_t-1];
  for(int i=1; i<size_t-1; i++) {
    t_theta[i-1]=t_s[i];
  }

  //compute covariance matrix


  //compute sig_uu
  for(int i=0; i<size_u; i++){
    for(int j=i; j<size_u; j++){
      Matrix<double, Dynamic, 1> xi(dim_x);
      Matrix<double, Dynamic, 1> xj(dim_x);
      for(int k=0; k<dim_x; k++){
        xi(k)=xu(k,i);
        xj(k)=xu(k,j);
      }
      sig_uu(i,j)=cov(t_theta,xi,xj);
      if(i!=j)
      {
        sig_uu(j,i)=sig_uu(i,j);
      } else {
        sig_uu(i,j)=sig_uu(i,j)+0.0001;
      }
    }
  }
  //std::cout.precision(2);
  //std::cout << sig_uu <<std::endl;

  for(int i=0; i<size_b; i++){
    for(int j=i; j<size_b; j++){
      Matrix<double, Dynamic, 1> xi(dim_x);
      Matrix<double, Dynamic, 1> xj(dim_x);
      for(int k=0; k<dim_x; k++){
        xi(k)=xb(k,i);
        xj(k)=xb(k,j);
      }
      sig_fbfb(i,j)=cov(t_theta,xi,xj);
      if(i!=j)
      {
        sig_fbfb(j,i)=sig_fbfb(i,j);
      } else {
        sig_fbfb(i,j)=sig_fbfb(i,j)+stan::math::exp(2*t_noise);
      }
    }

    for(int j=0; j<size_u; j++){
      Matrix<double, Dynamic, 1> xi(dim_x);
      Matrix<double, Dynamic, 1> xj(dim_x);
      for(int k=0; k<dim_x; k++){
        xi(k)=xb(k,i);
        xj(k)=xu(k,j);
      }
      sig_ufb(j,i)=cov(t_theta,xi,xj);
    }
  }

  //compute fb_m fb_k
  Matrix<stan::math::var, Dynamic, Dynamic> Lu(size_u,size_u);
  Matrix<stan::math::var, Dynamic, 1> alphau(size_u);
  Matrix<stan::math::var, Dynamic, Dynamic> vub(size_u,size_b);
  Lu=stan::math::cholesky_decompose(sig_uu);
  //std::cout.precision(5);
  //std::cout << Lu <<std::endl;
  //          << stan::math::multiply(Lu.transpose(),Lu)<<std::endl;
  //          << stan::math::multiply(Lu,Lu.transpose())<<std::endl;
  //stan::math::mdivide_left_tri_low(Lu,u)
  alphau=stan::math::mdivide_left_tri_low(Lu,u_s);
  vub=stan::math::mdivide_left_tri_low(Lu,sig_ufb);
  fb_m=stan::math::multiply(transpose(vub),alphau);
  fb_k=sig_fbfb-stan::math::multiply(transpose(vub),vub);
  //it suppose to have simi
  //std::ofstream output("log.txt");
  //output << fb_m << std::endl;

  //std::cout<<"MSE: "<<stan::math::dot_self(stan::math::subtract(fb_m,yb))/size_b<<std::endl;
          //<<", std: "<<stan::math::sd(stan::math::subtract(fb_m,yb))<<std::endl;
  stan::math::var logml0=0;
  //compute logml
  logml+=-0.5*size_b*(2*t_noise+log(2*stan::math::pi()));
  //std::cout<<"3:"<<logml<<",";
  logml0=logml;
  //std::cout<<"1:"<<-0.5*size_b*(t_noise+log(2*stan::math::pi()))<<std::endl;
  logml+= -0.5*sum(stan::math::divide(stan::math::diagonal(fb_k),exp(2*t_noise)));
  //std::cout<<"2:"<<logml-logml0<<",";
  logml0=logml;
  //std::cout<<"2:"<<-0.5*sum(stan::math::divide(stan::math::diagonal(fb_k),exp(t_noise)))<<std::endl;
  //fb_m=stan::math::multiply( stan::math::subtract(fb_m,stan::math::subtract(yb,t_mean)),exp(t_noise));
  logml+=-0.5*stan::math::dot_self(stan::math::divide( stan::math::subtract(fb_m,stan::math::subtract(yb,t_mean)),exp(t_noise)));
  //std::cout<<"1:"<<logml-logml0<<"|mu:"<<stan::math::dot_self(stan::math::subtract(fb_m,stan::math::subtract(yb,t_mean)))/size_b<<std::endl;

  //std::cout<<"3:"<<-0.5*stan::math::dot_self(stan::math::divide( stan::math::subtract(fb_m,stan::math::subtract(yb,t_mean)),exp(0.5*t_noise)))<<std::endl;

  //return
  ret_[0]=logml.val();
  logml.grad();
  ind=1;
  for(int i=0; i<size_t; i++)
  {
    ret_[ind++]=t_m[i].adj();
  }
  for(int i=0; i<size_t; i++)
  {
    ret_[ind++]=t_k[i].adj();
  }
  for(int i=0; i<size_u; i++)
  {
    ret_[ind++]=u_m[i].adj();
  }
  for(int i=0; i<size_u; i++)
  {
    for(int j=i; j<size_u; j++)
    {
      if (i==j) {
        ret_[ind++]=L_diag[i].adj();
      } else {
        ret_[ind++]=L(i,j).adj();
      }
    }
  }

  // Memory is allocated on a global stack
  stan::math::recover_memory();
  stan::math::ChainableStack::memalloc_.free_all();

  return ret_;
}


// [[Rcpp::export]]
NumericVector bks_logml_minibatch(NumericVector vec_vi_par, NumericMatrix mat_xu,
                           NumericVector init_, NumericVector seq_, NumericVector op_,
                           NumericVector vec_sample_t,  NumericVector vec_sample_u,
                           NumericVector vec_yb, NumericMatrix mat_xb) {
  //output
  NumericVector ret_(vec_vi_par.size() + 1);
  using Eigen::Matrix;
  using Eigen::Dynamic;
  //size of hyperparameter
  int size_t=vec_sample_t.size();
  //size of support
  int size_u=vec_sample_u.size();
  //data points
  int size_b=vec_yb.size();
  //dimension
  int dim_x=mat_xb.nrow();
  //std::cout <<"size of support:" << size_u <<std::endl
  //          << "size of hyper:" << size_t<<std::endl
  //          << "size of block:" << size_b<<std::endl
  //          << "dim of input:" << dim_x <<std::endl;
  //
  stan::math::var logml=0;
  Matrix<stan::math::var, Dynamic, 1> t_s(size_t);
  stan::math::var t_noise;
  stan::math::var t_mean;
  Matrix<stan::math::var, Dynamic, 1> t_theta(size_t-2);

  Matrix<stan::math::var, Dynamic, 1> u_s(size_u);

  Matrix<stan::math::var, Dynamic, 1> t_m(size_t);
  Matrix<stan::math::var, Dynamic, 1> t_k(size_t);
  Matrix<stan::math::var, Dynamic, 1> u_m(size_u);
  Matrix<stan::math::var, Dynamic, 1> L_diag(size_u);
  Matrix<stan::math::var, Dynamic, Dynamic> L(size_u,size_u);
  Matrix<stan::math::var, Dynamic, Dynamic> sig_uu(size_u,size_u);

  Matrix<stan::math::var, Dynamic, 1> fb_m(size_b);
  Matrix<stan::math::var, Dynamic, 1> fb_k(size_b);
  Matrix<stan::math::var, Dynamic, Dynamic> sig_ufb(size_u,size_b);
  Matrix<stan::math::var, Dynamic, 1> sig_fbfb(size_b);

  Matrix<double, Dynamic, 1> sample_t(size_t);
  Matrix<double, Dynamic, 1> sample_u(size_u);
  Matrix<double, Dynamic, Dynamic> xu(dim_x,size_u);
  Matrix<double, Dynamic, Dynamic> xb(dim_x,size_b);
  Matrix<double, Dynamic, 1> yb(size_b);
  Matrix<int, Dynamic, 1> op(op_.size());
  Matrix<int, Dynamic, 1> seq(seq_.size());
  int init;
  //initialization
  init=init_[0];
  for(int i=0; i<op_.size(); i++)
  {
    op[i]=op_[i];
    seq[i]=seq_[i];
  }

  //std::cout<<init<<std::endl<<op<<std::endl<<seq<<std::endl;


  for(int i=0; i<size_t; i++)
  {
    t_m[i]=vec_vi_par[i];
    t_k[i]=vec_vi_par[size_t+i];
    sample_t[i]=vec_sample_t[i];
  }

  // std::cout<<"mean of hyper:"<<t_m.transpose()<<std::endl
  //           <<"log sd of hyper:"<<t_k.transpose()<<std::endl
  //           << "sample of std norm:"<<sample_t.transpose()<<std::endl;


  for(int i=0; i<size_u; i++)
  {
    u_m[i]=vec_vi_par[size_t*2+i];
    sample_u[i]=vec_sample_u[i];
  }

  //std::cout<<u_m<<std::endl<<sample_u<<std::endl;


  L.fill(0);
  int ind=size_t*2+size_u;
  for(int i=0; i<size_u; i++)
  {
    for(int j=i; j<size_u; j++)
    {
      if (i==j) {
        L_diag[i] = vec_vi_par[ind++];
        L(i,j) = stan::math::exp(L_diag[i]);
      } else {
        L(i,j)=vec_vi_par[ind++];
      }
    }
  }
  //std::cout.precision(5);
  //std::cout<<L<<std::endl<<L.rows()<<std::endl<<L.cols()<<std::endl;

  for(int i=0; i<dim_x; i++)
  {
    for(int j=0; j<size_u; j++)
    {
      xu(i,j)=mat_xu(i,j);
    }
  }

  for(int j=0; j<size_b; j++)
  {
    yb[j]=vec_yb[j];
    for(int i=0; i<dim_x; i++)
    {
      xb(i,j)=mat_xb(i,j);
    }
  }

  //std::cout<<xu.cols()<<","<<xu.rows()<<","<<xb.cols()<<","<<xb.rows()<<","<<yb.size()<<std::endl;

  //compute theta^(s) and u^s
  t_s=t_m+stan::math::elt_multiply(stan::math::exp(t_k),sample_t);
  u_s=u_m+stan::math::multiply(L,sample_u);
  //std::cout<<"sample of hyper:"<<t_s.transpose()<<std::endl
  //         <<"sample of u:"<<u_s.transpose()<<std::endl;

  cov_comp cov(init,op,seq);

  t_mean=t_s[0];
  t_noise=t_s[size_t-1];
  for(int i=1; i<size_t-1; i++) {
    t_theta[i-1]=t_s[i];
  }

  //compute covariance matrix


  //compute sig_uu
  for(int i=0; i<size_u; i++){
    for(int j=i; j<size_u; j++){
      Matrix<double, Dynamic, 1> xi(dim_x);
      Matrix<double, Dynamic, 1> xj(dim_x);
      for(int k=0; k<dim_x; k++){
        xi(k)=xu(k,i);
        xj(k)=xu(k,j);
      }
      sig_uu(i,j)=cov(t_theta,xi,xj);
      if(i!=j)
      {
        sig_uu(j,i)=sig_uu(i,j);
      } else {
        sig_uu(i,j)=sig_uu(i,j)+0.0001;
      }
    }
  }
  //std::cout.precision(2);
  //std::cout << sig_uu <<std::endl;

  for(int i=0; i<size_b; i++){
    //for(int j=i; j<size_b; j++){
      Matrix<double, Dynamic, 1> xi(dim_x);
      //Matrix<double, Dynamic, 1> xj(dim_x);
      for(int k=0; k<dim_x; k++){
        xi(k)=xb(k,i);
      //  xj(k)=xb(k,j);
      }
      sig_fbfb(i)=cov(t_theta,xi,xi)+stan::math::exp(2*t_noise);
      //if(i!=j)
      //{
      //  sig_fbfb(j,i)=sig_fbfb(i,j);
      //} else {
      //  sig_fbfb(i,j)=sig_fbfb(i,j)+stan::math::exp(2*t_noise);
      //}
    //}

    for(int j=0; j<size_u; j++){
      Matrix<double, Dynamic, 1> xi(dim_x);
      Matrix<double, Dynamic, 1> xj(dim_x);
      for(int k=0; k<dim_x; k++){
        xi(k)=xb(k,i);
        xj(k)=xu(k,j);
      }
      sig_ufb(j,i)=cov(t_theta,xi,xj);
    }
  }

  //compute fb_m fb_k
  Matrix<stan::math::var, Dynamic, Dynamic> Lu(size_u,size_u);
  Matrix<stan::math::var, Dynamic, 1> alphau(size_u);
  Matrix<stan::math::var, Dynamic, Dynamic> vub(size_u,size_b);
  Lu=stan::math::cholesky_decompose(sig_uu);
  //std::cout.precision(5);
  //std::cout << Lu <<std::endl;
  //          << stan::math::multiply(Lu.transpose(),Lu)<<std::endl;
  //          << stan::math::multiply(Lu,Lu.transpose())<<std::endl;
  //stan::math::mdivide_left_tri_low(Lu,u)
  alphau=stan::math::mdivide_left_tri_low(Lu,u_s);
  vub=stan::math::mdivide_left_tri_low(Lu,sig_ufb);
  fb_m=stan::math::multiply(transpose(vub),alphau);
  fb_k=sig_fbfb-stan::math::rows_dot_self(transpose(vub));
// output log?
  if(OUTPUT_LOG){
  //it suppose to have simi
  std::ofstream output("log_fit.txt", std::ios_base::app);

  output<< "fit:" << fb_m <<std::endl;
//<< "orig:"<< yb << std::endl;

  std::ofstream output2("log_sup.txt", std::ios_base::app);

  output2 //<< "input:" << xu << std::endl
    << "support:" << u_s <<std::endl;
  }

  stan::math::var logml0=0;
  //compute logml
  logml+=-0.5*size_b*(2*t_noise+log(2*stan::math::pi()));
  //std::cout<<"3:"<<logml<<",";
  logml0=logml;
  logml+= -0.5*sum(stan::math::divide(fb_k,exp(2*t_noise)));
  //std::cout<<"2:"<<logml-logml0<<",";
  logml0=logml;
  logml+=-0.5*stan::math::dot_self(stan::math::divide( stan::math::subtract(fb_m,stan::math::subtract(yb,t_mean)),exp(t_noise)));

  //return
  ret_[0]=logml.val();
  logml.grad();
  ind=1;
  for(int i=0; i<size_t; i++)
  {
    ret_[ind++]=t_m[i].adj();
  }
  for(int i=0; i<size_t; i++)
  {
    ret_[ind++]=t_k[i].adj();
  }
  for(int i=0; i<size_u; i++)
  {
    ret_[ind++]=u_m[i].adj();
  }
  for(int i=0; i<size_u; i++)
  {
    for(int j=i; j<size_u; j++)
    {
      if (i==j) {
        ret_[ind++]=L_diag[i].adj();
      } else {
        ret_[ind++]=L(i,j).adj();
      }
    }
  }

  // Memory is allocated on a global stack
  stan::math::recover_memory();
  stan::math::ChainableStack::memalloc_.free_all();

  return ret_;
}

// [[Rcpp::export]]
NumericMatrix bks_cov_fu(NumericVector theta_, NumericVector init_, NumericVector seq_, NumericVector op_, NumericMatrix mat_xf, NumericMatrix mat_xu) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  int size_f=mat_xf.ncol();
  int size_u=mat_xu.ncol();
  int dim_x=mat_xu.nrow();
  NumericMatrix sig_fu(size_f,size_u);
  Matrix<double, Dynamic, 1> t_theta(theta_.size());
  Matrix<int, Dynamic, 1> op(op_.size());
  Matrix<int, Dynamic, 1> seq(seq_.size());
  int init;
  //initialization
  init=init_[0];
  for(int i=0; i<op_.size(); i++)
  {
    op[i]=op_[i];
    seq[i]=seq_[i];
  }
  cov_comp cov(init,op,seq);

  for(int i=0; i<theta_.size(); i++)
  {
    t_theta[i]=theta_[i];
  }

  //compute sig_uu
  for(int i=0; i<size_f; i++){
    for(int j=i; j<size_u; j++){
      Matrix<double, Dynamic, 1> xi(dim_x);
      Matrix<double, Dynamic, 1> xj(dim_x);
      for(int k=0; k<dim_x; k++){
        xi(k)=mat_xf(k,i);
        xj(k)=mat_xu(k,j);
      }
      sig_fu(i,j)=cov(t_theta,xi,xj);
    }
  }

  return sig_fu;
}



// [[Rcpp::export]]
NumericMatrix bks_cov_uu(NumericVector theta_, NumericVector init_, NumericVector seq_, NumericVector op_, NumericMatrix mat_xu) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  int size_u=mat_xu.ncol();
  int dim_x=mat_xu.nrow();
  NumericMatrix sig_uu(size_u,size_u);
  Matrix<double, Dynamic, 1> t_theta(theta_.size());
  Matrix<int, Dynamic, 1> op(op_.size());
  Matrix<int, Dynamic, 1> seq(seq_.size());
  int init;
  //initialization
  init=init_[0];
  for(int i=0; i<op_.size(); i++)
  {
    op[i]=op_[i];
    seq[i]=seq_[i];
  }
  cov_comp cov(init,op,seq);

  for(int i=0; i<theta_.size(); i++)
  {
    t_theta[i]=theta_[i];
  }

  //compute sig_uu
  for(int i=0; i<size_u; i++){
    for(int j=i; j<size_u; j++){
      Matrix<double, Dynamic, 1> xi(dim_x);
      Matrix<double, Dynamic, 1> xj(dim_x);
      for(int k=0; k<dim_x; k++){
        xi(k)=mat_xu(k,i);
        xj(k)=mat_xu(k,j);
      }
      sig_uu(i,j)=cov(t_theta,xi,xj);
      if(i!=j)
      {
        sig_uu(j,i)=sig_uu(i,j);
      } else {
        sig_uu(i,j)=sig_uu(i,j)+0.0001;
      }
    }
  }

  return sig_uu;
  }

// [[Rcpp::export]]
NumericVector bks_global_kl_g(NumericVector vec_vi_qg, NumericVector vec_vi_pg, NumericVector knum) {
  // vec_vi_qg contains:
  // 1. means m_g (size_k)
  // 2. Choleskey factor L_g (size_k(size_k + 1) / 2)
  // Similarly for vec_vi_pg
  NumericVector ret_(vec_vi_qg.size() + 1);
  using Eigen::Matrix;
  using Eigen::Dynamic;

  int size_k=knum[0];
  stan::math::var kl_g=0;
  Matrix<stan::math::var, Dynamic, 1> qg_m(size_k); // mean vector for q(g)
  Matrix<stan::math::var, Dynamic, 1> qg_L_diag(size_k);
  Matrix<stan::math::var, Dynamic, Dynamic> qg_L(size_k,size_k); // choleskey factor for q(g)
  Matrix<stan::math::var, Dynamic, 1> pg_m(size_k); // mean vector for p(g)
  Matrix<stan::math::var, Dynamic, 1> pg_L_diag(size_k);
  Matrix<stan::math::var, Dynamic, Dynamic> pg_L(size_k,size_k); // choleskey factor for p(g)
  //initialize structure

  for(int i=0; i<size_k; i++)
  {
    qg_m[i]=vec_vi_qg[i];
    pg_m[i]=vec_vi_pg[i];
  }

  qg_L.fill(0);
  pg_L.fill(0);
  int ind=size_k;
  for(int i=0; i<size_k; i++)
  {
    for(int j=i; j<size_k; j++)
    {
      if (i==j) {
        qg_L_diag[i]=vec_vi_qg[ind];
        pg_L_diag[i]=vec_vi_pg[ind];
        qg_L(j,i)=stan::math::exp(qg_L_diag[i]);
        pg_L(j,i)=stan::math::exp(pg_L_diag[i]);
      } else {
        qg_L(j,i)=vec_vi_qg[ind];
        pg_L(j,i)=vec_vi_pg[ind];
      }
      ind++;
    }
  }

  //KLD
  kl_g += stan::math::log(stan::math::pow(stan::math::prod(stan::math::diagonal(pg_L)), 2)/stan::math::pow(stan::math::prod(stan::math::diagonal(qg_L)), 2));
  //std::cout<<"1:"<<kl_g<<std::endl;
  kl_g -= size_k;
  //kl_g += stan::math::trace(stan::math::multiply(stan::math::inverse(stan::math::multiply(pg_L, stan::math::transpose(pg_L))), stan::math::multiply_lower_tri_self_transpose(qg_L)));
  // tr[Kq/Kp]=tr[Lq^TLq (Lp^TLp)^-1]=tr[Lq^T (Lq Lp^{-1}) Lp^{-T}]=tr[(Lp^{-1}(Lq Lp^{-1})^T Lq)^T]
  kl_g += stan::math::trace(stan::math::multiply(stan::math::mdivide_left_tri_low(pg_L,stan::math::transpose(stan::math::mdivide_right_tri_low(qg_L,pg_L))),qg_L));
  //std::cout<<"2:"<<kl_g<<std::endl;

  //kl_g += stan::math::quad_form(stan::math::inverse(stan::math::multiply_lower_tri_self_transpose(qg_L)), stan::math::subtract(qg_m, pg_m));
  // m^T Kp^-1 m = m^T (Lp^T Lp)^{-1} m = m^T Lp^{-1} (m^T Lp^{-1})^T
  kl_g += stan::math::dot_self(stan::math::mdivide_right_tri_low(stan::math::transpose(stan::math::subtract(pg_m, qg_m)),pg_L));
  //std::cout<<"3:"<<kl_g<<std::endl;

  kl_g = 0.5*kl_g;

  //logml+=-0.5*sum(stan::math::elt_divide(stan::math::exp(2*t_k),stan::math::exp(2*t_k0)));
  //logml+=-0.5*stan::math::dot_product(stan::math::elt_divide(stan::math::subtract(t_m0,t_m),stan::math::exp(2*t_k0)),stan::math::subtract(t_m0,t_m));
  //logml+=-0.5*log(stan::math::prod(stan::math::exp(t_k0))/stan::math::prod(stan::math::exp(t_k)));
  //logml+=-0.5*(2*stan::math::sum(t_k0)-2*stan::math::sum(t_k));
  //logml+=size_t;

  ret_[0]=kl_g.val();
  kl_g.grad();
  ind=1;
  for(int i=0; i<size_k; i++)
  {
    ret_[ind++]=qg_m[i].adj();
  }
  for(int i=0; i<size_k; i++)
  {
    for(int j=i; j<size_k; j++)
    {
      if (i==j) {
        ret_[ind++]=qg_L_diag[i].adj();
      } else {
        ret_[ind++]=qg_L(j,i).adj();
      }
    }
  }

  // Memory is allocated on a global stack
  stan::math::recover_memory();
  stan::math::ChainableStack::memalloc_.free_all();

  return ret_;
}

// [[Rcpp::export]]
NumericVector bks_global_inner(NumericVector vec_vi_qg, NumericVector vec_L_i, NumericVector knum, NumericVector vec_sample_g) {
  int size_k=knum[0];
  NumericVector ret_(vec_vi_qg.size() + 1 + size_k );
  using Eigen::Matrix;
  using Eigen::Dynamic;

  stan::math::var inner=0;
  stan::math::var denom=0;
  Matrix<stan::math::var, Dynamic, 1> qg_m(size_k); // mean vector for q(g)
  Matrix<stan::math::var, Dynamic, 1> qg_L_diag(size_k);
  Matrix<stan::math::var, Dynamic, Dynamic> qg_L(size_k,size_k); // choleskey factor for q(g)
  Matrix<stan::math::var, Dynamic, 1> g_s(size_k);

  Matrix<double, Dynamic, 1> sample_g(size_k);
  //initialize structure

  for(int i=0; i<size_k; i++)
  {
    qg_m[i]=vec_vi_qg[i];
    sample_g[i] = vec_sample_g[i];
  }

  qg_L.fill(0);
  int ind=size_k;
  for(int i=0; i<size_k; i++)
  {
    for(int j=i; j<size_k; j++)
    {
      if (i==j) {
        qg_L_diag[i] = vec_vi_qg[ind];
        qg_L(j,i)=stan::math::exp(qg_L_diag[i]);
      } else {
        qg_L(j,i)=vec_vi_qg[ind]; // Lower triangular
      }
      ind++;
    }
  }

  g_s = qg_m + stan::math::multiply(qg_L, sample_g);

  double temp=0;
  for (int i = 0; i < size_k; i++) {
    temp = vec_L_i[i];
    inner += stan::math::exp(g_s[i]) * temp;
    denom += stan::math::exp(g_s[i]);
  }

  inner =  inner / denom;

  ret_[0] = inner.val();
  inner.grad();
  ind = 1;
  for (int i = 0; i < size_k; i++) {
    ret_[ind++] = qg_m[i].adj();
  }
  for(int i=0; i<size_k; i++) {
    for(int j=i; j<size_k; j++) {
      if (i==j) {
        ret_[ind++]=qg_L_diag[i].adj();
      } else {
        ret_[ind++]=qg_L(j,i).adj();
      }
    }
  }

  // Samples
  for(int i=0; i < size_k; i++) {
    ret_[ind++]= g_s[i].val();
  }

  // Memory is allocated on a global stack
  stan::math::recover_memory();
  stan::math::ChainableStack::memalloc_.free_all();

  return ret_;
}

// [[Rcpp::export]]
int test() {
  return 2;
}
