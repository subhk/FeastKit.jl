/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!! FEAST Driver dense example !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 !!!!!!! solving quadratic eigenvalue problem (A2e^2+A1e^1+A0)x=0 with A2,A1,A0 real symmetric and dense matrix!!!!
  !!!!!!! by Eric Polizzi  2019   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
#include <stdio.h> 
#include <stdlib.h>
#include <string.h>

#include "feast.h"
#include "feast_dense.h"
int main() {

  /*!!!!!!!!!!!!!!!!! Matrix declaration variable */
  FILE *fp;
  char nameA0[]="system5A0.mtx";
  char nameA1[]="system5A1.mtx";
  char nameA2[]="system5A2.mtx"; 
  int  N,nnz;
  double *A;
  char UPLO='F';
  /*!!!!!!!!!!!!!!!!! Others */
  int  i,j,k,n2,err,d;
  int  M0,M,info;
  double Emid[2],r;
  double *X; //! eigenvectors
  double *E,*res; //! eigenvalue+residual
  int  fpm[64]; 
  double epsout;
  int loop;

/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!Create Matrices
!!!!!!!!! We consider the nonoverdamped mass-spring system
!!!!!!!!! Example 1 in
!!!Brendan Gavin, Agnieszka Międlar, Eric Polizzi,
!!!FEAST eigensolver for nonlinear eigenvalue problems,
!!!!Journal of Computational Science,Volume 27, 2018,Pages 107-117
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! A[2]=I;  A[1]=tau*tridiag[-1,3,-1]nxn; A[0]=kappa*tridiag[-1,3,-1]nxn   */ 
// tau=(double) 0.6202; and kappa=(double) 0.4807;
  
  N=1000;
  A=calloc(N*N*3,sizeof(double));
  memset(A,(double) 0.0,N*N*3*sizeof(double));

  //// Matrix A0
  fp = fopen (nameA0, "r");
  err=fscanf (fp, "%d%d%d\n",&N,&N,&nnz);
  for (k=0;k<=nnz-1;k++){
    err=fscanf(fp,"%d%d",&i,&j);
    err=fscanf(fp,"%lf\n",A+(j-1)*N+i-1);
  };
  fclose(fp);
 //// Matrix A1
  fp = fopen (nameA1, "r");
  err=fscanf (fp, "%d%d%d\n",&N,&N,&nnz);
  for (k=0;k<=nnz-1;k++){
    err=fscanf(fp,"%d%d",&i,&j);
    err=fscanf(fp,"%lf\n",A+(j-1)*N+i-1+N*N);
  };
  fclose(fp);
 //// Matrix A2
  fp = fopen (nameA2, "r");
  err=fscanf (fp, "%d%d%d\n",&N,&N,&nnz);
  for (k=0;k<=nnz-1;k++){
    err=fscanf(fp,"%d%d",&i,&j);
    err=fscanf(fp,"%lf\n",A+(j-1)*N+i-1+2*N*N);
  };
  fclose(fp); 

  
  //   for (k=0;k<N*N*3;k++){
  // printf("A %d %lf\n",k,*(A+k));  
  //};

  
    
  /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!! FEAST in dense format !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

  /*!!! search interval [Emin,Emax] including M eigenpairs*/
  Emid[0]=-1.55e0;
  Emid[1]= 0.0e0;
  r= (double) 0.05;
  M0=30;// !! M0>=M


  /*!!!!!!!!!!!!! ALLOCATE VARIABLE */
  E=calloc(M0,2*sizeof(double));  // eigenvalues (complex)
  res=calloc(M0,sizeof(double));// residual 
  X=calloc(N*M0,2*sizeof(double));// eigenvectors (complex) 


  /*!!!!!!!!!!!!  FEAST */
  feastinit(fpm);
  fpm[0]=1;  /*change from default value */
  fpm[17]=7; //! ellipse 7%
  d=2;
  dfeast_sypev(&UPLO,&d,&N,A,&N,fpm,&epsout,&loop,Emid,&r,&M0,E,X,&M,res,&info);
  
  /*!!!!!!!!!! REPORT !!!!!!!!!*/
  printf("FEAST OUTPUT INFO %d\n",info);
  if (info!=0)  printf(" Cdense_dfeast_sypev  -- failed\n");
  if (info==0) {
    printf(" Cdense_dfeast_sypev  -- success\n");
    printf("*************************************************\n");
    printf("************** REPORT ***************************\n");
    printf("*************************************************\n");
    printf("Eigenvalues/Residuals\n");
    for (i=0;i<=M-1;i=i+1){
      printf("   %d %.15e %.15e\n",i+1,*(E+i),*(res+i));
    }
  }
  return 0;
}
