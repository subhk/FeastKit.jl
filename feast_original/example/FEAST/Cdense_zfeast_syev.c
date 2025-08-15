/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!! FEAST Driver - Dense Storage
  !!!!!!! solving Ax=ex with A complex-symmetric (non-Hermitian)
  !!!!!!! James Kestyn, Eric Polizzi 2015
  !!!!!!! Eric Polizzi 2019
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>


#include "feast.h"
#include "feast_dense.h"
int main() {
  /*!!!!!!!!!!!!!!!!! Matrix declaration variable */
  FILE *fp;
  char name[]="system4.mtx";
  int  N,LDA,LDB,nnz;
  double *A,*B;
  char UPLO='F';

  /*!!!!!!!!!!!!!!!!! Others */
  int  fpm[64]; 
  int loop;
  double Emid[2],epsout;
  double r;
  int  i,j,k,n2,err;
  int  M0,M,info;
  double *E,*XR; //! eigenvectors
  double *res; //! eigenvalue+resridual

  /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!Read Coordinate format and convert to dense format
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
  fp = fopen (name, "r");
  err=fscanf (fp, "%d%d%d\n",&N,&N,&nnz);
  n2=2*N*N;  // factor 2 because of complex number
  A=calloc(n2,sizeof(double));
  memset(A,(double) 0.0,n2 * sizeof(double));
  for (k=0;k<=nnz-1;k++){
    err=fscanf(fp,"%d %d",&i,&j);
    err=fscanf(fp,"%lf%lf\n",A+(j-1)*2*N+2*i-2,A+(j-1)*2*N+2*i-1);
  };
  fclose(fp);
  
  /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!! FEAST in dense format !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

  /*!!! search contour including M eigenpairs*/
  Emid[0] = 4.0e0;
  Emid[1] = 0.0e0;
  r = 3.0e0;
  M0=40; // !! M0>=M

  
  /*!!!!!!!!!!!!! ALLOCATE VARIABLE */
  E=calloc(2*M0,sizeof(double));  // eigenvalues
  XR=calloc(2*N*M0,sizeof(double));// right eigenvector // factor 2 because of complex number
  res=calloc(M0,sizeof(double));// eigenvector residual 
  
  /*!!!!!!!!!!!!  FEAST */
  feastinit(fpm);
  fpm[0]=1;  /*change from default value */
  zfeast_syev(&UPLO,&N,A,&N,fpm,&epsout,&loop,Emid,&r,&M0,E,XR,&M,res,&info);


  /*!!!!!!!!!! REPORT !!!!!!!!!*/
  printf("FEAST OUTPUT INFO %d\n",info);
  if (info!=0)  printf(" Cdense_zfeast_syev   -- failed\n");
  if (info==0) {
    printf(" Cdense_zfeast_syev   -- success\n");
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

