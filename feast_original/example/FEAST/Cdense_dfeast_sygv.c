/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!! FEAST Driver dense example !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!! solving Ax=eBx with A real and B spd --- A and B dense matrix!!!!
  !!!!!!! by Eric Polizzi- 2009-2019!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
#include <stdio.h> 
#include <stdlib.h>
#include <string.h>

#include "feast.h"
#include "feast_dense.h"
int main() {

  /*!!!!!!!!!!!!!!!!! Matrix declaration variable */
  FILE *fp;
  char nameA[]="system1.mtx";
  char nameB[]="system1B.mtx";
  int  N,nnz;
  double *A,*B;
  char UPLO='F';

  /*!!!!!!!!!!!!!!!!! Others */
  int  i,j,k,n2,err;
  int  M0,M,info;
  double Emin,Emax;
  double *X; //! eigenvectors
  double *E,*res; //! eigenvalue+residual
  int  fpm[64]; 
  double epsout;
  int loop;
  

  /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!Read Coordinate format and convert to dense format
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
  //// Matrix A
  fp = fopen (nameA, "r");
  err=fscanf (fp, "%d%d%d\n",&N,&N,&nnz);
  n2=N*N;
  A=calloc(n2,sizeof(double));
  memset(A,(double) 0.0,n2 * sizeof(double));
  for (k=0;k<=nnz-1;k++){
    err=fscanf(fp,"%d%d",&i,&j);
    err=fscanf(fp,"%lf\n",A+(j-1)*N+i-1);
  };
  fclose(fp);   
  //// Matrix B
  fp = fopen (nameB, "r");
  err=fscanf (fp, "%d%d%d\n",&N,&N,&nnz);
  n2=N*N;
  B=calloc(n2,sizeof(double));
  memset(B,(double) 0.0,n2 * sizeof(double));
   for (k=0;k<=nnz-1;k++){
    err=fscanf(fp,"%d%d",&i,&j);
    err=fscanf(fp,"%lf\n",B+(j-1)*N+i-1);
  };
  fclose(fp);
  
  /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!! FEAST in dense format !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

  /*!!! search interval [Emin,Emax] including M eigenpairs*/
  Emin=(double) 0.18;
  Emax=(double) 1.0;
  M0=25; // !! M0>=M

  /*!!!!!!!!!!!!! ALLOCATE VARIABLE */
  E=calloc(M0,sizeof(double));  // eigenvalues
  res=calloc(M0,sizeof(double));// eigenvectors 
  X=calloc(N*M0,sizeof(double));// residual 


  /*!!!!!!!!!!!!  FEAST */
  feastinit(fpm);
  fpm[0]=1;  /*change from default value */
  dfeast_sygv(&UPLO,&N,A,&N,B,&N,fpm,&epsout,&loop,&Emin,&Emax,&M0,E,X,&M,res,&info);

  /*!!!!!!!!!! REPORT !!!!!!!!!*/
  printf("FEAST OUTPUT INFO %d\n",info);
  if (info!=0)  printf(" Cdense_dfeast_sygv   -- failed\n");
  if (info==0) {
    printf(" Cdense_dfeast_sygv   -- success\n");
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
