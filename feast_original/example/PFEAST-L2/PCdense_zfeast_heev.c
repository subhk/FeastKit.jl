/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!! PFEAST Driver dense example !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!! solving Ax=ex with A complex Hermitian --- A dense matrix!!!!!!!!
  !!!!!!! by Eric Polizzi- 2009-2019!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <mpi.h>

#include "feast.h"
#include "feast_dense.h"
int main(int argc, char **argv) {
  /*!!!!!!!!!!!!!!!!! Feast declaration variable */
  int  fpm[64]; 
  double epsout;
  int loop;
  char UPLO='F'; // ! 'L' or 'U' also fine here

  /*!!!!!!!!!!!!!!!!! Matrix declaration variable */
  FILE *fp;
  char name[]="system2.mtx";
  int  N,nnz;
  double *A;

  /*!!!!!!!!!!!!!!!!! Others */
  int  i,j,k,n2,err;
  int  M0,M,info;
  double Emin,Emax;
  double *X; //! eigenvectors
  double *E,*res; //! eigenvalue+residual

/*********** MPI *****************************/
int rank,numprocs;
MPI_Init(&argc,&argv);
//MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
/*********************************************/
  
  
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

  /*!!! search interval [Emin,Emax] including M eigenpairs*/
  Emin=(double) -0.35;
  Emax=(double) 0.23;
  M0=40; // !! M0>=M

  /*!!!!!!!!!!!!! ALLOCATE VARIABLE */
  E=calloc(M0,sizeof(double));  // eigenvalues
  res=calloc(M0,sizeof(double));// residual
  X=calloc(2*N*M0,sizeof(double));// eigenvectors  // factor 2 because of complex number


  /*!!!!!!!!!!!!  FEAST */
  feastinit(fpm);
  fpm[0]=1;  /*change from default value */
  zfeast_heev(&UPLO,&N,A,&N,fpm,&epsout,&loop,&Emin,&Emax,&M0,E,X,&M,res,&info);

  /*!!!!!!!!!! REPORT !!!!!!!!!*/
    if (rank==0) printf("FEAST OUTPUT INFO %d\n",info);
  if (info!=0 && rank==0)  printf(" PCdense_zfeast_heev   -- failed\n");
  if (info==0 && rank==0) {
    printf(" PCdense_zfeast_heev   -- success\n");
    printf("*************************************************\n");
    printf("************** REPORT ***************************\n");
    printf("*************************************************\n");
    printf("Eigenvalues/Residuals\n");
    for (i=0;i<=M-1;i=i+1){
      printf("   %d %.15e %.15e\n",i+1,*(E+i),*(res+i));
    }
  }
  
  MPI_Finalize(); /************ MPI ***************/
  return 0;
}
