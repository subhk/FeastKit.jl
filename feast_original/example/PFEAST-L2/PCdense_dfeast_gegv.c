/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!! PFEAST Driver Example  - Dense Storage
  !!!!!!! solving Ax=ex with A real non-symmetric 
  !!!!!!! James Kestyn, Eric Polizzi 2015
  !!!!!!! Eric Polizzi 2019
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <mpi.h>

#include "feast.h"
#include "feast_dense.h"
int main(int argc, char **argv) {
  /*!!!!!!!!!!!!!!!!! Matrix declaration variable */
  FILE *fp;
  char nameA[]="system3.mtx";
  char nameB[]="system3B.mtx";
  int  N,LDA,LDB,nnz;
  double *A,*B;
  
  /*!!!!!!!!!!!!!!!!! Others */  
  int  i,j,k,n2,err;
  int  M0,M,info;
  double *E,*X; //! eigenvalue+eigenvectors
  double *res; //! residual
  int  fpm[64]; 
  int loop;
  double Emid[2],epsout;
  double r;

/*********** MPI *****************************/
int rank,numprocs;
MPI_Init(&argc,&argv);
//MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
/*********************************************/
 
  
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

  // Search contour  including M eigenpairs*/
  Emid[0] = 0.590e0;
  Emid[1] = 0.0e0;
  r = 0.410e0;
  M0=30; // !! M0>=M

  /*!!!!!!!!!!!!! ALLOCATE VARIABLE */
  E=calloc(2*M0,sizeof(double));  // eigenvalues
  X=calloc(2*N*M0*2,sizeof(double));// right and left eigenvector // factor 2 because of complex number
  res=calloc(M0*2,sizeof(double));// right and left eigenvectors residual

  /*!!!!!!!!!!!!  FEAST */
  feastinit(fpm);
  fpm[0]=1;  /*change from default value */
  dfeast_gegv(&N,A,&N,B,&N,fpm,&epsout,&loop,Emid,&r,&M0,E,X,&M,res,&info);
  
   /*!!!!!!!!!! REPORT !!!!!!!!!*/
  if (rank==0) printf("FEAST OUTPUT INFO %d\n",info);
  if (info!=0 && rank==0)  printf(" PCdense_dfeast_gegv   -- failed\n");
  if (info==0 && rank==0) {
    printf(" PCdense_dfeast_gegv   -- success\n");
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



