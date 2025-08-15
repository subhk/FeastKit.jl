/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!! PFEAST Driver sparse example !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!! solving Ax=eBx with A real and B spd --- A and B sparse matrix!!!
  !!!!!!! by Eric Polizzi- 2009-2019!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "feast.h"
#include "feast_sparse.h"
int main(int argc, char **argv) {
  /*!!!!!!!!!!!!!!!!! Feast declaration variable */
  int  fpm[64]; 
  double epsout;
  int loop;
  char UPLO='F'; // full csr format here

  /*!!!!!!!!!!!!!!!!! Matrix declaration variable */
  FILE *fp;
  char nameA[]="system1.mtx";
  char nameB[]="system1B.mtx";
  int  N,nnz;
  double *sa,*sb;
  int *isa,*jsa,*isb,*jsb;
  /*!!!!!!!!!!!!!!!!! Others */
  int  i,k,err;
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
  

  /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!! read input file in csr format!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

  // !!!!!!!!!! form CSR arrays isa,jsa,sa 
  fp = fopen (nameA, "r");
  err=fscanf (fp, "%d%d%d\n",&N,&N,&nnz);
  sa=calloc(nnz,sizeof(double));
  isa=calloc(N+1,sizeof(int));
  jsa=calloc(nnz,sizeof(int));

  for (i=0;i<=N;i++){
    *(isa+i)=0;
  };
  *(isa)=1;
  for (k=0;k<=nnz-1;k++){
    err=fscanf(fp,"%d%d%lf\n",&i,jsa+k,sa+k);
    *(isa+i)=*(isa+i)+1;
  };
  fclose(fp);
  for (i=1;i<=N;i++){
    *(isa+i)=*(isa+i)+*(isa+i-1);
  };


  // !!!!!!!!!! form CSR arrays isb,jsb,sb 
  fp = fopen (nameB, "r");
  err=fscanf (fp, "%d%d%d\n",&N,&N,&nnz);
  sb=calloc(nnz,sizeof(double));
  isb=calloc(N+1,sizeof(int));
  jsb=calloc(nnz,sizeof(int));

  for (i=0;i<=N;i++){
    *(isb+i)=0;
  };
  *(isb)=1;
  for (k=0;k<=nnz-1;k++){
    err=fscanf(fp,"%d%d%lf\n",&i,jsb+k,sb+k);
    *(isb+i)=*(isb+i)+1;
  };
  fclose(fp);
  for (i=1;i<=N;i++){
    *(isb+i)=*(isb+i)+*(isb+i-1);
  };

  
  /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!! FEAST in sparse format !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
  dfeast_scsrgv(&UPLO,&N,sa,isa,jsa,sb,isb,jsb,fpm,&epsout,&loop,&Emin,&Emax,&M0,E,X,&M,res,&info);

  /*!!!!!!!!!! REPORT !!!!!!!!!*/
  if (rank==0) printf("FEAST OUTPUT INFO %d\n",info);
  if (info!=0 && rank==0)  printf(" PCsparse_dfeast_scsrgv   -- failed\n");
  if (info==0 && rank==0) {
    printf(" PCsparse_dfeast_scsrgv   -- success\n");
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
