/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!! PFEAST Driver sparse example !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!! solving Ax=ex with A complex Hermitian --- A sparse matrix!!
  !!!!!!! Using two search intervals and three levels of parallelism
  !!!!!!! by Eric Polizzi- 2009-2012!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <mpi.h>

#include "pfeast.h"
#include "pfeast_sparse.h"
int main(int argc, char **argv) {   

  /*!!!!!!!!!!!!!!!!! Matrix declaration variable */
  FILE *fp;
  char name[]="system2.mtx";
  int  N,nnz;
  double *sa;
  int *isa,*jsa;
  char UPLO='F';
  /*!!!!!!!!!!!!!!!!! Others */
  int  fpm[64]; 
  double epsout;
  int loop;
  int  i,k,err;
  int  M0,M,info;
  double Emin,Emax,trace;
  double *X; //! eigenvectors
  double *E,*res; //! eigenvalue+residual

int nL3;
/*********** MPI *****************************/
int lrank,lnumprocs,color,key;
int rank,numprocs;
MPI_Comm NEW_COMM_WORLD;

MPI_Init(&argc,&argv); 
MPI_Comm_size(MPI_COMM_WORLD,&numprocs); 
MPI_Comm_rank(MPI_COMM_WORLD,&rank); 
/*********************************************/

// READ INPUT # of procs in L3
nL3=atoi(argv[1]);
 
  
  /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!! read input file in csr format!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

  // !!!!!!!!!! form CSR arrays isa,jsa,sa 
  fp = fopen (name, "r");
  err=fscanf (fp, "%d%d%d\n",&N,&N,&nnz);
  sa=calloc(2*nnz,sizeof(double)); //factor 2 for complex 
  isa=calloc(N+1,sizeof(int));
  jsa=calloc(nnz,sizeof(int));

  for (i=0;i<=N;i++){
    *(isa+i)=0;
  };
  *(isa)=1;
  for (k=0;k<=nnz-1;k++){
    err=fscanf(fp,"%d%d%lf%lf\n",&i,jsa+k,sa+2*k,sa+2*k+1);
    *(isa+i)=*(isa+i)+1;
  };
  fclose(fp);
  for (i=1;i<=N;i++){
    *(isa+i)=*(isa+i)+*(isa+i-1);
  };

  /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!! FEAST in sparse format !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/


/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!! Definition of the two intervals 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

  if (rank<=numprocs/2-1) {
    color=1;} // first interval
  else {
    color=2; //! second interval
  }

  //!!!!!!!!!!!!!!!!! create new_mpi_comm_world
 key=0;
 MPI_Comm_split(MPI_COMM_WORLD,color,key,&NEW_COMM_WORLD);
 MPI_Comm_rank(NEW_COMM_WORLD,&lrank);
 // convert C communicator to Fortran type
 MPI_Fint fortran_comm_world = MPI_Comm_c2f(NEW_COMM_WORLD );
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  /*!!! search interval [Emin,Emax] including M eigenpairs*/
 if (color==1) { // 1st interval
  Emin=(double) -0.35;
  Emax=(double) 0.0;
  M0=40; // !! M0>=M
  }
 else if(color==2){ // 2nd interval
  Emin=(double) 0.0;
  Emax=(double) 0.23;
  M0=40; // !! M0>=M
  }

//!!!!!!!!!!!!!!!!!! RUN INTERVALS in PARALLEL

  /*!!!!!!!!!!!!! ALLOCATE VARIABLE */
  E=calloc(M0,sizeof(double));  // eigenvalues
  res=calloc(M0,sizeof(double));// residual
  X=calloc(2*N*M0,sizeof(double));// eigenvectors  // factor 2 for complex

  /*!!!!!!!!!!!!  FEAST */
  pfeastinit(fpm,&fortran_comm_world,&nL3);
  fpm[0]=-color;  /* print info the files feast1.log for contour1 and feast2.log for contour 2   */
  pzfeast_hcsrev(&UPLO,&N,sa,isa,jsa,fpm,&epsout,&loop,&Emin,&Emax,&M0,E,X,&M,res,&info);

 
  /*!!!!!!!!!! REPORT !!!!!!!!!*/
  if (rank==0) printf("FEAST OUTPUT INFO %d\n",info);
  if (info!=0 && rank==0)  printf(" 3PCsparse_pzfeast_hcsrev   -- failed\n");
  if (info==0 && rank==0) {
    printf(" 3PCsparse_pzfeast_hcsrev   -- success\n");
    printf("*************************************************\n");
    printf("************** REPORT ***************************\n");
    printf("*************************************************\n");
  }
    if (info==0 && lrank==0) { 
      printf("Eigenvalues/Residuals inside interval %d\n",color);
    for (i=0;i<=M-1;i=i+1){
      printf("   %d %.15e %.15e\n",i+1,*(E+i),*(res+i));
    }
  }

MPI_Finalize(); /************ MPI ***************/
  return 0;
}






