/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! PFEAST Driver sparse example !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! solving quadratic eigenvalue problem (A2e^2+A1e^1+A0)x=0 with A2,A1,A0 real symmetric and sparse matrix!!!!
!!!!!!! by Eric Polizzi- 2019       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "pfeast.h"
#include "pfeast_sparse.h"
int main(int argc, char **argv) {
 
  /*!!!!!!!!!!!!!!!!! Matrix declaration variable */
  FILE *fp;
  char nameA0[]="system5A0.mtx";
  char nameA1[]="system5A1.mtx";
  char nameA2[]="system5A2.mtx";
  int  N,nnz,nnzt;
  double *sa;
  int *isa,*jsa;
  char UPLO='F';
  
  /*!!!!!!!!!!!!!!!!! Others */
   int  fpm[64]; 
  double epsout;
  int loop;
  int  i,k,err,d;
  int  M0,M,info;
  double Emid[2],r;
  double *X; //! eigenvectors
  double *E,*res; //! eigenvalue+residual

  int nL3;
 /*********** MPI *****************************/
int rank,numprocs;

MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
// convert C communicator to Fortran type
MPI_Fint fortran_comm_world = MPI_Comm_c2f(MPI_COMM_WORLD );
/*********************************************/ 

// READ INPUT # of procs in L3
nL3=atoi(argv[1]);

/*
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!Create Matrices
!!!!!!!!! We consider the nonoverdamped mass-spring system
!!!!!!!!! Example 1 in
!!!Brendan Gavin, Agnieszka Międlar, Eric Polizzi,
!!!FEAST eigensolver for nonlinear eigenvalue problems,
!!!!Journal of Computational Science,Volume 27, 2018,Pages 107-117
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! A[2]=I;  A[1]=tau*tridiag[-1,3,-1]nxn; A[0]=kappa*tridiag[-1,3,-1]nxn
  !!tau=0.6202d0 kappa=0.4807d0*/

  N=1000;
  nnzt=3*N-2; //! should be the nnz max that we apply to all matrices (know a priori here)

  sa=calloc(3*nnzt,sizeof(double));
  isa=calloc(3*(N+1),sizeof(int));
  jsa=calloc(3*nnzt,sizeof(int));
  memset(isa,(int) 0,(N+1)*3*sizeof(int));
    
  // A0
  fp = fopen (nameA0, "r");
  err=fscanf (fp, "%d%d%d\n",&N,&N,&nnz);
  *(isa)=1;
  for (k=0;k<=nnz-1;k++){
    err=fscanf(fp,"%d%d%lf\n",&i,jsa+k,sa+k);
    *(isa+i)=*(isa+i)+1;
  };
  fclose(fp);
  for (i=1;i<=N;i++){
    *(isa+i)=*(isa+i)+*(isa+i-1);
  };
// A1
  fp = fopen (nameA1, "r");
  err=fscanf (fp, "%d%d%d\n",&N,&N,&nnz);
  *(isa+N+1)=1;
  for (k=0;k<=nnz-1;k++){
    err=fscanf(fp,"%d%d%lf\n",&i,jsa+k+nnzt,sa+k+nnzt);
    *(isa+i+N+1)=*(isa+i+N+1)+1;
  };
  fclose(fp);
  for (i=1;i<=N;i++){
    *(isa+i+N+1)=*(isa+i+N+1)+*(isa+i-1+N+1);
  };
// A2
  fp = fopen (nameA2, "r");
  err=fscanf (fp, "%d%d%d\n",&N,&N,&nnz);
  *(isa+2*(N+1))=1;
  for (k=0;k<=nnz-1;k++){
    err=fscanf(fp,"%d%d%lf\n",&i,jsa+k+2*nnzt,sa+k+2*nnzt);
    *(isa+i+2*(N+1))=*(isa+i+2*(N+1))+1;
  };
  fclose(fp);
  for (i=1;i<=N;i++){
    *(isa+i+2*(N+1))=*(isa+i+2*(N+1))+*(isa+i-1+2*(N+1));
  };


  
  /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!! FEAST in sparse format !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
  pfeastinit(fpm,&fortran_comm_world,&nL3);
  fpm[0]=1;  /*change from default value */
  fpm[17]=7; //! ellipse 7%
  d=2;
  pdfeast_scsrpev(&UPLO,&d,&N,sa,isa,jsa,fpm,&epsout,&loop,Emid,&r,&M0,E,X,&M,res,&info);

  /*!!!!!!!!!! REPORT !!!!!!!!!*/
  if (rank==0) printf("FEAST OUTPUT INFO %d\n",info);
  if (info!=0 && rank==0)  printf(" PCsparse_pdfeast_scsrpev  -- failed\n");
  if (info==0 && rank==0) {
    printf(" PCsparse_pdfeast_scsrpev  -- success\n");
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
