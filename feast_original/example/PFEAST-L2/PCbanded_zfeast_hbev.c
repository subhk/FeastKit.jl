/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!! PFEAST Driver banded example !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!! solving Ax=ex with A complex Hermitian --- A banded matrix!!!!!!!
  !!!!!!! by Eric Polizzi- 2009-2019!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "feast.h"
#include "feast_banded.h"
int main(int argc, char **argv) {
  /*!!!!!!!!!!!!!!!!! Feast declaration variable */
  int  fpm[64]; 
  double epsout;
  int loop,LDA;
  char UPLO='F'; // ! 'L' or 'U' also fine

  /*!!!!!!!!!!!!!!!!! Matrix declaration variable */
  FILE *fp;
  char name[]="system2.mtx";
  int  N,nnz,kl,ku;
  double *A;

  /*!!!!!!!!!!!!!!!!! Others */
  int  i,j,k,n2,err;
  int  M0,M,info;
  double Emin,Emax,dum;
  double *X; //! eigenvectors
  double *E,*res; //! eigenvalue+residual

/*********** MPI *****************************/
int rank,numprocs;
MPI_Init(&argc,&argv);
//MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
/*********************************************/

  /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!! read input file in coordinate format!!!!!!!
    !!!!!!!!!!!!!!!!transform  in banded format directly !!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

  // !! find kl,ku (kl==ku here)
  kl=0;
  ku=0;
  fp = fopen (name, "r");
  err=fscanf (fp, "%d%d%d\n",&N,&N,&nnz);
  for (k=0;k<=nnz-1;k++){
    err=fscanf(fp,"%d%d%lf%lf\n",&i,&j,&dum,&dum);
    if (abs(i-j)>kl){ kl=abs(i-j); }
    if (abs(j-i)>ku){ku=abs(j-i);}
  }
  fclose(fp);

  // !! form the banded matrix A 
  LDA=(kl+ku+1);
  n2=2*N*LDA; // factor 2 for complex
  A=calloc(n2,sizeof(double));
  memset(A,(double) 0.0,n2 * sizeof(double));

  fp = fopen (name, "r");
  err=fscanf (fp, "%d%d%d\n",&N,&N,&nnz); 
  for (k=0;k<=nnz-1;k++){
    err=fscanf(fp,"%d%d",&i,&j);
    err=fscanf(fp,"%lf%lf\n",A+(j-1)*2*LDA+2*(ku+1+(i-j))-2,A+(j-1)*2*LDA+2*(ku+1+i-j)-1);
  };
  fclose(fp);


  /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!! INFORMATION ABOUT MATRIX !!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
  printf("dense matrix -system2- size %.d\n",N);
  printf("bandwidth %d\n",LDA);

  
 
  /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!! FEAST in banded format !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

  /*!!! search interval [Emin,Emax] including M eigenpairs*/
  Emin=(double) -0.35;
  Emax=(double) 0.23;
  M0=40; // !! M0>=M

  /*!!!!!!!!!!!!! ALLOCATE VARIABLE */
  E=calloc(M0,sizeof(double));  // eigenvalues
  res=calloc(M0,sizeof(double));// residual
  X=calloc(2*N*M0,sizeof(double));// eigenvectors // factor 2 for complex


  /*!!!!!!!!!!!!  FEAST */
  feastinit(fpm);
  fpm[0]=1;  /*change from default value */
  zfeast_hbev(&UPLO,&N,&kl,A,&LDA,fpm,&epsout,&loop,&Emin,&Emax,&M0,E,X,&M,res,&info);


  /*!!!!!!!!!! REPORT !!!!!!!!!*/
  if (rank==0) printf("FEAST OUTPUT INFO %d\n",info);
  if (info!=0 && rank==0)  printf(" PCbanded_zfeast_hbev   -- failed\n");
  if (info==0 & rank==0) {
    printf(" PCbanded_zfeast_hbev   -- success\n");
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




