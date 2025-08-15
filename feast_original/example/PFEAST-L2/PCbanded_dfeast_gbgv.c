/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!! PFEAST Driver example - Banded Storage
  !!!!!!! solving Ax=ex with A and B real non-symmetric (non-Hermitian)
  !!!!!!! James Kestyn, Eric Polizzi 2015
  !!!!!!! Eric Polizzi 2019
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <mpi.h>

#include "feast.h"
#include "feast_banded.h"
int main(int argc, char **argv) {
  /*!!!!!!!!!!!!!!!!! Matrix declaration variable */
  FILE *fp;
  char nameA[]="system3.mtx";
  char nameB[]="system3B.mtx";
  int  N,nnz,kla,kua,klb,kub;
  int  LDA,LDB;
  double *A,*B;

  /*!!!!!!!!!!!!!!!!! Others */
  int  fpm[64]; 
  int loop;
  double epsout;
  int  i,j,k,n2,err;
  int  M0,M,info;
  double r,dum;
  double Emid[2];
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

  /////// for A
  // !! find kla,kua
  kla=0;
  kua=0;
  fp = fopen (nameA, "r");
  err=fscanf (fp, "%d%d%d\n",&N,&N,&nnz);
  for (k=0;k<=nnz-1;k++){
    err=fscanf(fp,"%d%d%lf\n",&i,&j,&dum);
    if (abs(i-j)>kla){kla=abs(i-j);}
    if (abs(j-i)>kua){kua=abs(j-i);}
  }
  fclose(fp);
  //  form the banded matrices A
  LDA=(kla+kua+1);
  n2=N*(kla+kua+1);
  A=calloc(n2,sizeof(double));
  memset(A,(double) 0.0,n2 * sizeof(double));

  fp = fopen (nameA, "r");
  err=fscanf (fp, "%d%d%d\n",&N,&N,&nnz); 
  for (k=0;k<=nnz-1;k++){
    err=fscanf(fp,"%d%d",&i,&j);
    err=fscanf(fp,"%lf\n",A+(j-1)*LDA+kua+(i-j));
  };
  fclose(fp);


  /////// for B
  // !! find klb,kub
  klb=0;
  kub=0;
  fp = fopen (nameB, "r");
  err=fscanf (fp, "%d%d%d\n",&N,&N,&nnz);
  for (k=0;k<=nnz-1;k++){
    err=fscanf(fp,"%d%d%lf\n",&i,&j,&dum);
    if (abs(i-j)>klb){klb=abs(i-j);}
    if (abs(j-i)>kub){kub=abs(j-i);}
  }
  fclose(fp);
  //  form the banded matrices A
  LDB=(klb+kub+1);
  n2=N*(klb+kub+1);
  B=calloc(n2,sizeof(double));
  memset(B,(double) 0.0,n2 * sizeof(double));

  fp = fopen (nameB, "r");
  err=fscanf (fp, "%d%d%d\n",&N,&N,&nnz); 
  for (k=0;k<=nnz-1;k++){
    err=fscanf(fp,"%d%d",&i,&j);
    err=fscanf(fp,"%lf\n",B+(j-1)*LDB+kub+(i-j));
  };
  fclose(fp);



  
  /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!! INFORMATION ABOUT MATRIX !!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
  printf("dense matrix -system3- size %.d\n",N);
  printf("bandwidth A and B %d %d \n",LDA,LDB);

  /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!! FEAST in banded format !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

  /*!!! search contour including M eigenpairs*/
  Emid[0] = 0.59e0;
  Emid[1] = 0.0e0;
  r = 0.41e0;
  M0=30; // !! M0>=M

  /*!!!!!!!!!!!!! ALLOCATE VARIABLE */
  E=calloc(M0*2,sizeof(double));  // eigenvalues  // factor 2 for complex
  res=calloc(M0*2,sizeof(double));// eigenvectors // factor 2 for L and R res
  X=calloc(2*N*M0*2,sizeof(double));// residual (if needed) // factor 2 for complex //factor 2 for L/R vectors


  /*!!!!!!!!!!!!  FEAST */
  feastinit(fpm);
  fpm[0]=1;  /*change from default value */
  dfeast_gbgv(&N,&kla,&kua,A,&LDA,&klb,&kub,B,&LDB,fpm,&epsout,&loop,Emid,&r,&M0,E,X,&M,res,&info);
 

  /*!!!!!!!!!! REPORT !!!!!!!!!*/
  if (rank==0) printf("FEAST OUTPUT INFO %d\n",info);
  if (info!=0 && rank==0)  printf(" PCbanded_dfeast_gbgv   -- failed\n");
  if (info==0 && rank==0) {
    printf(" PCbanded_dfeast_gbgv   -- success\n");
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




