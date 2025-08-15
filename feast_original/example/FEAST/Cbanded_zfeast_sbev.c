/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!! FEAST Driver Example - Banded Storage
  !!!!!!! solving Ax=ex with A complex-symmetric (non-Hermitian)
  !!!!!!! James Kestyn, Eric Polizzi 2015
  !!!!!!! Eric Polizzi 2019
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>

#include "feast.h"
#include "feast_banded.h"

int main() {
  /*!!!!!!!!!!!!!!!!! Matrix declaration variable */
  FILE *fp;
  char name[]="system4.mtx";
  int  N,nnz,kl,ku,LDA;
  double *A;
   char UPLO='F';
  /*!!!!!!!!!!!!!!!!! Others */
  int  fpm[64]; 
  double epsout;
  int loop;
  int  i,j,k,n2,err;
  int  M0,M,info;
  double r,dum;
  double Emid[2];
  double *X; //! eigenvectors
  double *E,*res; //! eigenvalue+residual




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
  printf("dense matrix -system4- size %.d\n",N);
  printf("bandwidth %d\n",LDA);

  
  
  /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!! FEAST in banded format !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

  
  /*!!! search interval [Emin,Emax] including M eigenpairs*/
  Emid[0] = 4.0e0;
  Emid[1] = 0.0e0;
  r = 3.0e0;
  M0=40; // !! M0>=M

  /*!!!!!!!!!!!!! ALLOCATE VARIABLE */
  E=calloc(M0*2,sizeof(double));  // eigenvalues  // factor 2 for complex
  res=calloc(M0,sizeof(double));// eigenvectors 
  X=calloc(2*N*M0,sizeof(double));// residual (if needed) // factor 2 for complex


  /*!!!!!!!!!!!!  FEAST */
  feastinit(fpm);
  fpm[0]=1;  /*change from default value */
  zfeast_sbev(&UPLO,&N,&kl,A,&LDA,fpm,&epsout,&loop,Emid,&r,&M0,E,X,&M,res,&info);


  /*!!!!!!!!!! REPORT !!!!!!!!!*/
  printf("FEAST OUTPUT INFO %d\n",info);
  if (info!=0)  printf(" Cbanded_zfeast_sbev   -- failed\n");
  if (info==0) {
    printf(" Cbanded_zfeast_sbev   -- success\n");
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









