/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!! FEAST Driver banded example !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!! solving Ax=eBx with A real and B spd --- A and B banded matrix!!!
  !!!!!!! by Eric Polizzi- 2009-2019!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

#include <stdio.h> 
#include <stdlib.h>
#include <string.h>

#include "feast.h"
#include "feast_banded.h"
int main() {
 
  /*!!!!!!!!!!!!!!!!! Matrix declaration variable */
  FILE *fp;
  char nameA[]="system1.mtx";
  char nameB[]="system1B.mtx";
  int  N,nnz,kla,kua,klb,kub;
  double *A,*B;
  int LDA,LDB;
  char UPLO='F'; 
  /*!!!!!!!!!!!!!!!!! Others */
  int  fpm[64]; 
  double epsout;
  int  i,j,k,n2,err;
  int  M0,M,info,loop;
  double Emin,Emax,dum;
  double *X; //! eigenvectors
  double *E,*res; //! eigenvalue+residual

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
  printf("dense matrix -system1- size %.d\n",N);
  printf("bandwidth A and B %d %d \n",LDA,LDB);

  /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!! FEAST in banded format !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
  dfeast_sbgv(&UPLO,&N,&kla,A,&LDA,&klb,B,&LDB,fpm,&epsout,&loop,&Emin,&Emax,&M0,E,X,&M,res,&info);

  
  /*!!!!!!!!!! REPORT !!!!!!!!!*/
  printf("FEAST OUTPUT INFO %d\n",info);
  if (info!=0)  printf(" Cbanded_dfeast_sbgv   -- failed\n");
  if (info==0) {
    printf(" Cbanded_dfeast_sbgv   -- success\n");
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




