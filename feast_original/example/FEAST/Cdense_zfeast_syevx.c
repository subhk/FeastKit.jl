/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!! FEAST Expert Driver - Dense Storage
  !!!!!!! solving Ax=ex with A complex-symmetric (non-Hermitian)
  !!!!!!! James Kestyn, Eric Polizzi 2015
  !!!!!!! Eric Polizzi 2019
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>


#include "feast.h"
#include "feast_dense.h"
int main() {
 
  /*!!!!!!!!!!!!!!!!! Matrix declaration variable */
  FILE *fp;
  char name[]="system4.mtx";
  int  N,LDA,LDB,nnz;
  double *A,*B;
  char UPLO='F'; 

  /*!!!!!!!!!!!!!!!!Custom CONTOUR */  
  double *Zedge,*Zne,*Wne;
  int ccN, Nodes, *Nedge, *Tedge;
 
  /*!!!!!!!!!!!!!!!!! Others */
  int  fpm[64]; 
  int loop;
  double Emid[2];
  double r,epsout;
  int  i,j,k,n2,err;
  int  M0,M,info;
  double *E,*XR; //! eigenvalue+eigenvectors
  double *res; //! residual

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
 

  /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
  /*! Create Custom Contour          */
  ccN = 3;     //!! number of pieces that make up contour
  Zedge = calloc(2*ccN,sizeof(double));
  Tedge = calloc(ccN,sizeof(int));
  Nedge = calloc(ccN,sizeof(int));

  /*!!! Example contour - triangle   */
  Zedge[0] = 0.1e0;  Zedge[1] = 0.41e0;  // 1st complex #
  Zedge[2] = 4.2e0;  Zedge[3] = 0.41e0;
  Zedge[4] = 4.2e0;  Zedge[5] = -8.30e0;
  Tedge[0] = 0; Tedge[1] = 0; Tedge[2] = 0;
  Nedge[0] = 6; Nedge[1] = 6; Nedge[2] = 18;

  /*!! Note: user must specify total # of contour points and edit fpm[7] later*/
  Nodes=0;
  for(i=0;i<ccN;i++) Nodes = Nodes + Nedge[i];
  Zne = calloc(2*Nodes,sizeof(double)); // Contains the complex valued contour points 
  Wne = calloc(2*Nodes,sizeof(double)); // Contains the complex valued integration weights

  /* !! Fill Zne/Wne */
  zfeast_customcontour(&Nodes,&ccN,Nedge,Tedge,Zedge,Zne,Wne);
  printf("---- Printing Countour Nodes (Re+iIm) and associated Weight (Re+iIm)----\n");
  for(i=0;i<Nodes;i++)
    printf("%d %le %le %le %le\n",i,Zne[2*i],Zne[2*i+1],Wne[2*i],Wne[2*i+1]);
  /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

  M0=40; // !! M0>=M
  
  /*!!!!!!!!!!!!! ALLOCATE VARIABLE */
  E=calloc(2*M0,sizeof(double));  // eigenvalues
  XR=calloc(2*N*M0,sizeof(double));// right eigenvector // factor 2 because of complex number
  res=calloc(M0,sizeof(double));// eigenvector residual 


  /*!!!!!!!!!!!!  FEAST */  
  feastinit(fpm);
  fpm[0]=1;  /*change from default value */
  fpm[7]=Nodes;
  zfeast_syevx(&UPLO,&N,A,&N,fpm,&epsout,&loop,Emid,&r,&M0,E,XR,&M,res,&info,Zne,Wne);
  

  /*!!!!!!!!!! REPORT !!!!!!!!!*/
  printf("FEAST OUTPUT INFO %d\n",info);
  if (info!=0)  printf(" Cdense_zfeast_syevx   -- failed\n");
  if (info==0) {
    printf(" Cdense_zfeast_syevx   -- success\n");
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



