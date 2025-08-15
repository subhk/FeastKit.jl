/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!! FEAST Expert Driver Example - CSR Storage
  !!!!!!! solving Ax=ex with A complex-symmetric (non-Hermitian)
  !!!!!!! James Kestyn, Eric Polizzi 2015
  !!!!!!! Eric Polizzi
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>

#include "feast.h"
#include "feast_sparse.h"
int main() {
  /*!!!!!!!!!!!!!!!!! Feast declaration variable */
 

  /*!!!!!!!!!!!!!!!!! Matrix declaration variable */
  FILE *fp;
  char name[]="system4.mtx";
  int  N,nnz;
  double *sa;
  int *isa,*jsa;
  char UPLO='F';
  /*!!!!!!!!!!!!!!!!! Contour */
  double *Zedge,*Zne,*Wne;
  int ccN, Nodes, *Nedge, *Tedge;
 /*!!!!!!!!!!!!!!!!! Others */
   int  fpm[64]; 
  double epsout;
  int loop; 
  int  i,k,err;
  int  M0,M,info;
  double Emid[2],r;
  double *X; //! eigenvectors
  double *E,*res; //! eigenvalue+residual


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


  /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
  /*! Create Custom Contour          */
  /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
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
  E=calloc(2*M0,sizeof(double));  // eigenvalues  // factor 2 fopr complex
  res=calloc(M0,sizeof(double));// eigenvectors 
  X=calloc(2*N*M0,sizeof(double));// residual (if needed) // factor 2 for complex
  
  /*!!!!!!!!!!!!  FEAST */
  feastinit(fpm);
  fpm[0]=1;  /*change from default value */
  fpm[7]=Nodes;
  zfeast_scsrevx(&UPLO,&N,sa,isa,jsa,fpm,&epsout,&loop,Emid,&r,&M0,E,X,&M,res,&info,Zne,Wne);

  /*!!!!!!!!!! REPORT !!!!!!!!!*/
  printf("FEAST OUTPUT INFO %d\n",info);
  if (info!=0)  printf(" Csparse_zfeast_scsrevx  -- failed\n");
  if (info==0) {
    printf(" Csparse_zfeast_scsrevx  -- success\n");
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





