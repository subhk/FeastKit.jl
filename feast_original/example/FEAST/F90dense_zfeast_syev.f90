!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! FEAST Driver Example - Dense Storage 
!!!!!!! solving Ax=ex with A complex-symmetric (non-Hermitian)
!!!!!!! Eric Polizzi, James Kestyn 2015
!!!!!!! Eric Polizzi 2019
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
program driver 

  implicit none

!!!!!!!!!!!!!!!!! Feast declaration variable
  integer,dimension(64) :: fpm 
  integer :: loop
  character(len=1) :: UPLO='F' 

!!!!!!!!!!!!!!!!! Matrix declaration variable
  character(len=100) :: name
  integer :: n,nnz
  complex(kind=kind(1.0d0)),dimension(:,:),allocatable :: A

!!!!!!!!!!!!!!!!! Others
  integer :: i,j,k
  integer :: M0,M,info
  double precision :: r,epsout,rea,img
  complex(kind=kind(1.0d0)) :: Emid 
  complex(kind=kind(1.0d0)),dimension(:,:),allocatable :: XR ! eigenvectors
  complex(kind=kind(1.0d0)),dimension(:),allocatable :: E ! eigenvalues
  double precision,dimension(:),allocatable :: res ! residual

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!Read Coordinate format and convert to dense format
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 name='system4'

  open(10,file=trim(name)//'.mtx',status='old')
  read(10,*) n,n,nnz
  allocate(A(1:n,1:n))
  A=0.0d0
  do k=1,nnz
     read(10,*) i,j,rea,img
     A(i,j)=rea*(1.0d0,0.0d0)+img*(0.0d0,1.0d0)
  enddo
  close(10)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!! FEAST in dense format !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!! search contour
  Emid = (4.0d0,0.00d0)
  r = 3.0e0
  M0=40 !! M0>=M
  
!!!!!!!!!!!!! ALLOCATE VARIABLE 
  allocate(E(1:M0))     ! Eigenvalue
  allocate(XR(1:n,1:M0)) ! Right Eigenvectors ( XL = CONJG(XR) )
  allocate(res(1:M0))   ! Residual

!!!!!!!!!!!!  FEAST
  call feastinit(fpm)
  fpm(1)=1
  call zfeast_syev(UPLO,N,A,N,fpm,epsout,loop,Emid,r,M0,E,XR,M,res,info)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!! POST-PROCESSING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  print *,'FEAST OUTPUT INFO',info
  if (info/=0) print *,'F90dense_zfeast_syev -- failed'
  if (info==0) then
     print *,'F90dense_zfeast_syev -- success'
     print *,'*************************************************'
     print *,'************** REPORT ***************************'
     print *,'*************************************************'
     print *,'Eigenvalues/Residuals (inside interval)'
     do i=1,M
        print *,i,E(i),res(i)
     enddo
  endif



end program driver 



