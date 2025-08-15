!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! FEAST Driver Example - Dense Storage 
!!!!!!! solving Ax=eBx with A and B real non-symmetric (non-Hermitian)
!!!!!!! James Kestyn, Eric Polizzi- 2015
!!!!!!! Eric Polizzi 2019
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
program driver

  implicit none

!!!!!!!!!!!!!!!!! Matrix declaration variable
  character(len=100) :: name
  integer :: n,nnz
  double precision,dimension(:,:),allocatable :: A,B

!!!!!!!!!!!!!!!!! Others
  integer,dimension(64) :: fpm 
  integer :: loop
  integer :: i,j,k
  integer :: M0,M,info
  complex(kind=kind(1.0d0)) :: Emid
  double precision :: r,epsout
  complex(kind=kind(1.0d0)),dimension(:,:),allocatable :: X    ! eigenvectors
  complex(kind=kind(1.0d0)),dimension(:),allocatable :: E ! eigenvalues
  double precision,dimension(:),allocatable :: res  ! residual

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!Read Coordinate format and convert to dense format
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  name='system3'

  open(10,file=trim(name)//'.mtx',status='old')
  read(10,*) n,n,nnz
  allocate(A(1:n,1:n))
  A=0.0d0
  do k=1,nnz
     read(10,*) i,j,A(i,j)
  enddo
  close(10)

  open(10,file=trim(name)//'B.mtx',status='old')
  read(10,*) n,n,nnz
  allocate(B(1:n,1:n))
  B=0.0d0
  do k=1,nnz
     read(10,*) i,j,B(i,j)
  enddo
  close(10)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!! FEAST in dense format !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! search contour
  Emid=(0.590d0,0.0d0)
  r=0.410d0
  M0=30 !! M0>=M

!!!!!!!!!!!!! ALLOCATE VARIABLE 
  allocate(E(1:M0))      ! Eigenvalue
  allocate(X(1:n,1:2*M0)) ! Eigenvectors
  allocate(res(1:2*M0))   ! Residual (if needed)

!!!!!!!!!!!!  FEAST
  call feastinit(fpm)
  fpm(1)=1
  call dfeast_gegv(N,A,N,B,N,fpm,epsout,loop,Emid,r,M0,E,X,M,res,info)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!! POST-PROCESSING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  print *,'FEAST OUTPUT INFO',info
  if (info/=0) print *,'F90dense_dfeast_gegv -- failed'
  if (info==0) then
     print *,'F90dense_dfeast_gegv -- success'
     print *,'*************************************************'
     print *,'************** REPORT ***************************'
     print *,'*************************************************'
     print *,'Eigenvalues/Residuals (inside interval)'
     do i=1,M
        print *,i,E(i),res(i)
     enddo
  endif

end program driver



