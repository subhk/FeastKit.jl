!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! FEAST Driver sparse example !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! solving quadratic eigenvalue problem (A2e^2+A1e^1+A0)x=0 with A2,A1,A0 real symmetric and sparse matrix!!!!
!!!!!!! by Eric Polizzi- 2019       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
program driver

  implicit none

!!!!!!!!!!!!!!!!! Matrix declaration variable
  character(len=100) :: name
  integer :: n,nnz
  double precision,dimension(:,:),allocatable :: sa
  integer,dimension(:,:),allocatable :: isa,jsa
  double precision,dimension(:),allocatable :: c
  integer,dimension(:),allocatable :: ic,jc
  character(len=1) :: UPLO='F' 
!!!!!!!!!!!!!!!!! Others
  integer,dimension(64) :: fpm 
  double precision :: epsout
  integer :: loop
  integer :: i
  integer :: M0,M,info
  double precision :: r
  complex(kind=kind(1.0d0)) :: Emid
  complex(kind=kind(1.0d0)),dimension(:),allocatable :: E  ! eigenvalues
  complex(kind=kind(1.0d0)),dimension(:,:),allocatable :: X ! eigenvectors
  double precision,dimension(:),allocatable :: res ! eigenvalue

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!Create Matrices
!!!!!!!!! We consider the nonoverdamped mass-spring system
!!!!!!!!! Example 1 in
!!!Brendan Gavin, Agnieszka Międlar, Eric Polizzi,
!!!FEAST eigensolver for nonlinear eigenvalue problems,
!!!!Journal of Computational Science,Volume 27, 2018,Pages 107-117
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! A[2]=I;  A[1]=tau*tridiag[-1,3,-1]nxn; A[0]=kappa*tridiag[-1,3,-1]nxn
!!tau=0.6202d0 kappa=0.4807d0

name='system5'
  
N=1000
!! create 3 sparse csr matrices
nnz=3*N-2 ! should be the nnz max that we apply to all matrices (know a priori here)
allocate(sa(nnz,3))
allocate(jsa(nnz,3))
allocate(isa(N+1,3))

 open(10,file=trim(name)//'A0.mtx',status='old')
  read(10,*) n,n,nnz
  allocate(ic(nnz),jc(nnz),c(nnz))
     do i=1,nnz
        read(10,*) ic(i),jc(i),c(i)
     end do
     close(10)
 call dcoo2csr(n,nnz,ic,jc,c,isa(1,1),jsa(1,1),sa(1,1))
 deallocate(ic,jc,c)
 
 open(10,file=trim(name)//'A1.mtx',status='old')
  read(10,*) n,n,nnz
  allocate(ic(nnz),jc(nnz),c(nnz))
     do i=1,nnz
        read(10,*) ic(i),jc(i),c(i)
     end do
     close(10)
 call dcoo2csr(n,nnz,ic,jc,c,isa(1,2),jsa(1,2),sa(1,2))
 deallocate(ic,jc,c)
 
 
 open(10,file=trim(name)//'A2.mtx',status='old')
  read(10,*) n,n,nnz
  allocate(ic(nnz),jc(nnz),c(nnz))
     do i=1,nnz
        read(10,*) ic(i),jc(i),c(i)
     end do
     close(10)
 call dcoo2csr(n,nnz,ic,jc,c,isa(1,3),jsa(1,3),sa(1,3))
 deallocate(ic,jc,c)
 

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!! FEAST in sparse format !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! define search contour including M eigenpairs
Emid=(-1.55d0,0.0d0)
r=0.05d0
M0=30 !! M0>=M

!!!!!!!!!!!!! ALLOCATE VARIABLE 
  allocate(E(1:M0))       ! Eigenvalue
  allocate(X(1:N,1:M0)) ! Eigenvectors
  allocate(res(1:M0))   ! Residual

!!!!!!!!!!!!  FEAST
  call feastinit(fpm)
  fpm(1)=1 ! change from default (printing info on screen)
  fpm(18)=100*(0.0035/r) ! ellipse
  call dfeast_scsrpev(UPLO,2,N,sa,isa,jsa,fpm,epsout,loop,Emid,r,M0,E,X,M,res,info)
  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!! POST-PROCESSING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  print *,'FEAST OUTPUT INFO',info
  if (info/=0) print *,'F90sparse_dfeast_scsrpev-- failed'
  if (info==0) then
     print *,'F90sparse_dfeast_scsrpev-- success'
     print *,'*************************************************'
     print *,'************** REPORT ***************************'
     print *,'*************************************************'
     print *,'Eigenvalues/Residuals (inside interval)'
     do i=1,M
        print *,i,E(i),res(i)
     enddo
  endif


end program driver



