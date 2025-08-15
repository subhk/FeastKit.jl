!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! FEAST Driver dense example !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! solving Ax=ex with A complex Hermitian --- A dense matrix!!!!!!!!
!!!!!!! by Eric Polizzi- 2009-2019!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
program driver 

  implicit none

!!!!!!!!!!!!!!!!! Feast declaration variable
  integer,dimension(64) :: fpm 
  double precision :: epsout
  integer :: loop
  character(len=1) :: UPLO='F' ! 'L' or 'U' also fine here

!!!!!!!!!!!!!!!!! Matrix declaration variable
  character(len=100) :: name
  integer :: n,nnz
  complex(kind=kind(1.0d0)),dimension(:,:),allocatable :: A

!!!!!!!!!!!!!!!!! Others
  integer :: i,j,k
  integer :: M0,M,info
  double precision :: Emin,Emax,rea,img
  complex(kind=kind(1.0d0)),dimension(:,:),allocatable :: X ! eigenvectors
  double precision,dimension(:),allocatable :: E,res ! eigenvalue+residual


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!Read Coordinate format and convert to dense format
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  name='system2'


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

!!! search interval [Emin,Emax] including M eigenpairs
  Emin=-0.35d0
  Emax= 0.23d0 
  M0=40 !! M0>=M


!!!!!!!!!!!!! ALLOCATE VARIABLE 
  allocate(E(1:M0))     ! Eigenvalue
  allocate(X(1:n,1:M0)) ! Eigenvectors
  allocate(res(1:M0))   ! Residual 

!!!!!!!!!!!!  FEAST
  call feastinit(fpm)
  fpm(1)=1  ! change from default (printing info on screen)
  call zfeast_heev(UPLO,N,A,N,fpm,epsout,loop,Emin,Emax,M0,E,X,M,res,info)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!! POST-PROCESSING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  print *,'FEAST OUTPUT INFO',info
  if (info/=0) print *,'F90dense_zfeast_heev -- failed'
  if (info==0) then
     print *,'F90dense_zfeast_heev -- success'
     print *,'*************************************************'
     print *,'************** REPORT ***************************'
     print *,'*************************************************'
     print *,'Eigenvalues/Residuals (inside interval)'
     do i=1,M
        print *,i,E(i),res(i)
     enddo
  endif



end program driver 



