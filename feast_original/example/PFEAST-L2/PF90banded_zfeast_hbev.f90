!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! PFEAST Driver banded example !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! solving Ax=ex with A complex Hermitian --- A banded matrix!!!!!!!
!!!!!!! by Eric Polizzi- 2009-2019
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
program driver

  implicit none
  
 include 'mpif.h'
  
!!!!!!!!!!!!!!!!! Feast declaration variable
  integer,dimension(64) :: fpm 
  double precision :: epsout
  integer :: loop
  character(len=1) :: UPLO='F'

!!!!!!!!!!!!!!!!! Matrix declaration variable
  character(len=100) :: name
  integer :: n,nnz,kla,kua
  complex(kind=kind(1.0d0)),dimension(:,:),allocatable :: A

!!!!!!!!!!!!!!!!! Others
  integer :: i,j,k
  integer :: M0,M,info
  double precision :: Emin,Emax,rea,img
  complex(kind=kind(1.0d0)),dimension(:,:),allocatable :: X ! eigenvectors
  double precision,dimension(:),allocatable :: E,res ! eigenvalue+residual

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MPI!!!!!!!!!!!!!!!!!!!!
integer :: code,rank,nb_procs
  call MPI_INIT(code)
  !call MPI_COMM_SIZE(MPI_COMM_WORLD,nb_procs,code)
  call MPI_COMM_RANK(MPI_COMM_WORLD,rank,code)

  
  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!! read input file in coordinate format!!!!!!!
!!!!!!!!!!!!!!!!transform  in banded format directly !!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  name='system2'

  !! find kla,kua
  kla=0
  kua=0
  open(10,file=trim(name)//'.mtx',status='old')
  read(10,*) n,n,nnz
  do k=1,nnz
     read(10,*) i,j
     if (abs(i-j)>kla) kla=abs(i-j)
     if (abs(j-i)>kua) kua=abs(j-i)    
  end do
  close(10)

  !! form the banded matrix
  allocate(A(1+kla+kua,N))
  A=(0.0d0,0.0d0)
  open(10,file=trim(name)//'.mtx',status='old')
  read(10,*) n,n,nnz
  do k=1,nnz
     read(10,*) i,j,rea,img
  A(kua+1+(i-j),j)=rea*(1.0d0,0.0d0)+img*(0.0d0,1.0d0)
  end do
  close(10)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!! INFORMATION ABOUT MATRIX !!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  print *,'banded matrix -system2- size',n
  print *,'bandwidth',kla+kua+1


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!! FEAST in banded format !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! search interval [Emin,Emax] including M eigenpairs
  Emin=-0.35d0
  Emax= 0.23d0 
  M0=40 !! M0>=M

!!!!!!!!!!!!! ALLOCATE VARIABLE 
  allocate(e(1:M0))     ! Eigenvalue
  allocate(X(1:n,1:M0)) ! Eigenvectors
  allocate(res(1:M0))   ! Residual 
!!!!!!!!!!!!!  FEAST
  call feastinit(fpm)
  fpm(1)=1
  call zfeast_hbev(UPLO,N,kla,A,kla+kua+1,fpm,epsout,loop,Emin,Emax,M0,E,X,M,res,info)
  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!! POST-PROCESSING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  if (rank==0)  print *,'FEAST OUTPUT INFO',info
  if ((info/=0).and.(rank==0))  print *,'PF90banded_zfeast_hbev -- failed'
  if ((info==0).and.(rank==0))  then
     print *,'PF90banded_zfeast_hbev -- success'
     print *,'*************************************************'
     print *,'************** REPORT ***************************'
     print *,'*************************************************'
     print *,'Eigenvalues/Residuals (inside interval)'
     do i=1,M
        print *,i,E(i),res(i)
     enddo
  endif

call MPI_FINALIZE(code)
end program driver



