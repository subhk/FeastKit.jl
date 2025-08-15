!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! PFEAST Driver sparse example - CSR Format 
!!!!!!! solving Ax=ex with A complex-symmetric (non-Hermitian)
!!!!!!! James Kestyn, Eric Polizzi 2015
!!!!!!! Eric Polizzi 2019
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
program driver

  implicit none
  
  include 'mpif.h'
  
!!!!!!!!!!!!!!!!! Matrix declaration variable
  character(len=100) :: name
  integer :: n,nnz
  complex(kind=kind(1.0d0)),dimension(:),allocatable :: sa,c
  integer,dimension(:),allocatable :: isa,jsa,ic,jc
  character(len=1) :: UPLO='F'
  
!!!!!!!!!!!! Others
  integer,dimension(64) :: fpm 
  integer :: loop
  integer :: i,j,k
  integer :: M0,M,info
  complex(kind=kind(1.0d0)) :: Emid
  double precision :: r,epsout,rea,img
  complex(kind=(kind(1.0d0))),dimension(:),allocatable :: E ! eigenvalue+residual
  complex(kind=kind(1.0d0)),dimension(:,:),allocatable :: XR ! eigenvectors
  double precision,dimension(:),allocatable :: res ! eigenvalue+residual

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MPI!!!!!!!!!!!!!!!!!!!!
integer :: code,rank,nb_procs
  call MPI_INIT(code)
  !call MPI_COMM_SIZE(MPI_COMM_WORLD,nb_procs,code)
  call MPI_COMM_RANK(MPI_COMM_WORLD,rank,code)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!Read Coordinate format and convert to csr format
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  name='system4'

 !!! A matrix 
  open(10,file=trim(name)//'.mtx',status='old')
  read(10,*) n,n,nnz
  allocate(ic(nnz))
  allocate(jc(nnz))
  allocate(c(nnz))
     do i=1,nnz
        read(10,*) ic(i),jc(i),rea,img
        c(i)=rea*(1.0d0,0.0d0)+img*(0.0d0,1.0d0)
     end do
     close(10)

 allocate(isa(1:n+1))
 allocate(jsa(1:nnz))
 allocate(sa(1:nnz))
 call zcoo2csr(n,nnz,ic,jc,c,isa,jsa,sa)
 deallocate(ic,jc,c)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!! FEAST in sparse format !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!! search contour
  Emid = (4.0d0,0.0d0)
  r = 3.0d0
  M0=40 !! M0>=M

!!!!!!!!!!!!! ALLOCATE VARIABLE 
  allocate(E(1:M0))     ! Eigenvalue
  allocate(XR(1:N,1:M0)) ! Eigenvectors
  allocate(res(1:M0))   ! Residual 

!!!!!!!!!!!!! FEAST  
call feastinit(fpm)
  fpm(1)=1  
  call zfeast_scsrev(UPLO,N,sa,isa,jsa,fpm,epsout,loop,Emid,r,M0,E,XR,M,res,info)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!! POST-PROCESSING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  if (rank==0) print *,'FEAST OUTPUT INFO',info
  if ((info/=0).and.(rank==0)) print *,'PF90sparse_zfeast_scsrev -- failed'
  if ((info==0).and.(rank==0)) then
     print *,'PF90sparse_zfeast_scsrev -- success'
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



