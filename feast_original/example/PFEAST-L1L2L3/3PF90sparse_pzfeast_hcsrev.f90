!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! PFEAST Driver sparse example !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! solving Ax=ex with A complex Hermitian --- A sparse matrix!!
!!!!!!! Using two search intervals and 3 levels of parallelism
!!!!!!! by Eric Polizzi- 2009-2019!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
program pdriver

  implicit none

include 'mpif.h'

!!!!!!!!!!!!!!!!! Matrix declaration variable
  character(len=100) :: name
  integer :: n,nnz
  complex(kind=kind(1.0d0)),dimension(:),allocatable :: sa,c
  integer,dimension(:),allocatable :: isa,jsa,ic,jc
  character(len=1) :: UPLO='F'
!!!!!!!!!!!!!!!!! Others
  integer,dimension(64) :: fpm 
  double precision :: epsout
  integer :: loop
  integer :: i,j,k
  integer :: M0,M,info
  double precision :: Emin,Emax,rea,img
  complex(kind=kind(1.0d0)),dimension(:,:),allocatable :: X ! eigenvectors
  double precision,dimension(:),allocatable :: E,res ! eigenvalue+residual

 character(len=3) :: cnL3
 integer :: nL3
 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MPI!!!!!!!!!!!!!!!!!!!!
 integer :: code,rank,lrank,nb_procs,lnb_procs,color,key,NEW_COMM_WORLD
  call MPI_INIT(code)
  call MPI_COMM_SIZE(MPI_COMM_WORLD,nb_procs,code)
  call MPI_COMM_RANK(MPI_COMM_WORLD,rank,code)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!!!!!!!!!! READ INPUT # of procs in L3
call getarg(1,cnL3) !! number of L3 processors
read(cnL3,'(I3)') nL3

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!Read Coordinate format and convert to csr format
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  name='system2'

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



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!! Definition of the two intervals 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
if (rank<=nb_procs/2-1) then
color=1 ! first interval
else
color=2 ! second interval
endif 

!!!!!!!!!!!!!!!!! create new_mpi_comm_world
key=0
call MPI_COMM_SPLIT(MPI_COMM_WORLD,color,key,NEW_COMM_WORLD,code)
call MPI_COMM_RANK(NEW_COMM_WORLD,lrank,code) ! local rank
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!! search interval [Emin,Emax] including M eigenpairs
if (color==1) then !! 1st interval
 Emin=-0.35d0
 Emax= 0.0d0 
 M0=40
elseif(color==2) then !! 2nd interval
 Emin= 0.0d0
 Emax= 0.23d0 
 M0=40
endif


!!!!!!!!!!!!!! RUN INTERVALS in PARALLEL


!!!!!!!!!!!!! ALLOCATE VARIABLE 
  allocate(E(1:M0))     ! Eigenvalue
  allocate(X(1:N,1:M0)) ! Eigenvectors
  allocate(res(1:M0))   ! Residual 

!!!!!!!!!!!!!  FEAST
  call pfeastinit(fpm,NEW_COMM_WORLD,nL3)
  fpm(1)=-color !! print info the files feast1.log for contour1 and feast2.log for contour 2 
  call pzfeast_hcsrev(UPLO,N,sa,isa,jsa,fpm,epsout,loop,Emin,Emax,M0,E,X,M,res,info)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!! POST-PROCESSING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
  if(rank==0) print *,'FEAST OUTPUT INFO',info
  if ((info/=0).and.(rank==0)) print *,'3PF90sparse_pzfeast_hcsrev -- failed'
  if ((info==0).and.(rank==0)) then
     print *,'3PF90sparse_pzfeast_hcsrev -- success'
     print *,'*************************************************'
     print *,'************** REPORT ***************************'
     print *,'*************************************************'
  end if
  if ((info==0).and.(lrank==0)) then
     print *,'Eigenvalues/Residuals - inside the interval ',color
     do i=1,M
        print *,i,E(i),res(i)
     enddo
  endif

call MPI_FINALIZE(code)


end program pdriver



