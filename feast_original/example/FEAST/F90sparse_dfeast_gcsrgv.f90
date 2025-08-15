!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! FEAST Driver Example - CSR Storage
!!!!!!! solving Ax=eBx with A and B real non-symmetric (non-Hermitian) 
!!!!!!! James Kestyn, Eric Polizzi 2015
!!!!!!! Eric Polizzi 2019
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
program driver

  implicit none
!!!!!!!!!!!!!!!!! Feast declaration variable
  integer,dimension(64) :: fpm 
  integer :: loop
!!!!!!!!!!!!!!!!! Matrix declaration variable
  character(len=100) :: name
  integer :: n,nnz
  double precision,dimension(:),allocatable :: sa,sb,c
  integer,dimension(:),allocatable :: isa,jsa,isb,jsb,ic,jc

!!!!!!!!!!!!!!!!! Others
  integer :: i,k
  integer :: M0,M,info
  complex(kind=(kind(1.0d0))) :: Emid
  double precision :: r,epsout
  complex(kind=(kind(1.0d0))),dimension(:),allocatable :: E ! eigenvectors
  complex(kind=(kind(1.0d0))),dimension(:,:),allocatable :: X ! eigenvectors
  double precision,dimension(:),allocatable :: res ! eigenvalue+residual

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!Read Coordinate format and convert to csr format
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  name='system3'

 !!! A matrix 
  open(10,file=trim(name)//'.mtx',status='old')
  read(10,*) n,n,nnz
  allocate(ic(nnz))
  allocate(jc(nnz))
  allocate(c(nnz))
     do i=1,nnz
        read(10,*) ic(i),jc(i),c(i)
     end do
     close(10)

 allocate(isa(1:n+1))
 allocate(jsa(1:nnz))
 allocate(sa(1:nnz))
 call dcoo2csr(n,nnz,ic,jc,c,isa,jsa,sa)
 deallocate(ic,jc,c)
 
!!! B matrix 
  open(10,file=trim(name)//'B.mtx',status='old')
  read(10,*) n,n,nnz
  allocate(ic(nnz))
  allocate(jc(nnz))
  allocate(c(nnz))
     do i=1,nnz
        read(10,*) ic(i),jc(i),c(i)
     end do
     close(10)

 allocate(isb(1:n+1))
 allocate(jsb(1:nnz))
 allocate(sb(1:nnz))
 call dcoo2csr(n,nnz,ic,jc,c,isb,jsb,sb)
 deallocate(ic,jc,c)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!! FEAST in sparse format !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!! search interval [Emin,Emax] including M eigenpairs
  Emid=(0.59d0,0.0d0)
  r=0.4100d0 
  M0=30 !! M0>=M

!!!!!!!!!!!!! ALLOCATE VARIABLE 
  allocate(E(1:M0))     ! Eigenvalue
  allocate(X(1:N,1:2*M0)) ! Eigenvectors
  allocate(res(1:2*M0))   ! Residual 

!!!!!!!!!!!!!  FEAST
  call feastinit(fpm)
  fpm(1)=1
  call dfeast_gcsrgv(N,sa,isa,jsa,sb,isb,jsb,fpm,epsout,loop,Emid,r,M0,E,X,M,res,info)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!! POST-PROCESSING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  print *,'FEAST OUTPUT INFO',info
  if (info/=0) print *,'F90sparse_dfeast_gcsrgv -- failed'
  if (info==0) then
     print *,'F90sparse_dfeast_gcsrgv -- success'
     print *,'*************************************************'
     print *,'************** REPORT ***************************'
     print *,'*************************************************'
     print *,'Eigenvalues/Residuals (inside interval)'
     do i=1,M
        print *,i,E(i),res(i)
     enddo
  endif

end program driver



