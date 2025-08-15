!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! FEAST Driver sparse example 
!!!!!!! solving Ax=ex with A complex-symmetric (non-Hermitian) - A CSR Format
!!!!!!! James Kestyn, Eric Polizzi 2015
!!!!!!! Eric Polizzi 2019
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
program driver

  implicit none

!!!!!!!!!!!!!!!!! Matrix declaration variable
  character(len=100) :: name
  integer :: n,nnz
  complex(kind=kind(1.0d0)),dimension(:),allocatable :: sa,c
  integer,dimension(:),allocatable :: isa,jsa,ic,jc
  character(len=1) :: UPLO='F'
!!!!!!!!!!!!!!!!! Contour
  integer :: ccN,Nodes
  complex(kind=kind(1.0d0)),dimension(:),allocatable :: Zne, Wne, Zedge
  integer, dimension(:), allocatable :: Nedge, Tedge 

!!!!!!!!!!!!!!!!! Others
  integer,dimension(64) :: fpm 
  integer :: loop
  integer :: i,j,k
  integer :: M0,M,info
  complex(kind=kind(1.0d0)) :: Emid
  double precision :: r,epsout,rea,img
  complex(kind=(kind(1.0d0))),dimension(:),allocatable :: E ! eigenvalue+residual
  complex(kind=kind(1.0d0)),dimension(:,:),allocatable :: XR ! eigenvectors
  double precision,dimension(:),allocatable :: res ! eigenvalue+residual

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


  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Create Custom Contour
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ccN = 3     !! number of pieces that make up contour
  allocate(Zedge(1:ccN))
  allocate(Nedge(1:ccN))
  allocate(Tedge(1:ccN))
  !!! Example contour - triangle  
  Zedge = (/(0.10d0,0.410d0),(4.2d0,0.41d0),(4.2d0,-8.3d0)/)
  Tedge(:) = (/0,0,0/)
  Nedge(:) = (/6,6,18/)
  !! Note: user must specify total # of contour points and edit fpm(8) later
  Nodes = sum(Nedge(1:ccN))
  allocate(Zne(1:Nodes)) !! Contains the complex valued contour points 
  allocate(Wne(1:Nodes)) !! Contains the complex valued integrations weights

  !! Fill Zne/Wne
  call zfeast_customcontour(Nodes,ccN,Nedge,Tedge,Zedge,Zne,Wne)
  print *,'---- Printing Countour Nodes (Re+iIm) and associated Weight (Re+iIm)----'
  do i=1,Nodes
     write(*,'(I3,4ES24.16)') i,dble(Zne(i)),aimag(Zne(i)),dble(Wne(i)),aimag(Wne(i))
  enddo
 
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 M0=40 !! M0>=M


 
!!!!!!!!!!!!! ALLOCATE VARIABLE 
  allocate(E(1:M0))     ! Eigenvalue
  allocate(XR(1:N,1:M0)) ! Eigenvectors
  allocate(res(1:M0))   ! Residual 

!!!!!!!!!!! FEAST
  
!!! custom contour
  call feastinit(fpm)
  fpm(1)=1
  fpm(8)=Nodes
  fpm(42)=0 !solver in double precision (single precision does not seem to work here)
  call zfeast_scsrevx(UPLO,N,sa,isa,jsa,fpm,epsout,loop,Emid,r,M0,E,XR,M,res,info,Zne,Wne)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!! POST-PROCESSING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   print *,'FEAST OUTPUT INFO',info
  if (info/=0) print *,'F90sparse_zfeast_scsrevx -- failed'
  if (info==0) then
     print *,'F90sparse_zfeast_scsrevx -- success'
     print *,'*************************************************'
     print *,'************** REPORT ***************************'
     print *,'*************************************************'
     print *,'Eigenvalues/Residuals (inside interval)'
     do i=1,M
        print *,i,E(i),res(i)
     enddo
  endif


end program driver



