!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! FEAST Expert Driver Example - Dense Storage 
!!!!!!! solving Ax=ex with A complex-symmetric (non-Hermitian_
!!!!!!! Eric Polizzi, James Kestyn 2015
!!!!!!! Eric Polizzi 2019
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
program driver 

  implicit none

!!!!!!!!!!!!!!!!! Matrix declaration variable
  character(len=100) :: name
  integer :: n,nnz
  complex(kind=kind(1.0d0)),dimension(:,:),allocatable :: A
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
  double precision :: r, epsout, rea,img
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


  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Create Custom Contour
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   M0=40 !! M0>=M 
  
!!!!!!!!!!!!! ALLOCATE VARIABLE 
  allocate(E(1:M0))     ! Eigenvalue
  allocate(XR(1:N,1:M0)) ! Right Eigenvectors ( XL = CONJG(XR) )
  allocate(res(1:M0))   ! Residual 

!!!!!!!!!!!!  FEAST

  call feastinit(fpm)
  fpm(1)=1
  fpm(8)=Nodes  
  call zfeast_syevx(UPLO,N,A,N,fpm,epsout,loop,Emid,r,M0,E,XR,M,res,info,Zne,Wne)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!! POST-PROCESSING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  print *,'FEAST OUTPUT INFO',info
  if (info/=0) print *,'F90dense_zfeast_syevx -- failed'
  if (info==0) then
     print *,'F90dense_zfeast_syevx -- success'
     print *,'*************************************************'
     print *,'************** REPORT ***************************'
     print *,'*************************************************'
     print *,'Eigenvalues/Residuals (inside interval)'
     do i=1,M
        print *,i,E(i),res(i)
     enddo
  endif



end program driver 



