!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! FEAST iExpert Driver Example - Banded Storage
!!!!!!! solving Ax=eBx with A complex-symmetric (non-Hermitian)
!!!!!!! James Kestyn, Eric Polizzi 2015
!!!!!!! Eric Polizzi 2019
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
program driver

  implicit none

!!!!!!!!!!!!!!!!! Feast declaration variable
  integer,dimension(64) :: fpm 
  integer :: loop
  character :: UPLO='F'

!!!!!!!!!!!!!!!!! Matrix declaration variable
  character(len=100) :: name
  integer :: n,nnz,kla,kua
  complex(kind=kind(1.0d0)),dimension(:,:),allocatable :: A

!!!!!!!!!!!!!!!!! Custom Contour
  integer :: ccN,Nodes
  complex(kind=kind(1.0d0)),dimension(:),allocatable :: Zne, Wne, Zedge
  integer, dimension(:), allocatable :: Nedge, Tedge 

!!!!!!!!!!!!!!!!! Others
  integer :: i,j,k
  integer :: M0,M,info
  complex(kind=kind(1.0d0)) :: Emid
  double precision :: r,epsout,rea,img
  double precision,dimension(:),allocatable :: res ! residual
  complex(kind=kind(1.0d0)),dimension(:,:),allocatable :: XR ! eigenvectors
  complex(kind=kind(1.0d0)),dimension(:),allocatable :: E ! eigenvalues


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!! read input file in coordinate format!!!!!!!
!!!!!!!!!!!!!!!!transform  in banded format directly !!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  name='system4'

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
  print *,'banded matrix -system4- size',n
  print *,'bandwidth',kla+kua+1


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!! FEAST in banded format !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   M0=40 !! M0>=M 
  
!!!!!!!!!!!!! ALLOCATE VARIABLE 
  allocate(E(1:M0))     ! Eigenvalue
  allocate(XR(1:n,1:M0)) ! Eigenvectors  ( XL = CONJG(XR) )
  allocate(res(1:M0))   ! Residual 

!!!!!!!!!!!!!  FEAST
  call feastinit(fpm)
  fpm(1)=1
  fpm(8)=Nodes
  call zfeast_sbevx(UPLO,N,kla,A,kla+kua+1,fpm,epsout,loop,Emid,r,M0,E,XR,M,res,info,Zne,Wne)
 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!! POST-PROCESSING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!! POST-PROCESSING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  print *,'FEAST OUTPUT INFO',info
  if (info/=0) print *,'F90banded_zfeast_sbevx -- failed'
  if (info==0) then
     print *,'F90banded_zfeast_sbevx -- success'
     print *,'*************************************************'
     print *,'************** REPORT ***************************'
     print *,'*************************************************'
     print *,'Eigenvalues/Residuals (inside interval)'
     do i=1,M
        print *,i,E(i),res(i)
     enddo
  endif


end program driver



