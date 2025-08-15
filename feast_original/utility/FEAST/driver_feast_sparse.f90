!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! FEAST general Driver sparse !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! solving Ax=eBx or Ax=eX      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! symmetric, hermitian or general sparse matrices !!!!!!!!!!!!!!!!!!
!!!!!!! double precision !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! where A (and B if any), provided by user in coordinate format !!!!                         
!!!!!!! by Eric Polizzi- 2009-2019  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

program driver_feast_sparse

  implicit none
!!!!!!!!!!!!!!!!! Feast declaration variable
  integer,dimension(64) :: fpm
  double precision :: depsout
  real :: sepsout
  integer :: loop

!!!!!!!!!!!!!!!!! Matrix declaration variable
  character(len=100) :: name
  character(len=1) :: UPLO,PRE,SHG,EG 
  integer :: n,nnza,nnzb
  double precision,dimension(:),allocatable :: dsa,dsb,dca,dcb
  complex(kind=kind(1.0d0)),dimension(:),allocatable :: zsa,zsb,zca,zcb
  integer,dimension(:),allocatable :: isa,jsa,isb,jsb,ica,jca,icb,jcb

!!!!!!!!!!!!!!!!! Others
  !integer :: t1,t2,tim
  integer :: i,j,k,pc
  integer :: M0,M,info
  character(len=1) :: cc

  double precision :: dEmin,dEmax,dr
  complex(kind=kind(1.0d0)):: zEmid
  double precision:: drea,dimg

  ! eigenvectors
  double precision,dimension(:,:),allocatable :: dX
  complex(kind=kind(1.0d0)),dimension(:,:),allocatable :: zX

  ! eigenvalue + residual
  double precision,dimension(:),allocatable :: dres
  real,dimension(:),allocatable :: sres
  double precision,dimension(:),allocatable :: dE!,zdummy
  real,dimension(:),allocatable :: sE
  complex,dimension(:),allocatable :: cE
  complex(kind=kind(1.0d0)),dimension(:),allocatable :: zE

  complex(kind=kind(1.0d0)),dimension(4,4) :: zAA

  complex(kind=kind(1.0d0)),dimension(4) :: zf

  integer :: code,rank,nb_procs

  integer :: itmax
  double precision :: mid,rad,re,im

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!! read main input file !!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




  call getarg(1,name)


  call feastinit(fpm)


!!!!!!!!!!!! DRIVER_FEAST_SPARSE input file  
  open(10,file=trim(name)//'.in',status='old')
  read(10,*) SHG ! type of eigenvalue problem "General, Hermitian, Symmetric" 
  read(10,*) EG ! type of eigenvalue probl  g== sparse generalized, e== sparse standard
  read(10,*) PRE  ! "PRE"==(d,z) resp. (double real, double complex) 
  read(10,*) UPLO ! UPLO==(F,L,U) reps. (Full csr, Lower csr, Upper csr) 

  if (SHG=='s') then

     if (PRE=='d') then
        read(10,*) dEmin
        read(10,*) dEmax
     elseif (PRE=='z') then
        read(10,*) drea,dimg
        zEmid=drea*(1.0d0,0.0d0)+dimg*(0.0d0,1.0d0)
        read(10,*) dr
     end if


  elseif (SHG=='h') then

     if (PRE=='z') then
        read(10,*) dEmin
        read(10,*) dEmax
     end if

  elseif (SHG=='g') then

     if ((PRE=='d').or.(PRE=='z')) then
        read(10,*) drea,dimg
        zEmid=drea*(1.0d0,0.0d0)+dimg*(0.0d0,1.0d0)
        read(10,*) dr
     end if
  end if

  read(10,*) M0   ! size subspace

  read(10,*) pc ! Some changes from default for fpm
  do i=1,pc
     read(10,*) j,fpm(j)
  enddo

  close(10)


!!!!!!!!!!!read matrix A
  open(10,file=trim(name)//'.mtx',status='old')
  k=0
  cc='%'
  do while(cc=='%')
     k=k+1 
     read(10,'(A1)') cc
  end do
  close(10)

  open(10,file=trim(name)//'.mtx',status='old')
  do i=1,k-1
     read(10,'(A1)') cc
  enddo
  read(10,*) n,n,nnza
  allocate(ica(nnza))
  allocate(jca(nnza))
  if (PRE=='d') then
     allocate(dca(nnza))
     do i=1,nnza
        read(10,*) ica(i),jca(i),dca(i)
     end do
  elseif (PRE=='z') then
     allocate(zca(nnza))
     do i=1,nnza
        read(10,*) ica(i),jca(i),drea,dimg 
        zca(i)=drea*(1.0d0,0.0d0)+dimg*(0.0d0,1.0d0)
     end do
  end if
  close(10)

  !! create csr format
  allocate(isa(1:n+1))
  allocate(jsa(1:nnza))


  if (PRE=='d') then
     allocate(dsa(1:nnza))
     call dcoo2csr(n,nnza,ica,jca,dca,isa,jsa,dsa)
  elseif (PRE=='z') then
     allocate(zsa(1:nnza))
     call zcoo2csr(n,nnza,ica,jca,zca,isa,jsa,zsa)
  end if

!!!!!!!!!!!read matrix B if any
  if (EG=='g') then

     open(10,file=trim(name)//'B.mtx',status='old')
     k=0
     cc='%'
     do while(cc=='%')
        k=k+1 
        read(10,'(A1)') cc
     end do
     close(10)

     open(10,file=trim(name)//'B.mtx',status='old')
     do i=1,k-1
        read(10,'(A1)') cc
     enddo
     read(10,*) n,n,nnzb
     allocate(icb(nnzb))
     allocate(jcb(nnzb))
     if (PRE=='d') then
        allocate(dcb(nnzb))
        do i=1,nnzb
           read(10,*) icb(i),jcb(i),dcb(i)
        end do
     elseif (PRE=='z') then
        allocate(zcb(nnzb))
        do i=1,nnzb
           read(10,*) icb(i),jcb(i),drea,dimg 
           zcb(i)=drea*(1.0d0,0.0d0)+dimg*(0.0d0,1.0d0)
        end do
     end if
     close(10)

     !! create csr format
     allocate(isb(1:n+1))
     allocate(jsb(1:nnzb))


     if (PRE=='d') then
        allocate(dsb(1:nnzb))
        call dcoo2csr(n,nnzb,icb,jcb,dcb,isb,jsb,dsb)
     elseif (PRE=='z') then
        allocate(zsb(1:nnzb))
        call zcoo2csr(n,nnzb,icb,jcb,zcb,isb,jsb,zsb)
     end if


  end if


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!! INFORMATION ABOUT MATRIX !!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  print *,'matrix name ',trim(name)
  print *,'matrix -coordinate format- size',n
  print *,'sparse matrix A- nnz',nnza
  if (EG=='g') print *,'sparse matrix B- nnz',nnzb
  print *,''


  info=-1
  !call system_clock(t1,tim)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!! FEAST  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!!!!!!!!!!!!!  FEAST SYMMETRIC

  if (SHG=='s') then 

     if ((PRE=='d').and.(EG=='g')) then 
        !print *,'Routine  ','dfeast_scsrgv'
        allocate(dE(1:M0))     ! Eigenvalue
        allocate(dres(1:M0))   ! Residual 
        allocate(dX(1:n,1:M0))

        call dfeast_scsrgv(UPLO,N,dsa,isa,jsa,dsb,isb,jsb,fpm,depsout,loop,dEmin,dEmax,M0,dE,dX,M,dres,info)


     elseif  ((PRE=='d').and.(EG=='e')) then 
        !print *,'Routine  ','dfeast_scsrev'
        allocate(dE(1:M0))     ! Eigenvalue
        allocate(dres(1:M0))   ! Residual 
        allocate(dX(1:n,1:M0))

        call dfeast_scsrev(UPLO,N,dsa,isa,jsa,fpm,depsout,loop,dEmin,dEmax,M0,dE,dX,M,dres,info)

     elseif ((PRE=='z').and.(EG=='g')) then 
        !print *,'Routine  ','zfeast_scsrgv'
        allocate(zE(1:M0))     ! Eigenvalue
        allocate(dres(1:M0))   ! Residual 
        allocate(zX(1:n,1:M0))
        call zfeast_scsrgv(UPLO,N,zsa,isa,jsa,zsb,isb,jsb,fpm,depsout,loop,zEmid,dr,M0,zE,zX,M,dres,info)

     elseif ((PRE=='z').and.(EG=='e')) then 
        !print *,'Routine  ','zfeast_scsrev'
        allocate(zE(1:M0))     ! Eigenvalue
        allocate(dres(1:M0))   ! Residual 
        allocate(zX(1:n,1:M0))
        call zfeast_scsrev(UPLO,N,zsa,isa,jsa,fpm,depsout,loop,zEmid,dr,M0,zE,zX,M,dres,info)
     end if



!!!!!!!!!!!!!  FEAST HERMITIAN

  elseif (SHG=='h') then 


     if ((PRE=='z').and.(EG=='g')) then 
        !        print *,'Routine  ','zfeast_hcsrgv'
        allocate(dE(1:M0))     ! Eigenvalue
        allocate(dres(1:M0))   ! Residual 
        allocate(zX(1:n,1:M0))
        call zfeast_hcsrgv(UPLO,N,zsa,isa,jsa,zsb,isb,jsb,fpm,depsout,loop,dEmin,dEmax,M0,dE,zX,M,dres,info)

     elseif ((PRE=='z').and.(EG=='e')) then 
        !       print *,'Routine  ','zfeast_hcsrev'
        allocate(dE(1:M0))     ! Eigenvalue
        allocate(dres(1:M0))   ! Residual 
        allocate(zX(1:n,1:M0))
        call zfeast_hcsrev(UPLO,N,zsa,isa,jsa,fpm,depsout,loop,dEmin,dEmax,M0,dE,zX,M,dres,info)
     end if



!!!!!!!!!!!!!  FEAST GENERAL

  elseif (SHG=='g') then 


     if ((PRE=='d').and.(EG=='g')) then 
        !        print *,'Routine  ','dfeast_gcsrgv'
        allocate(zE(1:M0))     ! Eigenvalue
        allocate(dres(1:2*M0))   ! Residual 
        allocate(zX(1:n,1:2*M0))
        call dfeast_gcsrgv(N,dsa,isa,jsa,dsb,isb,jsb,fpm,depsout,loop,zEmid,dr,M0,zE,zX,M,dres,info)

     elseif  ((PRE=='d').and.(EG=='e')) then 
        !        print *,'Routine  ','dfeast_gcsrev'
        allocate(zE(1:M0))     ! Eigenvalue
        allocate(dres(1:2*M0))   ! Residual 
        allocate(zX(1:n,1:2*M0))
        call dfeast_gcsrev(N,dsa,isa,jsa,fpm,depsout,loop,zEmid,dr,M0,zE,zX,M,dres,info)

     elseif ((PRE=='z').and.(EG=='g')) then 
        !       print *,'Routine  ','zfeast_gcsrgv'
        allocate(zE(1:M0))     ! Eigenvalue
        allocate(dres(1:2*M0))   ! Residual 
        allocate(zX(1:n,1:2*M0))
        call zfeast_gcsrgv(N,zsa,isa,jsa,zsb,isb,jsb,fpm,depsout,loop,zEmid,dr,M0,zE,zX,M,dres,info)

     elseif ((PRE=='z').and.(EG=='e')) then 
        !      print *,'Routine  ','zfeast_gcsrev'
        allocate(zE(1:M0))     ! Eigenvalue
        allocate(dres(1:2*M0))   ! Residual 
        allocate(zX(1:n,1:2*M0))
        call zfeast_gcsrev(N,zsa,isa,jsa,fpm,depsout,loop,zEmid,dr,M0,zE,zX,M,dres,info)

     end if

  end if


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!! POST-PROCESSING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !call system_clock(t2,tim)
  print *,'FEAST OUTPUT INFO',info
  IF ((info==0).or.(info==6)) then
     print *,'Eigenvalues saved in file: eig.out'

     open(10,file='eig.out',status='replace')   
     if (SHG/='g') then
             write(10,*)'inside interval- Eigenvalue/Residual'
        do i=1,M0
           if ((SHG=='s').and.(PRE=='z')) then
              write(10,'(I4,3ES25.16)') i,dble(zE(i)),aimag(zE(i)),dres(i)
           elseif ((SHG=='s').and.(PRE=='d')) then
              write(10,*) i,dE(i),dres(i)
           elseif ((SHG=='h').and.(PRE=='z')) then
              write(10,*) i,dE(i),dres(i)
           end if
           if (i==M) then 
              write(10,*)''
              write(10,*)'outside interval'
           endif
        enddo
     else
        write(10,*)'inside interval- Eigenvalue/Residual(right/left)'
        do i=1,M0
           if (fpm(15)==0) then ! 2 sided
              write(10,'(I4,4ES25.16)') i,dble(zE(i)),aimag(zE(i)),dres(i),dres(i+M0)
           else ! 1 sided
              write(10,*) i,zE(i),dres(i)
           endif

           if (i==M) then
              print *,''
              write(10,*) 'outside interval'
           endif
        enddo

     end if

  end IF



end program driver_feast_sparse
