!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! PFEAST general Driver sparse !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! solving Ax=eBx or Ax=eX      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! symmetric, hermitian or general sparse matrices !!!!!!!!!!!!!!!!!!
!!!!!!! single or double precision !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! where A (and B if any), provided by user in coordinate format !!!!                         
!!!!!!! by Eric Polizzi- 2012-2019       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

program driver_pfeast_sparse

  implicit none
  include "mpif.h"

!!!!!!!!!!!!!!!!! Feast declaration variable
  integer,dimension(64) :: fpm
  double precision :: depsout
  real :: sepsout
  integer :: loop

  character(len=100) :: name
  character(len=1) :: UPLO,PRE,SHG,EG 
  integer :: n,nnza,nnzb
  double precision,dimension(:),allocatable :: dsa,dsb,dca,dcb
  complex(kind=kind(1.0d0)),dimension(:),allocatable :: zsa,zsb,zca,zcb
  integer,dimension(:),allocatable :: isa,jsa,isb,jsb,ica,jca,icb,jcb


  !double precision :: t1,t2
  integer :: i,j,k,pc
  integer :: M0,M,info
  character(len=1) :: cc


  double precision :: dEmin,dEmax,dr
  complex(kind=kind(1.0d0)):: zEmid
  double precision:: drea,dimg

  double precision,dimension(:,:),allocatable :: dX
  complex(kind=kind(1.0d0)),dimension(:,:),allocatable :: zX

  double precision,dimension(:),allocatable :: dres

  double precision,dimension(:),allocatable :: dE
  complex(kind=kind(1.0d0)),dimension(:),allocatable :: zE
  integer :: code,rank,nb_procs
  character(len=3) :: cnL3
  integer :: nL3
  integer :: nb_procs3
  integer,dimension(:),allocatable :: Nsize
  integer :: startj,endj,Nlocal
  !!  For local partitionning examples
  type dcsr
     integer ::n,m,nnz
     double precision,dimension(:),allocatable :: sa
     integer,dimension(:),allocatable :: isa,jsa     
  end type dcsr
  type(dcsr) :: matAj,matBj
  type zcsr
     integer ::n,m,nnz
     complex(kind=kind(1.0d0)),dimension(:),allocatable :: sa
     integer,dimension(:),allocatable :: isa,jsa     
  end type zcsr
  type(zcsr) :: zmatAj,zmatBj
!!!!!!!!!!!!!


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MPI!!!!!!!!!!!!!!!!!!!!
  call MPI_INIT(code)
  call MPI_COMM_SIZE(MPI_COMM_WORLD,nb_procs,code)
  call MPI_COMM_RANK(MPI_COMM_WORLD,rank,code)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!! read main input file !!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  call getarg(1,name)
  call getarg(2,cnL3) !! number of L3 processors

  read(cnL3,'(I3)') nL3




  call pfeastinit(fpm,MPI_COMM_WORLD,nL3)



!!!!!!!!!!!! DRIVER_FEAST_SPARSE input file  
  open(10,file=trim(name)//'.in',status='old')
  read(10,*) SHG ! type of eigenvalue problem "General, Hermitian, Symmetric" 
  read(10,*) EG ! type of eigenvalue probl  g== sparse generalized, e== sparse standard
  read(10,*) PRE  ! "PRE"==(s,d,c,z) resp. (single real,double real,complex,double complex) 
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
  if (rank==0) then
     print *,'matrix name ',trim(name)
     print *,'matrix -coordinate format- size',n
     print *,'sparse matrix A- nnz',nnza
     if (EG=='g') print *,'sparse matrix B- nnz',nnzb
     print *,''
  endif

  info=-1
  !t1=MPI_WTIME()
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!! FEAST  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!!!!!!!!!!!!!  FEAST SYMMETRIC

  if (SHG=='s') then 

     if ((PRE=='d').and.(EG=='g')) then 
        !        if (rank==0) print *,'Routine  ','dfeast_scsrgv'
        allocate(dE(1:M0))     ! Eigenvalue
        allocate(dres(1:M0))   ! Residual

!!! global partitionning 
        allocate(dX(1:n,1:M0))  
        call pdfeast_scsrgv(UPLO,N,dsa,isa,jsa,dsb,isb,jsb,fpm,depsout,loop,dEmin,dEmax,M0,dE,dX,M,dres,info)
        !! example with local partitionning
        !call MPI_COMM_SIZE(fpm(49),nb_procs3,code)
        !allocate(Nsize(nb_procs3))
        !call dcsr_distribute_row(N,dsa,isa,jsa,matAj,startj,endj,Nlocal,Nsize,fpm(49)) !! A
        !call dcsr_distribute_row(N,dsb,isb,jsb,matBj,startj,endj,Nlocal,Nsize,fpm(49)) !! B
        !allocate(dX(1:Nlocal,1:M0))
        !call pdfeast_scsrgv(UPLO,Nlocal,matAj%sa,matAj%isa,matAj%jsa,matBj%sa,matBj%isa,matBj%jsa,fpm,depsout,loop,dEmin,dEmax,M0,dE,dX,M,dres,info)


     elseif  ((PRE=='d').and.(EG=='e')) then 
        !        if (rank==0) print *,'Routine  ','dfeast_scsrev'
        allocate(dE(1:M0))     ! Eigenvalue
        allocate(dres(1:M0))   ! Residual
        !! global partitionning        
        allocate(dX(1:n,1:M0))
        call pdfeast_scsrev(UPLO,N,dsa,isa,jsa,fpm,depsout,loop,dEmin,dEmax,M0,dE,dX,M,dres,info)
          
        !! example with local partitionning
!!$call MPI_COMM_SIZE(fpm(49),nb_procs3,code)
!!$allocate(Nsize(nb_procs3))
!!$call dcsr_distribute_row(N,dsa,isa,jsa,matAj,startj,endj,Nlocal,Nsize,fpm(49)) !! A
!!$allocate(dX(1:Nlocal,1:M0))
!!$call pdfeast_scsrev(UPLO,Nlocal,matAj%sa,matAj%isa,matAj%jsa,fpm,depsout,loop,dEmin,dEmax,M0,dE,dX,M,dres,info)

     elseif ((PRE=='z').and.(EG=='g')) then 
        !        if (rank==0) print *,'Routine  ','zfeast_scsrgv'
        allocate(zE(1:M0))     ! Eigenvalue
        allocate(dres(1:M0))   ! Residual 
        allocate(zX(1:n,1:M0))
        call pzfeast_scsrgv(UPLO,N,zsa,isa,jsa,zsb,isb,jsb,fpm,depsout,loop,zEmid,dr,M0,zE,zX,M,dres,info)

     elseif ((PRE=='z').and.(EG=='e')) then 
        !        if (rank==0) print *,'Routine  ','zfeast_scsrev'
        allocate(zE(1:M0))     ! Eigenvalue
        allocate(dres(1:M0))   ! Residual 
        allocate(zX(1:n,1:M0))
        call pzfeast_scsrev(UPLO,N,zsa,isa,jsa,fpm,depsout,loop,zEmid,dr,M0,zE,zX,M,dres,info)

     end if



!!!!!!!!!!!!!  FEAST HERMITIAN

  elseif (SHG=='h') then 


     if ((PRE=='z').and.(EG=='g')) then 
        !  if (rank==0) print *,'Routine  ','zfeast_hcsrgv'
        allocate(dE(1:M0))     ! Eigenvalue
        allocate(dres(1:M0))   ! Residual 
        allocate(zX(1:n,1:M0))
        call pzfeast_hcsrgv(UPLO,N,zsa,isa,jsa,zsb,isb,jsb,fpm,depsout,loop,dEmin,dEmax,M0,dE,zX,M,dres,info)

     elseif ((PRE=='z').and.(EG=='e')) then 
        ! if (rank==0) print *,'Routine  ','zfeast_hcsrev'
        allocate(dE(1:M0))     ! Eigenvalue
        allocate(dres(1:M0))   ! Residual
        !! global partitionning
        allocate(zX(1:n,1:M0))
        call pzfeast_hcsrev(UPLO,N,zsa,isa,jsa,fpm,depsout,loop,dEmin,dEmax,M0,dE,zX,M,dres,info)

        !!  example with local partionning 
!!$call MPI_COMM_SIZE(fpm(49),nb_procs3,code)
!!$allocate(Nsize(nb_procs3))
!!$call zcsr_distribute_row(N,zsa,isa,jsa,zmatAj,startj,endj,Nlocal,Nsize,fpm(49)) !! A
!!$!
!!$allocate(zX(1:Nlocal,1:M0))
!!$call pzfeast_hcsrev(UPLO,Nlocal,zmatAj%sa,zmatAj%isa,zmatAj%jsa,fpm,depsout,loop,dEmin,dEmax,M0,dE,zX,M,dres,info)

     end if



!!!!!!!!!!!!!  FEAST GENERAL

  elseif (SHG=='g') then 


     if ((PRE=='d').and.(EG=='g')) then 
        if (rank==0) print *,'Routine  ','dfeast_gcsrgv'
        allocate(zE(1:M0))     ! Eigenvalue
        allocate(dres(1:2*M0))   ! Residual

        !! global partitionning
        allocate(zX(1:n,1:2*M0))
        call pdfeast_gcsrgv(N,dsa,isa,jsa,dsb,isb,jsb,fpm,depsout,loop,zEmid,dr,M0,zE,zX,M,dres,info)

        !! example witj local partitioning
!!$call MPI_COMM_SIZE(fpm(49),nb_procs3,code)
!!$allocate(Nsize(nb_procs3))
!!$call dcsr_distribute_row(N,dsa,isa,jsa,matAj,startj,endj,Nlocal,Nsize,fpm(49)) !! A
!!$call dcsr_distribute_row(N,dsb,isb,jsb,matBj,startj,endj,Nlocal,Nsize,fpm(49)) !! B
!!$allocate(zX(1:Nlocal,1:2*M0))
!!$call pdfeast_gcsrgv(Nlocal,matAj%sa,matAj%isa,matAj%jsa,matBj%sa,matBj%isa,matBj%jsa,fpm,depsout,loop,zEmid,dr,M0,zE,zX,M,dres,info)

     elseif  ((PRE=='d').and.(EG=='e')) then 
        ! if (rank==0) print *,'Routine  ','dfeast_gcsrev'
        allocate(zE(1:M0))     ! Eigenvalue
        allocate(dres(1:2*M0))   ! Residual 
        allocate(zX(1:n,1:2*M0))
        call pdfeast_gcsrev(N,dsa,isa,jsa,fpm,depsout,loop,zEmid,dr,M0,zE,zX,M,dres,info)


     elseif ((PRE=='z').and.(EG=='g')) then 
        !     if (rank==0) print *,'Routine  ','zfeast_gcsrgv'
        allocate(zE(1:M0))     ! Eigenvalue
        allocate(dres(1:2*M0))   ! Residual 
        allocate(zX(1:n,1:2*M0))
        call pzfeast_gcsrgv(N,zsa,isa,jsa,zsb,isb,jsb,fpm,depsout,loop,zEmid,dr,M0,zE,zX,M,dres,info)

     elseif ((PRE=='z').and.(EG=='e')) then 
        !    if (rank==0) print *,'Routine  ','zfeast_gcsrev'
        allocate(zE(1:M0))     ! Eigenvalue
        allocate(dres(1:2*M0))   ! Residual 
        allocate(zX(1:n,1:2*M0))
        call pzfeast_gcsrev(N,zsa,isa,jsa,fpm,depsout,loop,zEmid,dr,M0,zE,zX,M,dres,info)

     end if

  end if


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!! POST-PROCESSING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  t2=MPI_WTIME()

  if (rank==0) then
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
  end if

     call MPI_FINALIZE(code)


   end program driver_pfeast_sparse


