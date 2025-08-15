COMP?=intel

MPI?=impi

N?=2

NL3?=2

help:
	@echo "=========================================================================================================="
	@echo "This Makefile provides a complete testing of few selected SPARSE interfaces in Fortran & C"
	@echo "It makes use of Intel-MKL for BLAS/LAPACK and cluster-PARDISO (if PFEAST was compiled with MKL flag on)"
	@echo
	@echo "Usage: make COMP=<comp> MPI=<mpi> N=<n> NL3=<nl3> {allF rallF allC rallC all rall}"
	@echo
	@echo "		<comp> is used to select your compilers (possible choices: intel, gnu, pgi). Defaults to 'intel'"
	@echo "			this should be the same Fortran compiler used for compiling PFEAST"
	@echo "		    -'intel' selects ifort for F90 examples and icc for C examples"
	@echo "		    -'gnu'   selects gfortran for F90 examples and gcc for C examples"
	@echo "		    -'pgi'   selects pgf90 for F90 examples and pgcc for C examples"
	@echo "		<mpi>  is used to select your MPI directives (possible choices: impi, mpich, ompi). Defaults to 'impi'"
	@echo "			this should be the same MPI used for compiling PFEAST"
	@echo "		    -'impi'   selects intel MPI using mpiifort, mpicc and mpirun"
	@echo "		    -'mpich'  selects MPICH using mpif90.mpich, mpicc.mpich, and mpirun.mpich"
	@echo "		    -'openmpi'selects OPENMPI using mpif90.openmpi, mpicc.openmpi and mpirun.openmpi"
	@echo "		<n>    is the total number of MPI processors used for running the test. Defaults to 2"
	@echo "		<nl3>  is the number of MPI processors reserved for the MPI solvers (L3 parallelism). Defaults to 2"
	@echo "		       <n> should be a multiple of <nl3>"
	@echo 
	@echo "REMARK:  PATH To FEAST/MKL libraries can also be defined as shell variables FEASTROOT/MKLROOT"
	@echo
	@echo "************* Compilation ************************ "
	@echo "make COMP=<comp> MPI=<mpi>  all  -- Compile all F90 and C examples"
	@echo "make COMP=<comp> MPI=<mpi>  allF -- Compile all F90 examples"
	@echo "make COMP=<comp> MPI=<mpi>  allC -- Compile all C examples"
	@echo
	@echo "************* Both Compilation and Testing ********** "
	@echo "make COMP=<comp> MPI=<mpi> N=<n> NL3=<nl3> rall  -- Compile & Test all F90 and C examples"
	@echo "make COMP=<comp> MPI=<mpi> N=<n> NL3=<nl3> rallF -- Compile & Test all F90 examples"
	@echo "make COMP=<comp> MPI=<mpi> N=<n> NL3=<nl3> rallC -- Compile & Test all C examples"
	@echo
	@echo "************* Cleaning ************************** "
	@echo "make clean  -- clean all C and F90 examples object files"
	@echo "=========================================================================================================="

#===========================================================
# PATH To PFEAST/SPIKE/MKL library  can be defined as shell variable
#===========================================================
ARCH?=x64
FEASTROOT?=$(PWD)/../..
MKLROOT?=/opt/intel/compilers_and_libraries/linux/

LOCLIBS=-L${MKLROOT}/lib/intel64 -L$(FEASTROOT)/lib/$(ARCH)
FLIBS=$(LOCLIBS) -lpfeast
CLIBS=$(LOCLIBS) -lpfeast
INCL=-I$(FEASTROOT)/include

ifeq ($(BANDED),yes)
SPIKEROOT?=$(HOME)/spike-1.0
FLIBS+=-L$(SPIKEROOT)/lib/$(ARCH) -lspike
CLIBS+=-L$(SPIKEROOT)/lib/$(ARCH) -lspike
endif


#===========================================================
# Compiler set-up
#===========================================================
#####################
# intel
######################## 
ifeq ($(COMP),intel)
F90 = ifort
F90FLAGS= -O3 -qopenmp
FLIBS += -mkl=parallel -lmkl_blacs_intelmpi_lp64  -liomp5 -lpthread -lm -ldl

C = icc
CFLAGS= -O3 -qopenmp
CLIBS += -mkl=parallel -lmkl_blacs_intelmpi_lp64  -liomp5 -lpthread -lm -ldl -lirc -lifcore -lifcoremt
endif

#########################
# gnu 
######################### 
ifeq ($(COMP),gnu)
F90 = gfortran
F90FLAGS= -O3 -fopenmp -ffree-line-length-none -ffixed-line-length-none
FLIBS +=-Wl,--no-as-needed -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -lgomp -lpthread -lm -ldl

C= gcc
CFLAGS= -O3 -fopenmp -w
CLIBS +=-Wl,--no-as-needed -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -lgomp -lpthread -lm -ldl -lgfortran -lifcore
endif

#######################
# portland group
####################### 
ifeq ($(COMP),pgi)
F90 = pgf90
F90FLAGS= -O3 -mp
FLIBS +=-Wl,--no-as-needed -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -pgf90libs -mp -lpthread -lm -ldl

C= pgcc
CFLAGS= -O3 -mp 
CLIBS +=-Wl,--no-as-needed -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -pgf90libs -mp -lpthread -lm -ldl
endif



#===========================================================
# MPI directives set-up
#===========================================================

##################
### intel mpi
##################
ifeq ($(MPI),impi)
PF90=mpiifort -f90=$(F90)

PC=mpiicc -cc=$(C)

MPIRUN = mpirun
endif

###################
### mpich
###################
ifeq ($(MPI),mpich)
PF90= mpif90.mpich -fc=$(F90)

PC= mpicc -cc=$(C)

MPIRUN = mpirun
endif

####################
### openmpi ...requires shell environment variable "OMPI_FC=$(F90)"
###################
ifeq ($(MPI),openmpi)
export OMPI_FC=$(F90) # for BASH shell
PF90= mpif90.openmpi

export OMPI_CC=$(C) # for BASH shell
PC= mpicc.openmpi

MPIRUN = mpirun.openmpi
endif




#============================================================
# COMPILE and LINK
#============================================================

############# SPARSE


F90EXAMPLES_SPARSE = PF90sparse_pdfeast_scsrgv PF90sparse_pzfeast_hcsrev  PF90sparse_pdfeast_scsrgv_lowest PF90sparse_pdfeast_gcsrgv PF90sparse_pzfeast_scsrev PF90sparse_pdfeast_scsrpev 
sparse: examples_sparse
examples_sparse: 
	@for file in $(F90EXAMPLES_SPARSE); \
	do \
		echo $(PF90)  $(F90FLAGS)  -c $$file.f90 ;\
		$(PF90)  $(F90FLAGS)  -c $$file.f90 ;\
		echo $(PF90)   -o $$file $$file.o $(FLIBS) ;\
		$(PF90)   -o $$file $$file.o $(FLIBS) ;\
	done

CEXAMPLES_SPARSE = PCsparse_pdfeast_scsrgv PCsparse_pzfeast_hcsrev  PCsparse_pdfeast_scsrgv_lowest PCsparse_pdfeast_gcsrgv PCsparse_pzfeast_scsrev PCsparse_pdfeast_scsrpev
sparseC: examples_sparseC
examples_sparseC: 
	@for file in $(CEXAMPLES_SPARSE); \
	do \
		echo $(PC)  $(CFLAGS)  $(INCL) -c $$file.c ;\
		$(PC)  $(CFLAGS) $(INCL)  -c $$file.c ;\
		echo $(PC) $(CFLAGS)   -o $$file $$file.o $(CLIBS) ;\
		$(PC) $(CFLAGS)  -o $$file $$file.o $(CLIBS) ;\
	done



rsparse: examples_sparse
	@for file in $(F90EXAMPLES_SPARSE); \
	do \
		echo $(MPIRUN) -n $(N) ./$$file $(NL3); \
		$(MPIRUN) -n $(N) ./$$file $(NL3) |grep "F90sparse"; \
	done

rsparseC: examples_sparseC
	@for file in $(CEXAMPLES_SPARSE); \
	do \
		echo $(MPIRUN) -n $(N) ./$$file $(NL3); \
		$(MPIRUN) -n $(N) ./$$file $(NL3) |grep "F90sparse"; \
	done

####################################
################# GENERAL COMMAND
####################################


allF: examples_sparse
rallF: allF
	@for file in $(F90EXAMPLES_SPARSE); \
	do \
		echo $(MPIRUN) -n $(N) ./$$file $(NL3); \
		$(MPIRUN) -n $(N) ./$$file $(NL3) |grep "F90dense\|F90sparse\|F90banded"; \
	done

allC: examples_sparseC
rallC: allC
	@for file in $(CEXAMPLES_SPARSE); \
	do \
		echo $(MPIRUN) -n $(N) ./$$file $(NL3); \
		$(MPIRUN) -n $(N) ./$$file $(NL3) |grep "Cdense\|Csparse\|Cbanded"; \
	done

all: examples_sparse examples_sparseC
rall: all
	@for file in $(F90EXAMPLES_SPARSE) $(CEXAMPLES_SPARSE); \
	do \
		echo $(MPIRUN) -n $(N) ./$$file $(NL3); \
		$(MPIRUN) -n $(N) ./$$file $(NL3) |grep "F90dense\|F90sparse\|F90banded\|Cdense\|Csparse\|Cbanded"; \
	done

#==========================================================
# Clean up directory: delete object files and modules
#==========================================================
clean: 
	-@rm -fr $(F90EXAMPLES_DENSE) $(CEXAMPLES_DENSE)  $(F90EXAMPLES_BANDED)  $(CEXAMPLES_BANDED) $(F90EXAMPLES_SPARSE)  $(CEXAMPLES_SPARSE) *.o
