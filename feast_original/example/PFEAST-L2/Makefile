COMP?=intel

BANDED?=no

MPI?=impi

N?=2

help:
	@echo "=========================================================================================================="
	@echo "This Makefile provides a complete testing of few selected DENSE, BANDED, SPARSE interfaces in Fortran & C"
	@echo "It makes use of Intel-MKL for BLAS/LAPACK and PARDISO (if FEAST was compiled with MKL flag on)"
	@echo
	@echo "Usage: make COMP=<comp> MPI=<mpi> N=<n> BANDED=<banded> {allF rallF allC rallC all rall}"
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
	@echo "		<n>    is the number of MPI processors used for running the test. Defaults to 2"
	@echo "		<banded> is used to compile the banded interfaces (possible choices: yes, no). Defaults to 'no'"
	@echo "			To compile and run the BANDED interfaces, you need the SPIKE solver www.spike-solver.org"
	@echo "			then specify the SPIKEROOT directory as shell variable" 
	@echo "			(SPIKE must be compiled using same Fortran compiler used here)"
	@echo 
	@echo "REMARK:  PATH To FEAST/SPIKE/MKL libraries can also be defined as shell variables FEASTROOT/SPIKEROOT/MKLROOT"
	@echo
	@echo "************* Compilation ************************ "
	@echo "make COMP=<comp> MPI=<mpi> BANDED=<banded> all  -- Compile all F90 and C examples"
	@echo "make COMP=<comp> MPI=<mpi> BANDED=<banded> allF -- Compile all F90 examples"
	@echo "make COMP=<comp> MPI=<mpi> BANDED=<banded> allC -- Compile all C examples"
	@echo
	@echo "************* Both Compilation and Testing ********** "
	@echo "make COMP=<comp> MPI=<mpi> N=<n> BANDED=<banded> rall  -- Compile & Test all F90 and C examples"
	@echo "make COMP=<comp> MPI=<mpi> N=<n> BANDED=<banded> rallF -- Compile & Test all F90 examples"
	@echo "make COMP=<comp> MPI=<mpi> N=<n> BANDED=<banded> rallC -- Compile & Test all C examples"
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
FLIBS += -mkl

C = icc
CFLAGS= -O3 -qopenmp
CLIBS += -mkl -lirc -lifcore -lifcoremt
endif

#########################
# gnu 
######################### 
ifeq ($(COMP),gnu)
F90 = gfortran
F90FLAGS= -O3 -fopenmp -ffree-line-length-none -ffixed-line-length-none
FLIBS += -Wl,--start-group -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core  -Wl,--end-group -lgomp -lpthread -lm -ldl -lifcore

C= gcc
CFLAGS= -O3 -fopenmp -w
CLIBS += -Wl,--start-group -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core  -Wl,--end-group -lgomp -lpthread -lm -ldl -lgfortran -lifcore
endif

#######################
# portland group
####################### 
ifeq ($(COMP),pgi)
F90 = pgf90
F90FLAGS= -O3 -mp
FLIBS += -Wl,--start-group -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -Wl,--end-group -pgf90libs -mp -lpthread -lm -ldl  

C= pgcc
CFLAGS= -O3 -mp 
CLIBS += -Wl,--start-group -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -Wl,--end-group -lgomp -lpthread -lm -ldl -pgf90libs
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

############# DENSE

F90EXAMPLES_DENSE = PF90dense_dfeast_sygv PF90dense_zfeast_heev PF90dense_dfeast_gegv PF90dense_zfeast_syev PF90dense_dfeast_sypev
dense: examples_dense
examples_dense: 
	@for file in $(F90EXAMPLES_DENSE); \
	do \
		echo $(PF90)  $(F90FLAGS) -c $$file.f90 ;\
		$(PF90)  $(F90FLAGS) -c $$file.f90 ;\
		echo $(PF90)   -o $$file $$file.o $(FLIBS) ;\
		$(PF90)   -o $$file $$file.o $(FLIBS) ;\
	done

CEXAMPLES_DENSE = PCdense_dfeast_sygv PCdense_zfeast_heev PCdense_dfeast_gegv PCdense_zfeast_syev PCdense_dfeast_sypev
denseC: examples_denseC
examples_denseC: 
	@for file in $(CEXAMPLES_DENSE); \
	do \
		echo $(PC)  $(CFLAGS) $(INCL) -c $$file.c ;\
		$(PC)  $(CFLAGS) $(INCL) -c $$file.c ;\
		echo $(PC) $(CFLAGS)   -o $$file $$file.o $(CLIBS) ;\
		$(PC) $(CFLAGS) -o $$file $$file.o $(CLIBS) ;\
	done



rdense: examples_dense
	@for file in $(F90EXAMPLES_DENSE); \
	do \
	        echo $(MPIRUN) -n $(N) ./$$file; \
	        $(MPIRUN) -n $(N) ./$$file | grep "F90dense"; \
	done

rdenseC:examples_denseC
	@for file in $(CEXAMPLES_DENSE); \
	do \
	        echo $(MPIRUN) -n $(N) ./$$file; \
	        $(MPIRUN) -n $(N) ./$$file |grep "Cdense"; \
	done


############ BANDED

F90EXAMPLES_BANDED = PF90banded_dfeast_sbgv  PF90banded_zfeast_hbev PF90banded_dfeast_gbgv PF90banded_zfeast_sbev
banded: examples_banded
examples_banded: 
	@for file in $(F90EXAMPLES_BANDED); \
	do \
		echo $(PF90)  $(F90FLAGS) -c $$file.f90 ;\
		$(PF90)  $(F90FLAGS) -c $$file.f90 ;\
		echo $(PF90)   -o $$file $$file.o $(FLIBS);\
		$(PF90)   -o $$file $$file.o  $(FLIBS);\
	done

rbanded:examples_banded
	@for file in $(F90EXAMPLES_BANDED); \
	do \
		echo $(MPIRUN) -n $(N) ./$$file; \
		$(MPIRUN) -n $(N) ./$$file |grep "F90banded"; \
	done

CEXAMPLES_BANDED = PCbanded_dfeast_sbgv  PCbanded_zfeast_hbev PCbanded_dfeast_gbgv PCbanded_zfeast_sbev
bandedC: examples_bandedC
examples_bandedC: 
	@for file in $(CEXAMPLES_BANDED); \
	do \
		echo $(PC)  $(FLAGS) $(INCL) -c $$file.c ;\
		$(PC)  $(CFLAGS) $(INCL) -c $$file.c ;\
		echo $(PC) $(CFLAGS)   -o $$file $$file.o $(CLIBS);\
		$(PC) $(CFLAGS)   -o $$file $$file.o $(CLIBS);\
	done

rbandedC:examples_bandedC
	@for file in $(CEXAMPLES_BANDED); \
	do \
		echo $(MPIRUN) -n $(N) ./$$file; \
		$(MPIRUN) -n $(N) ./$$file |grep "Cbanded"; \
	done

############# SPARSE


F90EXAMPLES_SPARSE = PF90sparse_dfeast_scsrgv PF90sparse_zfeast_hcsrev  PF90sparse_dfeast_scsrgv_lowest PF90sparse_dfeast_gcsrgv PF90sparse_zfeast_scsrev PF90sparse_dfeast_scsrpev
sparse: examples_sparse
examples_sparse: 
	@for file in $(F90EXAMPLES_SPARSE); \
	do \
		echo $(PF90)  $(F90FLAGS)  -c $$file.f90 ;\
		$(PF90)  $(F90FLAGS)  -c $$file.f90 ;\
		echo $(PF90)   -o $$file $$file.o $(FLIBS) ;\
		$(PF90)   -o $$file $$file.o $(FLIBS) ;\
	done

CEXAMPLES_SPARSE = PCsparse_dfeast_scsrgv PCsparse_zfeast_hcsrev  PCsparse_dfeast_scsrgv_lowest PCsparse_dfeast_gcsrgv PCsparse_zfeast_scsrev PCsparse_dfeast_scsrpev
sparseC: examples_sparseC
examples_sparseC: 
	@for file in $(CEXAMPLES_SPARSE); \
	do \
		echo $(PC)  $(CFLAGS)  $(INCL) -c $$file.c ;\
		$(PC)  $(CFLAGS) $(INCL)  -c $$file.c ;\
		echo $(PC) $(CFLAGS)   -o $$file $$file.o $(CLIBS) ;\
		$(PC) $(CFLAGS)   -o $$file $$file.o $(CLIBS) ;\
	done


rsparse: examples_sparse
	@for file in $(F90EXAMPLES_SPARSE); \
	do \
		echo $(MPIRUN) -n $(N) ./$$file; \
		$(MPIRUN) -n $(N) ./$$file |grep "F90sparse"; \
	done

rsparseC: examples_sparseC
	@for file in $(CEXAMPLES_SPARSE); \
	do \
		echo $(MPIRUN) -n $(N) ./$$file; \
		$(MPIRUN) -n $(N) ./$$file |grep "F90sparse"; \
	done

####################################
################# GENERAL COMMAND
####################################

ifeq ($(BANDED),yes) 

allF: examples_dense examples_sparse examples_banded
rallF: allF
	@for file in $(F90EXAMPLES_DENSE) $(F90EXAMPLES_SPARSE) $(F90EXAMPLES_BANDED); \
	do \
		echo $(MPIRUN) -n $(N) ./$$file; \
		$(MPIRUN) -n $(N) ./$$file |grep "F90dense\|F90sparse\|F90banded"; \
	done

allC: examples_denseC examples_sparseC examples_bandedC
rallC: allC
	@for file in $(CEXAMPLES_DENSE) $(CEXAMPLES_SPARSE) $(CEXAMPLES_BANDED); \
	do \
		echo $(MPIRUN) -n $(N) ./$$file; \
		$(MPIRUN) -n $(N) ./$$file |grep "Cdense\|Csparse\|Cbanded"; \
	done

all: examples_dense examples_sparse examples_banded examples_denseC examples_sparseC examples_bandedC
rall: all
	@for file in $(F90EXAMPLES_DENSE) $(F90EXAMPLES_SPARSE) $(F90EXAMPLES_BANDED) $(CEXAMPLES_DENSE) $(CEXAMPLES_SPARSE) $(CEXAMPLES_BANDED); \
	do \
		echo $(MPIRUN) -n $(N) ./$$file; \
		$(MPIRUN) -n $(N) ./$$file |grep "F90dense\|F90sparse\|F90banded\|Cdense\|Csparse\|Cbanded"; \
	done
else


allF: examples_dense examples_sparse
rallF: allF
	@for file in $(F90EXAMPLES_DENSE) $(F90EXAMPLES_SPARSE); \
	do \
		echo $(MPIRUN) -n $(N) ./$$file; \
		 $(MPIRUN) -n $(N) ./$$file |grep "F90dense\|F90sparse"; \
	done

allC: examples_denseC examples_sparseC
rallC: allC
	@for file in $(CEXAMPLES_DENSE) $(CEXAMPLES_SPARSE); \
	do \
		echo $(MPIRUN) -n $(N) ./$$file; \
		 $(MPIRUN) -n $(N) ./$$file |grep "Cdense\|Csparse"; \
	done

all: examples_dense examples_sparse examples_denseC examples_sparseC
rall: all
	@for file in $(F90EXAMPLES_DENSE) $(F90EXAMPLES_SPARSE) $(CEXAMPLES_DENSE) $(CEXAMPLES_SPARSE); \
	do \
		echo $(MPIRUN) -n $(N) ./$$file; \
		$(MPIRUN) -n $(N) ./$$file |grep "F90dense\|F90sparse\|Cdense\|Csparse"; \
	done
endif



#==========================================================
# Clean up directory: delete object files and modules
#==========================================================
clean: 
	-@rm -fr $(F90EXAMPLES_DENSE) $(CEXAMPLES_DENSE)  $(F90EXAMPLES_BANDED)  $(CEXAMPLES_BANDED) $(F90EXAMPLES_SPARSE)  $(CEXAMPLES_SPARSE) *.o
