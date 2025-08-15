COMP?=intel

BANDED?=no

help:
	@echo "=========================================================================================================="
	@echo "This Makefile provides a complete testing of few selected DENSE, BANDED, SPARSE interfaces in Fortran & C"
	@echo "It makes use of Intel-MKL for BLAS/LAPACK and PARDISO (if FEAST was compiled with MKL flag on)"
	@echo
	@echo "Usage: make COMP=<comp> BANDED=<banded> {allF rallF allC rallC all rall}"
	@echo
	@echo "      <comp> is used to select your compilers (possible choices: intel, gnu, pgi). Defaults to 'intel'"
	@echo "                 this should be the same Fortran compiler used for compiling FEAST"
	@echo "		    -'intel' selects ifort for F90 examples and icc for C examples"
	@echo "		    -'gnu'   selects gfortran for F90 examples and gcc for C examples"
	@echo "		    -'pgi'   selects pgf90 for F90 examples and pgcc for C examples"
	@echo "      <banded> is used to compile the banded interfaces (possible choices: yes, no). Defaults to 'no'"
	@echo "			To compile and run the BANDED interfaces, you need the SPIKE solver www.spike-solver.org"
	@echo "			then specify the SPIKEROOT directory as shell variable" 
	@echo "			(SPIKE must be compiled using same Fortran compiler used here)"
	@echo 
	@echo "REMARK:  PATH To FEAST/SPIKE/MKL libraries can also be defined as shell variables FEASTROOT/SPIKEROOT/MKLROOT"
	@echo
	@echo "************* Compilation ************************ "
	@echo "make COMP=<comp> BANDED=<banded> all  -- Compile all F90 and C examples"
	@echo "make COMP=<comp> BANDED=<banded> allF -- Compile all F90 examples"
	@echo "make COMP=<comp> BANDED=<banded> allC -- Compile all C examples"
	@echo
	@echo "************* Both Compilation and Testing ********** "
	@echo "make COMP=<comp> BANDED=<banded> rall  -- Compile & Test all F90 and C examples"
	@echo "make COMP=<comp> BANDED=<banded> rallF -- Compile & Test all F90 examples"
	@echo "make COMP=<comp> BANDED=<banded> rallC -- Compile & Test all C examples"
	@echo
	@echo "************* Cleaning ************************** "
	@echo "make clean  -- clean all C and F90 examples object files"
	@echo 
	@echo "************* Few Examples ************************** "
	@echo "1- 'make all'                        Compile all Fortran/C examples (no banded) using ifort and icc (default)"
	@echo "2- 'make rall'                       Compile & Test all Fortran/C examples (no banded) (return success or Failure)"
	@echo "3- 'make COMP=gnu allC'              Compile all C examples using gcc"
	@echo "4- 'make COMP=pgi BANDED=yes rallF'  Compile & Test all Fortran examples (with banded) using pgf90"
	@echo "5- 'make COMP=gnu BANDED=yes rall'   Compile & Test all Fortran/C examples (with banded) using gfortran/gcc"
	@echo "=========================================================================================================="

#===========================================================
# PATH To FEAST/SPIKE/MKL library  can be defined as shell variable
#===========================================================
ARCH?=x64
FEASTROOT?=$(PWD)/../..
MKLROOT?=/opt/intel/compilers_and_libraries/linux/

LOCLIBS=-L${MKLROOT}/lib/intel64 -L$(FEASTROOT)/lib/$(ARCH)
FLIBS=$(LOCLIBS) -lfeast
CLIBS=$(LOCLIBS) -lfeast
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
FLIBS += -Wl,--start-group -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core  -Wl,--end-group -lgomp -lpthread -lm -ldl

C= gcc
CFLAGS= -O3 -fopenmp -w
CLIBS += -Wl,--start-group -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core  -Wl,--end-group -lgomp -lpthread -lm -ldl -lgfortran
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


#============================================================
# COMPILE and LINK
#============================================================

############# DENSE

F90EXAMPLES_DENSE = F90dense_dfeast_sygv F90dense_zfeast_heev F90dense_dfeast_gegv F90dense_zfeast_syev F90dense_zfeast_syevx F90dense_dfeast_sypev 
dense: examples_dense
examples_dense: 
	@for file in $(F90EXAMPLES_DENSE); \
	do \
		echo $(F90)  $(F90FLAGS) -c $$file.f90 ;\
		$(F90)  $(F90FLAGS) -c $$file.f90 ;\
		echo $(F90)   -o $$file $$file.o $(FLIBS) ;\
		$(F90)   -o $$file $$file.o $(FLIBS) ;\
	done

CEXAMPLES_DENSE = Cdense_dfeast_sygv Cdense_zfeast_heev Cdense_dfeast_gegv Cdense_zfeast_syev Cdense_zfeast_syevx Cdense_dfeast_sypev 
denseC: examples_denseC
examples_denseC: 
	@for file in $(CEXAMPLES_DENSE); \
	do \
		echo $(C)  $(CFLAGS) $(INCL) -c $$file.c ;\
		$(C)  $(CFLAGS) $(INCL) -c $$file.c ;\
		echo $(C) $(CFLAGS) -o $$file $$file.o $(CLIBS) ;\
		$(C) $(CFLAGS) -o $$file $$file.o $(CLIBS) ;\
	done



rdense: examples_dense
	@for file in $(F90EXAMPLES_DENSE); \
	do \
	        ./$$file |grep "F90dense"; \
	done

rdenseC:examples_denseC
	@for file in $(CEXAMPLES_DENSE); \
	do \
	        ./$$file |grep "Cdense"; \
	done


############ BANDED

F90EXAMPLES_BANDED = F90banded_dfeast_sbgv  F90banded_zfeast_hbev  F90banded_dfeast_gbgv  F90banded_zfeast_sbev F90banded_zfeast_sbevx
banded: examples_banded
examples_banded: 
	@for file in $(F90EXAMPLES_BANDED); \
	do \
		echo $(F90)  $(F90FLAGS) -c $$file.f90 ;\
		$(F90)  $(F90FLAGS) -c $$file.f90 ;\
		echo $(F90)   -o $$file $$file.o $(FLIBS);\
		$(F90)   -o $$file $$file.o  $(FLIBS);\
	done

rbanded:examples_banded
	@for file in $(F90EXAMPLES_BANDED); \
	do \
		./$$file |grep "F90banded"; \
	done

CEXAMPLES_BANDED = Cbanded_dfeast_sbgv  Cbanded_zfeast_hbev  Cbanded_dfeast_gbgv  Cbanded_zfeast_sbev Cbanded_zfeast_sbevx
bandedC: examples_bandedC
examples_bandedC: 
	@for file in $(CEXAMPLES_BANDED); \
	do \
		echo $(C)  $(CFLAGS) $(INCL) -c $$file.c ;\
		$(C)  $(CFLAGS) $(INCL) -c $$file.c ;\
		echo $(C) $(CFLAGS)   -o $$file $$file.o $(CLIBS);\
		$(C) $(CFLAGS)   -o $$file $$file.o $(CLIBS);\
	done

rbandedC:examples_bandedC
	@for file in $(CEXAMPLES_BANDED); \
	do \
		./$$file |grep "Cbanded"; \
	done



############# SPARSE


F90EXAMPLES_SPARSE = F90sparse_dfeast_scsrgv F90sparse_zfeast_hcsrev  F90sparse_dfeast_scsrgv_lowest F90sparse_dfeast_gcsrgv F90sparse_zfeast_scsrev  F90sparse_zfeast_scsrevx F90sparse_dfeast_scsrpev
sparse: examples_sparse
examples_sparse: 
	@for file in $(F90EXAMPLES_SPARSE); \
	do \
		echo $(F90)  $(F90FLAGS)  -c $$file.f90 ;\
		$(F90)  $(F90FLAGS)  -c $$file.f90 ;\
		echo $(F90)   -o $$file $$file.o $(FLIBS) ;\
		$(F90)   -o $$file $$file.o $(FLIBS) ;\
	done

CEXAMPLES_SPARSE = Csparse_dfeast_scsrgv Csparse_zfeast_hcsrev  Csparse_dfeast_scsrgv_lowest Csparse_dfeast_gcsrgv Csparse_zfeast_scsrev  Csparse_zfeast_scsrevx Csparse_dfeast_scsrpev
sparseC: examples_sparseC
examples_sparseC: 
	@for file in $(CEXAMPLES_SPARSE); \
	do \
		echo $(C)  $(CFLAGS)  $(INCL) -c $$file.c ;\
		$(C)  $(CFLAGS) $(INCL)  -c $$file.c ;\
		echo $(C) $(CFLAGS)   -o $$file $$file.o $(CLIBS) ;\
		$(C) $(CFLAGS)   -o $$file $$file.o $(CLIBS) ;\
	done



rsparse: examples_sparse
	@for file in $(F90EXAMPLES_SPARSE); \
	do \
		./$$file |grep "F90sparse"; \
	done

rsparseC: examples_sparseC
	@for file in $(CEXAMPLES_SPARSE); \
	do \
		./$$file |grep "Csparse"; \
	done

####################################
################# GENERAL COMMAND
####################################

ifeq ($(BANDED),yes) 

allF: examples_dense examples_sparse examples_banded
rallF: allF
	@for file in $(F90EXAMPLES_DENSE) $(F90EXAMPLES_SPARSE) $(F90EXAMPLES_BANDED); \
	do \
		./$$file |grep "F90dense\|F90sparse\|F90banded"; \
	done

allC: examples_denseC examples_sparseC examples_bandedC
rallC: allC
	@for file in $(CEXAMPLES_DENSE) $(CEXAMPLES_SPARSE) $(CEXAMPLES_BANDED); \
	do \
		./$$file |grep "Cdense\|Csparse\|Cbanded"; \
	done

all: examples_dense examples_sparse examples_banded examples_denseC examples_sparseC examples_bandedC
rall: all
	@for file in $(F90EXAMPLES_DENSE) $(F90EXAMPLES_SPARSE) $(F90EXAMPLES_BANDED) $(CEXAMPLES_DENSE) $(CEXAMPLES_SPARSE) $(CEXAMPLES_BANDED); \
	do \
		./$$file |grep "F90dense\|F90sparse\|F90banded\|Cdense\|Csparse\|Cbanded"; \
	done
else


allF: examples_dense examples_sparse
rallF: allF
	@for file in $(F90EXAMPLES_DENSE) $(F90EXAMPLES_SPARSE); \
	do \
		./$$file |grep "F90dense\|F90sparse"; \
	done

allC: examples_denseC examples_sparseC
rallC: allC
	@for file in $(CEXAMPLES_DENSE) $(CEXAMPLES_SPARSE); \
	do \
		./$$file |grep "F90dense\|F90sparse\|Cdense\|Csparse"; \
	done

all: examples_dense examples_sparse examples_denseC examples_sparseC
rall: all
	@for file in $(F90EXAMPLES_DENSE) $(F90EXAMPLES_SPARSE) $(CEXAMPLES_DENSE) $(CEXAMPLES_SPARSE); \
	do \
		./$$file |grep "F90dense\|F90sparse\|Cdense\|Csparse"; \
	done
endif



#==========================================================
# Clean up directory: delete object files and modules
#==========================================================
clean: 
	-@rm -fr $(F90EXAMPLES_DENSE) $(CEXAMPLES_DENSE)  $(F90EXAMPLES_BANDED)  $(CEXAMPLES_BANDED) $(F90EXAMPLES_SPARSE)  $(CEXAMPLES_SPARSE) *.o
