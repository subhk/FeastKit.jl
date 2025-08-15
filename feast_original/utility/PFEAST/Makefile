F90?=ifort
MPI?=impi

help:	
	@echo "----------------------------------------------------------------------------------------------------------"
	@echo "Usage: make F90=<f90> MPI=<mpi> {all clean} "
	@echo "" 
	@echo "<f90> is your own Fortran90 compiler (possible choices: ifort, gfortran, pgf90). Defaults to ifort"
	@echo "                           ...this should be the same Fortran used for compiling PFEAST"
	@echo "<mpi> is your MPI library (possible choices: impi, mpich, openmpi). Defaults to impi (intel MPI)"
	@echo "                           ...this should be the same MPI used for compiling PFEAST"
	@echo ""
	@echo "Remark: This Makefile uses *Intel-MKL* for BLAS/LAPACK/Cluster-PARDISO"
	@echo "        if you wish to use another BLAS/LAPACK you must compile by hand e.g.:"
	@echo "        	       <mpif90> -o driver_feast_sparse driver_feast_sparse.f90 <lapack> <blas>"
	@echo "        Note that MKL-Cluster-PARDISO will not then be available and the sparse interfaces will default to PIFEAST"
	@echo "----------------------------------------------------------------------------------------------------------"
        

#===========================================================
# Fortran Compiler set-up
#===========================================================

# intel fortran
ifeq ($(F90),ifort)
F90FLAGS= -O3 -qopenmp
endif

# gnu fortran
ifeq ($(F90),gfortran)
F90FLAGS= -O3 -fopenmp -ffree-line-length-none -ffixed-line-length-none 
endif

# portland group fortran compiler
ifeq ($(F90),pgf90)
F90FLAGS= -O3 -mp
endif


##############################################################
######### include MPI directives
##############################################################


### intel mpi
ifeq ($(MPI),impi)
PF90=mpiifort -f90=$(F90)
PF90FLAGS= $(F90FLAGS) #-DMPI
endif


### mpich
ifeq ($(MPI),mpich)
PF90= mpif90.mpich -fc=$(F90)
PF90FLAGS= $(F90FLAGS) #-DMPI
endif

### openmpi ...requires shell environment variable "OMPI_FC=$(F90)"
ifeq ($(MPI),openmpi)
export OMPI_FC=$(F90) # for BASH shell
PF90= mpif90.openmpi
PF90FLAGS= $(F90FLAGS) #-DMPI
endif



#===========================================================
# PATH To FEAST library
#===========================================================
ARCH?=x64
FEASTROOT?=$(PWD)/../..
LOCLIBS=-L$(FEASTROOT)/lib/$(ARCH)
LIB=$(LOCLIBS) -lpfeast

#===========================================================
# MKL Link set-up
#===========================================================


# intel fortran
ifeq ($(F90),ifort)
LIB += -mkl=parallel -lmkl_blacs_intelmpi_lp64  -liomp5 -lpthread -lm -ldl
endif

# gnu fortran
ifeq ($(F90),gfortran)
LIB += -Wl,--no-as-needed -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -lgomp -lpthread -lm -ldl
endif

# portland group fortran compiler (=========>not tested<==============)
ifeq ($(F90),pgf90)
FLIBS +=-Wl,--no-as-needed -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -pgf90libs -mp -lpthread -lm -ldl
endif



#============================================================
# COMPILE and LINK
#============================================================
EXAMPLES = driver_pfeast_sparse 


all: examples 


examples: 
	@for file in $(EXAMPLES); \
	do \
		echo $(PF90)  $(PF90FLAGS) -c $$file.f90 ;\
		$(PF90)  $(PF90FLAGS) -c $$file.f90 ;\
		echo $(PF90)   -o $$file $$file.o $(LIB) ;\
		$(PF90)   -o $$file $$file.o $(LIB) ;\
	done


#==========================================================
# Clean up directory: delete object files and modules
#==========================================================
clean: 
	-@rm  $(EXAMPLES) *.o
