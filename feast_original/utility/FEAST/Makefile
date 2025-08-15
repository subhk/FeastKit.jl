F90?=ifort

help:	
	@echo "----------------------------------------------------------------------------------------------------------"
	@echo "Usage: make F90=<f90> {all clean} "
	@echo "" 
	@echo "<f90> is your own Fortran90 compiler (possible choices: ifort, gfortran, pgf90). Defaults to ifort"
	@echo "                           ...this should be the same Fortran used for compiling PFEAST"
	@echo ""
	@echo "Remark: This Makefile uses *Intel-MKL* for BLAS/LAPACK/PARDISO"
	@echo "        if you wish to use another BLAS/LAPACK you must compile by hand e.g.:"
	@echo "        	       <f90> -o driver_feast_sparse driver_feast_sparse.f90 <lapack> <blas>"
	@echo "        Note that MKL-PARDISO will not then be available and the sparse interfaces will default to IFEAST"
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


#===========================================================
# PATH To FEAST library
#===========================================================
ARCH?=x64
FEASTROOT?=$(PWD)/../..
MKLROOT?=/opt/intel/compilers_and_libraries/linux/
LOCLIBS=-L${MKLROOT}/lib/intel64 -L$(FEASTROOT)/lib/$(ARCH)
LIB=$(LOCLIBS) -lfeast


#===========================================================
# MKL Link set-up
#===========================================================


# intel fortran
ifeq ($(F90),ifort)
LIB += -mkl
endif

# gnu fortran
ifeq ($(F90),gfortran)
LIB += -Wl,--start-group -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core  -Wl,--end-group -lgomp -lpthread -lm -ldl
endif

# portland group fortran compiler (=========>not tested<==============)
ifeq ($(F90),pgf90)
LIB += -Wl,--start-group -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -Wl,--end-group -pgf90libs -mp -lpthread -lm -ldl
endif



#============================================================
# COMPILE and LINK
#============================================================
EXAMPLES = driver_feast_sparse 


all: examples 


examples: 
	@for file in $(EXAMPLES); \
	do \
		echo $(F90)  $(F90FLAGS) -c $$file.f90 ;\
		$(F90)  $(F90FLAGS) -c $$file.f90 ;\
		echo $(F90)   -o $$file $$file.o $(LIB) ;\
		$(F90)   -o $$file $$file.o $(LIB) ;\
	done


#==========================================================
# Clean up directory: delete object files and modules
#==========================================================
clean: 
	-@rm  $(EXAMPLES) *.o
