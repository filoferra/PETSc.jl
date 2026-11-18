# Build the C version of ex45 against an installed PETSc:  make PETSC_DIR=/path/to/petsc
include $(PETSC_DIR)/lib/petsc/conf/variables
include $(PETSC_DIR)/lib/petsc/conf/rules

ex45_julia: ex45_julia.o
	$(CLINKER) -o $@ $^ $(PETSC_LIB)
