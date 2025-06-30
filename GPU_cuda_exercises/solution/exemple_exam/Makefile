# make 						# Compile all binaries
# make clean				# Remove all binaries and reports
# make ncu_reports			# Compile all binaries (if not done before) and create all ncu reports for them
# make clean_ncu_reports 	# Remove all ncu reports
# make nsys_reports			# Compile all binaries (if not done before) and create all nsys reports for them
# make clean_nsys_reports	# Remove all nsys reports
# make ncu_report_<name>	# Compile binary of name <name> (if not done before) and create its ncu report  

.PHONY: all clean print_header ncu_reports clean_ncu_reports \
	nsys_reports clean_nsys_reports ncu_report_%

# Compiler to use
CC := nvcc
SRCS := $(wildcard *.cu)
NCU_REPORTS := $(SRCS:%.cu=%.ncu-rep)
NSYS_REPORTS := $(SRCS:%.cu=%.nsys-rep)
#Replace %(Any match).cu contained in SRCS with %(Any match).o 
BINS := $(SRCS:%.cu=%.o)
			

	
all: print_header $(BINS)
	
print_header :
	@echo "--------------------------SOLUTIONS--------------------------"
	@echo "Binaries to build: "$(BINS)

#Target : Prerequisites
%.o : %.cu 
	@echo "Creating binary $@..."
	@if [ "$<" = matrix-mul.cu ]; then \
		$(CC) -arch=sm_75 $< -lcublas -o $@; \
	else $(CC) -arch=sm_75 $< -o $@; \
	fi
# $@ = Target generated (%.o in this case)
# $< = First prerequisite (%.cu in this case)

ncu_report_% : %.o
	@echo "Creating report of $<..."
	ncu --set full -o $(patsubst %.o,%,$<) ./$<

ncu_reports : $(NCU_REPORTS)

%.ncu-rep : %.o
	@echo "Creating report of $<..."
	ncu --set full -o $@ ./$<

nsys_reports : $(NSYS_REPORTS)

%.nsys-rep : %.o
	@echo "Creating report of $<..."
	nsys profile -o $@ $<

clean: clean_ncu_reports clean_nsys_reports
	@echo "Cleaning everything up..."
	rm -rf *.o $(BINS)

clean_ncu_reports :
	@echo "Cleaning up ncu reports..."
	rm -rf *.ncu-rep $(NCU_REPORTS)

clean_nsys_reports:
	@echo "Cleaning up nsys reports..."
	rm -rf *.nsys-rep $(NSYS_REPORTS)

