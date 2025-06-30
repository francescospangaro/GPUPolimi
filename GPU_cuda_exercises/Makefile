objects = ex sol
dirs = exercise solution

all: ${objects}

ex :
	cd ./exercise && $(MAKE)

sol :
	cd ./solution && $(MAKE)

clean :
	@for i in $(dirs) ; do \
		cd $$i && $(MAKE) clean; \
		cd .. ; \
	done

.PHONY : all ex sol clean