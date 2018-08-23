# Old style Makefile for people who find CMake mysterious.

CPP=g++ -Wall -fPIC ${INCLUDE} -Wno-unused-variable -Wno-format-security -std=c++1y
CC=gcc -Wall ${INCLUDE}
EXEC = run_tests
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

%.o : %.cpp %.h
	${CPP} -c $< -o $@

test: run_tests
	( LD_LIBRARY_PATH=. ; ./run_tests )

run_tests: run_tests.c libminorminer_clib.so
	${CC} -L. -o run_tests run_tests.c -lminorminer_clib -lpthread

libminorminer_clib.so: $(OBJECTS)
	g++ -shared -o libminorminer_clib.so $(OBJECTS)

clean:
	rm -f ${EXEC} $(OBJECTS) libminorminer_clib.so
