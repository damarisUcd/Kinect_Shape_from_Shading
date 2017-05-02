CXX = clang++
LINK = clang++
NVCC = /usr/local/cuda/bin/nvcc
SRC ?= src
CUDAHOME += /usr/local/cuda
TERRAHOME ?= $(HOME)/Optlang/terra
TERRA ?= $(HOME)/Optlang/terra/release/terra
OPT = $(HOME)/Optlang/Opt/API/release

ifeq ($(UNAME), Linux)
  S4 = 64
endif

FLAGS += -O3 -g -std=c++11 -Isrc -Isrc/cutil/inc -I$(OPT)/include -I$(TERRAHOME)/include -I$(CUDAHOME)/include -Iexternal/mLib/include -Iexternal -Iexternal/OpenMesh/include
NVCCFLAGS += -arch sm_30 $(FLAGS)
LFLAGS += -std=c++11 -L$(OPT)/lib -L$(CUDAHOME)/lib$(S4) -lcudart -L$(TERRAHOME)/lib -lterra -ldl
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv)
OPENCV_LIBS = $(shell pkg-config --libs opencv)
FREE_INCLUDE = -I$(HOME)/freenect2/include/
FREE_LIBS_DIR = $(HOME)/freenect2/lib/
FREE_LIBS = -L$(FREE_LIBS_DIR) -lfreenect2 -Wl,-rpath=$(FREE_LIBS_DIR),--enable-new-dtags
UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
  LFLAGS += -pagezero_size 10000 -image_base 100000000
endif

ifeq ($(UNAME), Linux)
  LFLAGS += -Wl,-rpath,$(CUDAHOME)/lib$(S4) -pthread
  S4 = 64
endif

.SUFFIXES:
.PHONY:	all purge clean

all: $(EXECUTABLE)

OPTLIB = $(OPT)/lib/libOptDev.a

build/%.o:	$(SRC)/%.cpp
	@mkdir -p build
	$(CXX) $(FLAGS) $(FREE_INCLUDE) -MP -MD $< -c -o $@
build/%.o:	$(SRC)/%.cu
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) $(FREE_INCLUDE) $< -c -o $@

$(EXECUTABLE):	$(OBJS) $(OPTLIB)
	$(CXX) $(OPENCV_CFLAGS) $^ -o $@ $(LFLAGS) $(OPENCV_LIBS) $(FREE_LIBS)

clean:
	rm -rf build/*.o build/*.d
	rm -rf $(EXECUTABLE)

purge: clean
	rm -rf build/*

# dependency rules
include $(wildcard build/*.d)
