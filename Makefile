EXECUTABLE = shape_from_shading
OBJS = build/mLibSource.o build/main.o build/SFSSolver.o  build/SimpleBuffer.o


UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
  LFLAGS += -Lexternal/OpenMesh/lib/osx -Wl,-rpath,external/OpenMesh/lib/osx
endif

ifeq ($(UNAME), Linux)
  LFLAGS += -Lexternal/OpenMesh/lib/linux -Wl,-rpath,external/OpenMesh/lib/linux
endif

LFLAGS += -lOpenMeshCore -lOpenMeshTools


include make_template.inc
