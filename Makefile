ARCH = 61

NVCC = nvcc
NVCC_ARCH = -gencode arch=compute_$(ARCH),code=sm_$(ARCH)
NVCC_FLAGS = --std=c++14 -O3 -arch=sm_70 -Xcompiler="-pthread" -D CUDA -D MULTI

BIN = -o bin

INC_DIR = include
SRC_DIR = src
OBJ_DIR = bin
EXEC_NAME = demo

SRC_FILES := $(wildcard $(SRC_DIR)/*.cc)
CU_SRC_FILES := $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cc,$(OBJ_DIR)/%.o,$(SRC_FILES))
CU_OBJ_FILES := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.obj,$(CU_SRC_FILES))

default: link

link: multians gpu
	$(NVCC) $(OBJ_FILES) $(CU_OBJ_FILES) -o $(OBJ_DIR)/$(EXEC_NAME)

multians: $(OBJ_FILES)

gpu: $(CU_OBJ_FILES)

$(OBJ_DIR)/%.obj: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_ARCH) $(NVCC_FLAGS) -I $(INC_DIR) -c -o $@ $<

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc
	$(NVCC) $(NVCC_ARCH) $(NVCC_FLAGS) -I $(INC_DIR) -c -o $@ $<

.PHONY: clean
clean:
	rm -f $(RM_FLAGS) $(OBJ_DIR)/*.o
	rm -f $(RM_FLAGS) $(OBJ_DIR)/*.obj
	rm -f $(RM_FLAGS) $(OBJ_DIR)/$(EXEC_NAME)
