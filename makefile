# credits to Basharam (https://gist.github.com/basharam/9511906)
#  Makefile template for Static library. 
# 1. Compile every *.cpp in the folder 
# 2. All obj files under obj folder
# 3. static library .a at lib folder
# 4. run 'make dirmake' before calling 'make'


CC = gcc
OUT_FILE_NAME = libworldrep.a

CFLAGS= -fPIC -O0 -Wall -c -fpermissive -std=c++11 -DARMA_DONT_USE_WRAPPER -lopenblas -llapack -larmadillo

INC = -I.

OBJ_DIR=./obj

OUT_DIR=./lib

_LIBSNN = libsnnlfisrm.a
LIBSNN = $(patsubst %,./libsnnlfisrm/lib/%,$(_LIBSNN))

# Enumerating of every *.cpp as *.o and using that as dependency.	
# filter list of .c files in a directory.
FILES=world_rep.cpp

$(OUT_FILE_NAME): $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(wildcard $(FILES))) $(LIBSNN)
	ar -r -o $(OUT_DIR)/$@ $^

# Enumerating of every *.cpp as *.o and using that as dependency
#$(OUT_FILE_NAME): $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(wildcard *.cpp))




#Compiling every *.cpp to *.o
$(OBJ_DIR)/%.o: %.cpp dirmake
	$(CC) -c $(INC) $(CFLAGS) -o $@  $<
	
dirmake:
	@mkdir -p $(OUT_DIR)
	@mkdir -p $(OBJ_DIR)
	
clean:
	rm -f $(OBJ_DIR)/*.o $(OUT_DIR)/$(OUT_FILE_NAME) Makefile.bak

rebuild: clean build

####################################################################