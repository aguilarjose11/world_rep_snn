# credits to Basharam (https://gist.github.com/basharam/9511906)
#  Makefile template for Static library. 
# 1. Compile every *.cpp in the folder 
# 2. All obj files under obj folder
# 3. static library .a at lib folder
# 4. run 'make dirmake' before calling 'make'


CC = g++
OUT_FILE_NAME = libsnnlfisrm.a

CFLAGS= -fPIC -O0 -g -Wall -c -fpermissive -std=c++11

INC = -I.

OBJ_DIR=./obj

OUT_DIR=./lib

# Enumerating of every *.cpp as *.o and using that as dependency.	
# filter list of .c files in a directory.
# FILES =dump_l.c \
#	kter.c \
#
# $(OUT_FILE_NAME): $(patsubst %.c,$(OBJ_DIR)/%.o,$(wildcard $(FILES))) 


# Enumerating of every *.cpp as *.o and using that as dependency
$(OUT_FILE_NAME): $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(wildcard *.cpp))
	ar -r -o $(OUT_DIR)/$@ $^



#Compiling every *.cpp to *.o
$(OBJ_DIR)/%.o: %.cpp dirmake
	$(CC) -c $(INC) $(CFLAGS) -o $@  $<
	
dirmake:
	@mkdir -p $(OUT_DIR)
	@mkdir -p $(OBJ_DIR)
	
clean:
	rm -f $(OBJ_DIR)/*.o $(OUT_DIR)/$(OUT_FILE_NAME) Makefile.bak

rebuild: clean build