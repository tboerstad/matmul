# Compiler
CXX = clang++

# Include directory
INCLUDE_DIR = include

# Compiler flags
CXXFLAGS = -Xclang -fopenmp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp -std=c++17 -O3 -march=native -mtune=native -flto -ffast-math -funroll-loops -I$(INCLUDE_DIR)

# Source file
SRC = main.cpp

# Output binary
TARGET = myprogram

# Default target
all: $(TARGET)

# Compile the program
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

# Run the program
run: $(TARGET)
	./$(TARGET) --iters=10

# Clean up
clean:
	rm -f $(TARGET)

.PHONY: all run Clean