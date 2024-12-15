# Compiler and Flags
CC = gcc
CFLAGS = -Wall -Wextra -pedantic -std=c11  -fopenmp -O3 -lm -lsodium

# Source Files
SRCS = demo_EnergyEfficiency.c scratchANN.c readData.c
OBJS = $(SRCS:.c=.o)

# Header Files
HEADERS = scratchANN.h

# Output Executable
TARGET = sample

# Default Rule
all: $(TARGET)

# Build the Target
$(TARGET): $(OBJS)
	$(CC) $(OBJS) $(CFLAGS) -o $(TARGET)

# Compile Source Files
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET)

# Phony Targets
.PHONY: all clean
