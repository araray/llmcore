# Plain Makefile fallback (no CMake). Requires pkg-config + libcurl + libcjson.
#   apt install libcurl4-openssl-dev libcjson-dev
CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
CFLAGS += -Iinclude $(shell pkg-config --cflags libcurl libcjson)
LDLIBS += $(shell pkg-config --libs libcurl libcjson)
BUILD := build
SRC := src/llmcore_client.c
HDR := include/llmcore_client.h

all: $(BUILD)/quickstart $(BUILD)/e2e

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/llmcore_client.o: $(SRC) $(HDR) | $(BUILD)
	$(CC) $(CFLAGS) -c $(SRC) -o $@

$(BUILD)/quickstart: examples/quickstart.c $(BUILD)/llmcore_client.o | $(BUILD)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD)/e2e: tests/e2e.c $(BUILD)/llmcore_client.o | $(BUILD)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

test: $(BUILD)/e2e
	./$(BUILD)/e2e

clean:
	rm -rf $(BUILD)

.PHONY: all test clean
