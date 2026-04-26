.PHONY: help build release build-linux build-darwin deps run clean test fmt lint install

BINARY_NAME=goldie-mcp
BUILD_DIR=build
UNAME_S := $(shell uname -s)

# Show this help when `make` is run with no arguments.
.DEFAULT_GOAL := help

# Ad-hoc codesign on macOS to avoid Gatekeeper killing the binary
define codesign_macos
	$(if $(filter Darwin,$(UNAME_S)),codesign -s - -f $(1),)
endef

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "; printf "Goldie MCP — make targets\n\nUsage: make <target>\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build the goldie-mcp binary in the project root (CGO + ad-hoc codesign on macOS)
	CGO_ENABLED=1 go build -o $(BINARY_NAME) .
	$(call codesign_macos,$(BINARY_NAME))

release: ## Build with optimizations (-s -w) for smaller binaries
	CGO_ENABLED=1 go build -ldflags="-s -w" -o $(BINARY_NAME) .
	$(call codesign_macos,$(BINARY_NAME))

build-linux: ## Cross-compile for Linux amd64 into ./build/
	CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -o $(BUILD_DIR)/$(BINARY_NAME)-linux-amd64 .

build-darwin: ## Cross-compile for macOS (amd64 + arm64) into ./build/
	CGO_ENABLED=1 GOOS=darwin GOARCH=amd64 go build -o $(BUILD_DIR)/$(BINARY_NAME)-darwin-amd64 .
	CGO_ENABLED=1 GOOS=darwin GOARCH=arm64 go build -o $(BUILD_DIR)/$(BINARY_NAME)-darwin-arm64 .
	$(call codesign_macos,$(BUILD_DIR)/$(BINARY_NAME)-darwin-amd64)
	$(call codesign_macos,$(BUILD_DIR)/$(BINARY_NAME)-darwin-arm64)

deps: ## Download and tidy Go module dependencies
	go mod download
	go mod tidy

run: ## Run the server in the foreground (for ad-hoc testing)
	CGO_ENABLED=1 go run .

clean: ## Remove built binaries and the build/ directory
	rm -f $(BINARY_NAME)
	rm -rf $(BUILD_DIR)

test: ## Run the test suite with -race
	CGO_ENABLED=1 go test -v -race ./...

fmt: ## Format Go sources
	go fmt ./...

lint: ## Run golangci-lint
	golangci-lint run

# Install the binary to GOPATH/bin, or install to a specific directory
# (avoids macOS xattr issues from copying signed binaries around).
# Usage: make install DEST=~/bin
install: ## Install to GOPATH/bin, or to DEST=<dir> (recommended on macOS)
ifndef DEST
	@echo "Installing to GOPATH/bin ..."
	CGO_ENABLED=1 go install .
else
	@echo "Installing to $(DEST) ..."
	CGO_ENABLED=1 go build -ldflags="-s -w" -o $(DEST)/$(BINARY_NAME) .
	$(call codesign_macos,$(DEST)/$(BINARY_NAME))
endif
