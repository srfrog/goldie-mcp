.PHONY: build clean test run deps

BINARY_NAME=goldie-mcp
BUILD_DIR=build
UNAME_S := $(shell uname -s)

# Ad-hoc codesign on macOS to avoid Gatekeeper killing the binary
define codesign_macos
	$(if $(filter Darwin,$(UNAME_S)),codesign -s - -f $(1),)
endef

# Build with CGO for sqlite-vec
build:
	CGO_ENABLED=1 go build -o $(BINARY_NAME) .
	$(call codesign_macos,$(BINARY_NAME))

# Build for release (with optimizations)
release:
	CGO_ENABLED=1 go build -ldflags="-s -w" -o $(BINARY_NAME) .
	$(call codesign_macos,$(BINARY_NAME))

# Build for specific platforms (requires cross-compilation toolchain)
build-linux:
	CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -o $(BUILD_DIR)/$(BINARY_NAME)-linux-amd64 .

build-darwin:
	CGO_ENABLED=1 GOOS=darwin GOARCH=amd64 go build -o $(BUILD_DIR)/$(BINARY_NAME)-darwin-amd64 .
	CGO_ENABLED=1 GOOS=darwin GOARCH=arm64 go build -o $(BUILD_DIR)/$(BINARY_NAME)-darwin-arm64 .
	$(call codesign_macos,$(BUILD_DIR)/$(BINARY_NAME)-darwin-amd64)
	$(call codesign_macos,$(BUILD_DIR)/$(BINARY_NAME)-darwin-arm64)

# Get dependencies
deps:
	go mod download
	go mod tidy

# Run the server (for testing)
run:
	CGO_ENABLED=1 go run .

# Clean build artifacts
clean:
	rm -f $(BINARY_NAME)
	rm -rf $(BUILD_DIR)

# Run tests
test:
	CGO_ENABLED=1 go test -v -race ./...

# Format code
fmt:
	go fmt ./...

# Lint code
lint:
	golangci-lint run

# Install the binary to GOPATH/bin, or,
# install to a specific directory (avoids xattr issues on macOS)
# Usage: make install DEST=~/bin
install:
ifndef DEST
	@echo "Installing to GOPATH/bin ..."
	CGO_ENABLED=1 go install .
else
	@echo "Installing to $(DEST) ..."
	CGO_ENABLED=1 go build -ldflags="-s -w" -o $(DEST)/$(BINARY_NAME) .
	$(call codesign_macos,$(DEST)/$(BINARY_NAME))
endif
