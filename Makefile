# Default target: print this help message
.PHONY: help
.DEFAULT_GOAL := help
help:
	@echo 'Usage:'
	@echo '  make <target>'
	@echo ''
	@echo 'Targets:'
	@sed -n 's/^##//p' $(MAKEFILE_LIST) | column -t -s ':' | sed -e 's/^/  /'

## genkit: Run Genkit locally
.PHONY: genkit
genkit:
	cd examples/openai && genkit start -o

## test: Test the Go modules within this package
.PHONY: test
test:
	go test -v ./...

## tidy: Tidy modfiles, format and lint .go files
.PHONY: tidy
tidy:
	go mod tidy -v
	go fmt ./...
	gofmt -s -w .
	golangci-lint run

## publish: Publish the Go modules to the registry
.PHONY: publish
publish:
	@if [ -z "$(version)" ]; then echo "Error: version is not set"; exit 1; fi
	git tag $(version)
	git push origin $(version)
	GOPROXY=proxy.golang.org go list -m github.com/yukinagae/genkit-go-plugins@$(version)
