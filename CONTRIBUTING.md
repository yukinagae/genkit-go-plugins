# How to contribute

Thank you for your interest in contributing to this project! We appreciate your contributions.

## Environment Setup

- Go 1.22 or higher

## Running Tests

### Unit Tests

To run unit tests, use the following command:

```bash
make test
```

### Live Tests

To run live tests, use the following command:

```bash
go test -v ./... -key=[your-openai-api-key]
```

NOTE: You may encounter the error message: `Rate limit reached for gpt-4o-mini on requests per min (RPM)`. If this happens, increase your rate limit or run live tests one by one.

## Code Formatting

Ensure your code is properly formatted before committing. Use the following command:

```bash
make tidy
```
