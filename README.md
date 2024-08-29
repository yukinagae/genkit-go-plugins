# Firebase Genkit Go Community Plugins

[![Go Reference](https://pkg.go.dev/badge/github.com/yukinagae/genkit-go-plugins.svg)](https://pkg.go.dev/github.com/yukinagae/genkit-go-plugins)
[![Go Report Card](https://goreportcard.com/badge/github.com/yukinagae/genkit-go-plugins)](https://goreportcard.com/report/github.com/yukinagae/genkit-go-plugins)
[![GitHub License](https://img.shields.io/github/license/yukinagae/genkit-go-plugins)](https://github.com/yukinagae/genkit-go-plugins/blob/main/LICENSE)

## Available plugins

### Model / Embedding Plugins

- [openai](https://github.com/yukinagae/genkit-go-plugins/openai) - Plugins for OpenAI APIs

## Installation

```bash
go get github.com/yukinagae/genkit-go-plugins@latest
```

## Usage

### OpenAI

Get your OpenAI API key from [here](https://platform.openai.com/account/api-keys) and run:

```bash
export OPENAI_API_KEY=your-api-key
```

Run `genkit start -o` with the following sample code:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"

	"github.com/yukinagae/genkit-go-plugins/plugins/openai"
)

func main() {
	ctx := context.Background()

	if err := openai.Init(ctx, nil); err != nil {
		log.Fatal(err)
	}

	genkit.DefineFlow("sampleFlow", func(ctx context.Context, input string) (string, error) {
		model := openai.Model("gpt-4o-mini")

		resp, err := model.Generate(ctx,
			ai.NewGenerateRequest(
				&ai.GenerationCommonConfig{Temperature: 1},
				ai.NewUserTextMessage("Hello!")),
			nil)
		if err != nil {
			return "", err
		}

		text := resp.Text()
		return text, nil
	})

	if err := genkit.Init(ctx, nil); err != nil {
		log.Fatal(err)
	}
}
```

For detailed information about Firebase Genkit, please refer to the official documentation: [Get started with Genkit using Go (alpha) ](https://firebase.google.com/docs/genkit-go/get-started-go)
