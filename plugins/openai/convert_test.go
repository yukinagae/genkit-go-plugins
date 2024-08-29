package openai

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/firebase/genkit/go/ai"
	goopenai "github.com/sashabaranov/go-openai"
)

func TestConvertRole(t *testing.T) {
	tests := []struct {
		name  string
		input ai.Role
		want  string
	}{
		{
			name:  "system -> system",
			input: ai.RoleSystem,
			want:  goopenai.ChatMessageRoleSystem,
		},
		{
			name:  "user -> user",
			input: ai.RoleUser,
			want:  goopenai.ChatMessageRoleUser,
		},
		{
			name:  "model -> assistant",
			input: ai.RoleModel,
			want:  goopenai.ChatMessageRoleAssistant,
		},
		{
			name:  "tool role",
			input: ai.RoleTool,
			want:  goopenai.ChatMessageRoleTool,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := convertRole(tt.input)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("convertRole() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestConvertPart(t *testing.T) {
	tests := []struct {
		name  string
		input *ai.Part
		want  goopenai.ChatMessagePart
	}{
		{
			name:  "text part",
			input: ai.NewTextPart("hi"),
			want: goopenai.ChatMessagePart{
				Type: goopenai.ChatMessagePartTypeText,
				Text: "hi",
			},
		},
		{
			name:  "media part",
			input: ai.NewMediaPart("image/jpeg", "https://example.com/image.jpg"),
			want: goopenai.ChatMessagePart{
				Type: goopenai.ChatMessagePartTypeImageURL,
				ImageURL: &goopenai.ChatMessageImageURL{
					URL:    "https://example.com/image.jpg",
					Detail: goopenai.ImageURLDetailAuto,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := convertPart(tt.input)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("convertPart() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestConvertMessages(t *testing.T) {
	tests := []struct {
		name  string
		input []*ai.Message
		want  []goopenai.ChatCompletionMessage
	}{
		{
			name: "tool request",
			input: []*ai.Message{
				{
					Role: ai.RoleModel,
					Content: []*ai.Part{ai.NewToolRequestPart(
						&ai.ToolRequest{
							Name: "tellAFunnyJoke",
							Input: map[string]any{
								"topic": "bob",
							},
						},
					)},
				},
			},
			want: []goopenai.ChatCompletionMessage{
				{
					Role: goopenai.ChatMessageRoleAssistant,
					ToolCalls: []goopenai.ToolCall{
						{
							ID:   "tellAFunnyJoke",
							Type: goopenai.ToolTypeFunction,
							Function: goopenai.FunctionCall{
								Name:      "tellAFunnyJoke",
								Arguments: "{\"topic\":\"bob\"}",
							},
						},
					},
				},
			},
		},
		{
			name: "tool response",
			input: []*ai.Message{
				{
					Role: ai.RoleTool,
					Content: []*ai.Part{ai.NewToolResponsePart(
						&ai.ToolResponse{
							Name: "tellAFunnyJoke",
							Output: map[string]any{
								"joke": "Why did the bob cross the road?",
							},
						},
					)},
				},
			},
			want: []goopenai.ChatCompletionMessage{
				{
					Role:       goopenai.ChatMessageRoleTool,
					ToolCallID: "tellAFunnyJoke",
					Content:    "{\"joke\":\"Why did the bob cross the road?\"}",
				},
			},
		},
		{
			name: "text",
			input: []*ai.Message{
				{
					Role:    ai.RoleUser,
					Content: []*ai.Part{ai.NewTextPart("hi")},
				},
				{
					Role:    ai.RoleModel,
					Content: []*ai.Part{ai.NewTextPart("how can I help you?")},
				},
				{
					Role:    ai.RoleUser,
					Content: []*ai.Part{ai.NewTextPart("I am testing")},
				},
			},
			want: []goopenai.ChatCompletionMessage{
				{
					Role: goopenai.ChatMessageRoleUser,
					MultiContent: []goopenai.ChatMessagePart{
						{
							Type: goopenai.ChatMessagePartTypeText,
							Text: "hi",
						},
					},
				},
				{
					Role:    goopenai.ChatMessageRoleAssistant,
					Content: "how can I help you?",
				},
				{
					Role: goopenai.ChatMessageRoleUser,
					MultiContent: []goopenai.ChatMessagePart{
						{
							Type: goopenai.ChatMessagePartTypeText,
							Text: "I am testing",
						},
					},
				},
			},
		},
		{
			name: "multi-modal (text + media)",
			input: []*ai.Message{
				{
					Role: ai.RoleUser,
					Content: []*ai.Part{
						ai.NewTextPart("describe the following image:"),
						ai.NewMediaPart("image/jpeg", "https://example.com/image.jpg"),
					},
				},
			},
			want: []goopenai.ChatCompletionMessage{
				{
					Role: goopenai.ChatMessageRoleUser,
					MultiContent: []goopenai.ChatMessagePart{
						{
							Type: goopenai.ChatMessagePartTypeText,
							Text: "describe the following image:",
						},
						{
							Type: goopenai.ChatMessagePartTypeImageURL,
							ImageURL: &goopenai.ChatMessageImageURL{
								URL:    "https://example.com/image.jpg",
								Detail: goopenai.ImageURLDetailAuto,
							},
						},
					},
				},
			},
		},
		{
			name: "system message",
			input: []*ai.Message{
				{
					Role:    ai.RoleSystem,
					Content: []*ai.Part{ai.NewTextPart("system message")},
				},
			},
			want: []goopenai.ChatCompletionMessage{
				{
					Role:    goopenai.ChatMessageRoleSystem,
					Content: "system message",
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := convertMessages(tt.input)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("convertMessages() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestConvertToolCall(t *testing.T) {
	tests := []struct {
		name  string
		input *ai.Part
		want  goopenai.ToolCall
	}{
		{
			name: "tool call",
			input: ai.NewToolRequestPart(
				&ai.ToolRequest{
					Name: "tellAFunnyJoke",
					Input: map[string]any{
						"topic": "bob",
					},
				},
			),
			want: goopenai.ToolCall{
				ID:   "tellAFunnyJoke",
				Type: goopenai.ToolTypeFunction,
				Function: goopenai.FunctionCall{
					Name:      "tellAFunnyJoke",
					Arguments: "{\"topic\":\"bob\"}",
				},
			},
		},
		{
			name: "tool call with empty input",
			input: ai.NewToolRequestPart(
				&ai.ToolRequest{
					Name:  "tellAFunnyJoke",
					Input: map[string]any{},
				},
			),
			want: goopenai.ToolCall{
				ID:   "tellAFunnyJoke",
				Type: goopenai.ToolTypeFunction,
				Function: goopenai.FunctionCall{
					Name:      "tellAFunnyJoke",
					Arguments: "",
				},
			},
		},
		{
			name: "tool call with nil input",
			input: ai.NewToolRequestPart(
				&ai.ToolRequest{
					Name:  "tellAFunnyJoke",
					Input: nil,
				},
			),
			want: goopenai.ToolCall{
				ID:   "tellAFunnyJoke",
				Type: goopenai.ToolTypeFunction,
				Function: goopenai.FunctionCall{
					Name:      "tellAFunnyJoke",
					Arguments: "",
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := convertToolCall(tt.input)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("convertToolCall() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestConvertTool(t *testing.T) {
	tests := []struct {
		name  string
		input *ai.ToolDefinition
		want  goopenai.Tool
	}{
		{
			name: "text part",
			input: &ai.ToolDefinition{
				Name:        "tellAFunnyJoke",
				Description: "use when want to tell a funny joke",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"topic": map[string]any{
							"type": "string",
						},
					},
					"required":             []string{"topic"},
					"additionalProperties": false,
					"$schema":              "http://json-schema.org/draft-07/schema#",
				},
				OutputSchema: map[string]any{
					"type":    "string",
					"$schema": "http://json-schema.org/draft-07/schema#",
				},
			},
			want: goopenai.Tool{
				Type: goopenai.ToolTypeFunction,
				Function: &goopenai.FunctionDefinition{
					Name:        "tellAFunnyJoke",
					Description: "use when want to tell a funny joke",
					Strict:      false,
					Parameters:  json.RawMessage("{\"$schema\":\"http://json-schema.org/draft-07/schema#\",\"additionalProperties\":false,\"properties\":{\"topic\":{\"type\":\"string\"}},\"required\":[\"topic\"],\"type\":\"object\"}"),
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := convertTool(tt.input)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("convertTool() = %#v, want %#v", got, tt.want)
				t.Errorf("convertTool() = %q, want %q", string(got.Function.Parameters.(json.RawMessage)), string(tt.want.Function.Parameters.(json.RawMessage)))
			}
		})
	}
}

func TestConvertRequest(t *testing.T) {
	tests := []struct {
		name  string
		input struct {
			model string
			req   *ai.GenerateRequest
		}
		want goopenai.ChatCompletionRequest
	}{
		{
			name: "request with text messages",
			input: struct {
				model string
				req   *ai.GenerateRequest
			}{
				model: goopenai.GPT4o,
				req: &ai.GenerateRequest{
					Messages: []*ai.Message{
						{
							Role:    ai.RoleUser,
							Content: []*ai.Part{ai.NewTextPart("Tell a joke about dogs.")},
						},
					},
					Candidates: 3,
					Config: &ai.GenerationCommonConfig{
						MaxOutputTokens: 10,
						StopSequences:   []string{"\n"},
						Temperature:     0.7,
						TopP:            0.9,
					},
					Output: &ai.GenerateRequestOutput{
						Format: ai.OutputFormatText,
					},
				},
			},
			want: goopenai.ChatCompletionRequest{
				Model: goopenai.GPT4o,
				Messages: []goopenai.ChatCompletionMessage{
					{
						Role: goopenai.ChatMessageRoleUser,
						MultiContent: []goopenai.ChatMessagePart{
							{
								Type: goopenai.ChatMessagePartTypeText,
								Text: "Tell a joke about dogs.",
							},
						},
					},
				},
				ResponseFormat: &goopenai.ChatCompletionResponseFormat{
					Type: goopenai.ChatCompletionResponseFormatTypeText,
				},
				N:           3,
				MaxTokens:   10,
				Stop:        []string{"\n"},
				Temperature: 0.7,
				TopP:        0.9,
			},
		},
		{
			name: "request with text messages and tools",
			input: struct {
				model string
				req   *ai.GenerateRequest
			}{
				model: goopenai.GPT4o,
				req: &ai.GenerateRequest{
					Messages: []*ai.Message{
						{
							Role:    ai.RoleUser,
							Content: []*ai.Part{ai.NewTextPart("Tell a joke about dogs.")},
						},
						{
							Role: ai.RoleModel,
							Content: []*ai.Part{ai.NewToolRequestPart(
								&ai.ToolRequest{
									Name: "tellAFunnyJoke",
									Input: map[string]any{
										"topic": "dogs",
									},
								},
							)},
						},
						{
							Role: ai.RoleTool,
							Content: []*ai.Part{ai.NewToolResponsePart(
								&ai.ToolResponse{
									Name: "tellAFunnyJoke",
									Output: map[string]any{
										"joke": "Why did the dogs cross the road?",
									},
								},
							)},
						},
					},
					Tools: []*ai.ToolDefinition{
						{
							Name:        "tellAFunnyJoke",
							Description: "Tells jokes about an input topic. Use this tool whenever user asks you to tell a joke.",
							InputSchema: map[string]any{
								"type": "object",
								"properties": map[string]any{
									"topic": map[string]any{
										"type": "string",
									},
								},
								"required":             []string{"topic"},
								"additionalProperties": false,
								"$schema":              "http://json-schema.org/draft-07/schema#",
							},
							OutputSchema: map[string]any{
								"type":    "string",
								"$schema": "http://json-schema.org/draft-07/schema#",
							},
						},
					},
					Output: &ai.GenerateRequestOutput{
						Format: ai.OutputFormatText,
					},
				},
			},
			want: goopenai.ChatCompletionRequest{
				Model: goopenai.GPT4o,
				Messages: []goopenai.ChatCompletionMessage{
					{
						Role: goopenai.ChatMessageRoleUser,
						MultiContent: []goopenai.ChatMessagePart{
							{
								Type: goopenai.ChatMessagePartTypeText,
								Text: "Tell a joke about dogs.",
							},
						},
					},
					{
						Role: goopenai.ChatMessageRoleAssistant,
						ToolCalls: []goopenai.ToolCall{
							{
								ID:   "tellAFunnyJoke",
								Type: goopenai.ToolTypeFunction,
								Function: goopenai.FunctionCall{
									Name:      "tellAFunnyJoke",
									Arguments: "{\"topic\":\"dogs\"}",
								},
							},
						},
					},
					{
						Role:       goopenai.ChatMessageRoleTool,
						ToolCallID: "tellAFunnyJoke",
						Content:    "{\"joke\":\"Why did the dogs cross the road?\"}",
					},
				},
				Tools: []goopenai.Tool{
					{
						Type: goopenai.ToolTypeFunction,
						Function: &goopenai.FunctionDefinition{
							Name:        "tellAFunnyJoke",
							Description: "Tells jokes about an input topic. Use this tool whenever user asks you to tell a joke.",
							Parameters:  json.RawMessage("{\"$schema\":\"http://json-schema.org/draft-07/schema#\",\"additionalProperties\":false,\"properties\":{\"topic\":{\"type\":\"string\"}},\"required\":[\"topic\"],\"type\":\"object\"}"),
						},
					},
				},
				ResponseFormat: &goopenai.ChatCompletionResponseFormat{
					Type: goopenai.ChatCompletionResponseFormatTypeText,
				},
			},
		},
		{
			name: "request with structured output: json",
			input: struct {
				model string
				req   *ai.GenerateRequest
			}{
				model: goopenai.GPT4o,
				req: &ai.GenerateRequest{
					Messages: []*ai.Message{
						{
							Role:    ai.RoleUser,
							Content: []*ai.Part{ai.NewTextPart("Tell a joke about dogs.")},
						},
						{
							Role: ai.RoleModel,
							Content: []*ai.Part{ai.NewToolRequestPart(
								&ai.ToolRequest{
									Name: "tellAFunnyJoke",
									Input: map[string]any{
										"topic": "dogs",
									},
								},
							)},
						},
						{
							Role: ai.RoleTool,
							Content: []*ai.Part{ai.NewToolResponsePart(
								&ai.ToolResponse{
									Name: "tellAFunnyJoke",
									Output: map[string]any{
										"joke": "Why did the dogs cross the road?",
									},
								},
							)},
						},
					},
					Tools: []*ai.ToolDefinition{
						{
							Name:        "tellAFunnyJoke",
							Description: "Tells jokes about an input topic. Use this tool whenever user asks you to tell a joke.",
							InputSchema: map[string]any{
								"type": "object",
								"properties": map[string]any{
									"topic": map[string]any{
										"type": "string",
									},
								},
								"required":             []string{"topic"},
								"additionalProperties": false,
								"$schema":              "http://json-schema.org/draft-07/schema#",
							},
							OutputSchema: map[string]any{
								"type":    "string",
								"$schema": "http://json-schema.org/draft-07/schema#",
							},
						},
					},
					Output: &ai.GenerateRequestOutput{
						Format: ai.OutputFormatJSON,
					},
				},
			},
			want: goopenai.ChatCompletionRequest{
				Model: goopenai.GPT4o,
				Messages: []goopenai.ChatCompletionMessage{
					{
						Role: goopenai.ChatMessageRoleUser,
						MultiContent: []goopenai.ChatMessagePart{
							{
								Type: goopenai.ChatMessagePartTypeText,
								Text: "Tell a joke about dogs.",
							},
						},
					},
					{
						Role: goopenai.ChatMessageRoleAssistant,
						ToolCalls: []goopenai.ToolCall{
							{
								ID:   "tellAFunnyJoke",
								Type: goopenai.ToolTypeFunction,
								Function: goopenai.FunctionCall{
									Name:      "tellAFunnyJoke",
									Arguments: "{\"topic\":\"dogs\"}",
								},
							},
						},
					},
					{
						Role:       goopenai.ChatMessageRoleTool,
						ToolCallID: "tellAFunnyJoke",
						Content:    "{\"joke\":\"Why did the dogs cross the road?\"}",
					},
				},
				Tools: []goopenai.Tool{
					{
						Type: goopenai.ToolTypeFunction,
						Function: &goopenai.FunctionDefinition{
							Name:        "tellAFunnyJoke",
							Description: "Tells jokes about an input topic. Use this tool whenever user asks you to tell a joke.",
							Parameters:  json.RawMessage("{\"$schema\":\"http://json-schema.org/draft-07/schema#\",\"additionalProperties\":false,\"properties\":{\"topic\":{\"type\":\"string\"}},\"required\":[\"topic\"],\"type\":\"object\"}"),
						},
					},
				},
				ResponseFormat: &goopenai.ChatCompletionResponseFormat{
					Type: goopenai.ChatCompletionResponseFormatTypeJSONObject,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := convertRequest(tt.input.model, tt.input.req)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("convertRequest() = %#v, want %#v", got, tt.want)
				t.Errorf("convertRequest() = %q, want %q", string(got.Tools[0].Function.Parameters.(json.RawMessage)), string(tt.want.Tools[0].Function.Parameters.(json.RawMessage)))
			}
		})
	}
}
