package openai

import (
	"reflect"
	"testing"

	"github.com/firebase/genkit/go/ai"
	goopenai "github.com/sashabaranov/go-openai"
)

func TestTranslateCandidate(t *testing.T) {
	tests := []struct {
		name  string
		input struct {
			choice   goopenai.ChatCompletionChoice
			jsonMode bool
		}
		want *ai.Candidate
	}{
		{
			name: "text",
			input: struct {
				choice   goopenai.ChatCompletionChoice
				jsonMode bool
			}{
				choice: goopenai.ChatCompletionChoice{
					Index: 0,
					Message: goopenai.ChatCompletionMessage{
						Role:    goopenai.ChatMessageRoleAssistant,
						Content: "Tell a joke about dogs.",
					},
					FinishReason: goopenai.FinishReasonLength,
				},
				jsonMode: false,
			},
			want: &ai.Candidate{
				Index:        0,
				FinishReason: ai.FinishReasonLength,
				Message: &ai.Message{
					Role:    ai.RoleModel,
					Content: []*ai.Part{ai.NewTextPart("Tell a joke about dogs.")},
				},
				Custom: nil,
			},
		},
		{
			name: "json",
			input: struct {
				choice   goopenai.ChatCompletionChoice
				jsonMode bool
			}{
				choice: goopenai.ChatCompletionChoice{
					Index: 0,
					Message: goopenai.ChatCompletionMessage{
						Role:    goopenai.ChatMessageRoleAssistant,
						Content: "{\"json\": \"test\"}",
					},
					FinishReason: goopenai.FinishReasonContentFilter,
				},
				jsonMode: true,
			},
			want: &ai.Candidate{
				Index:        0,
				FinishReason: ai.FinishReasonBlocked,
				Message: &ai.Message{
					Role:    ai.RoleModel,
					Content: []*ai.Part{ai.NewDataPart("{\"json\": \"test\"}")},
				},
				Custom: nil,
			},
		},
		{
			name: "tools",
			input: struct {
				choice   goopenai.ChatCompletionChoice
				jsonMode bool
			}{
				choice: goopenai.ChatCompletionChoice{
					Index: 0,
					Message: goopenai.ChatCompletionMessage{
						Role:    goopenai.ChatMessageRoleAssistant,
						Content: "Tool call",
						ToolCalls: []goopenai.ToolCall{
							goopenai.ToolCall{
								ID:   "exampleTool",
								Type: goopenai.ToolTypeFunction,
								Function: goopenai.FunctionCall{
									Name:      "exampleTool",
									Arguments: "{\"param\": \"value\"}",
								},
							},
						},
					},
					FinishReason: goopenai.FinishReasonToolCalls,
				},
				jsonMode: false,
			},
			want: &ai.Candidate{
				Index:        0,
				FinishReason: ai.FinishReasonStop,
				Message: &ai.Message{
					Role: ai.RoleModel,
					Content: []*ai.Part{ai.NewToolRequestPart(&ai.ToolRequest{
						Name: "exampleTool",
						Input: map[string]any{
							"param": "value",
						},
					})},
				},
				Custom: nil,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := translateCandidate(tt.input.choice, tt.input.jsonMode)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("convertRole() = %#v, want %#v", got, tt.want)
			}
		})
	}
}
