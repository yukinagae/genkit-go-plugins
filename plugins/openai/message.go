package openai

import (
	"fmt"

	"github.com/firebase/genkit/go/ai"
	goopenai "github.com/sashabaranov/go-openai"
)

type SystemMessage struct {
	Content string `json:"content"`
	Name    string `json:"name,omitempty"`
}

func convertSystemMessage(msg *ai.Message) *SystemMessage {
	return &SystemMessage{
		Content: msg.Content[0].Text,
	}
}

func (m *SystemMessage) toMessage() goopenai.ChatCompletionMessage {
	return goopenai.ChatCompletionMessage{
		Role:    goopenai.ChatMessageRoleSystem,
		Content: m.Content,
	}
}

type UserMessage struct {
	Content []goopenai.ChatMessagePart `json:"content"`
	Name    string                     `json:"name,omitempty"`
}

func convertUserMessage(msg *ai.Message) (*UserMessage, error) {
	var multiContent []goopenai.ChatMessagePart
	for _, p := range msg.Content {
		part, err := convertPart(p)
		if err != nil {
			return nil, err
		}
		multiContent = append(multiContent, part)
	}
	return &UserMessage{
		Content: multiContent,
	}, nil
}

func (m *UserMessage) toMessage() goopenai.ChatCompletionMessage {
	return goopenai.ChatCompletionMessage{
		Role:         goopenai.ChatMessageRoleUser,
		MultiContent: m.Content,
	}
}

type AssistantMessage struct {
	ToolCalls []goopenai.ToolCall `json:"tool_calls"`
	Content   string              `json:"content"`
	Refusal   string              `json:"refusal,omitempty"`
	Name      string              `json:"name,omitempty"`
}

func convertAssistantMessage(msg *ai.Message) *AssistantMessage {
	toolCalls := convertToolCalls(msg.Content)
	return &AssistantMessage{
		ToolCalls: toolCalls,
		Content:   msg.Content[0].Text,
	}
}

func (m *AssistantMessage) toMessage() goopenai.ChatCompletionMessage {
	return goopenai.ChatCompletionMessage{
		Role:      goopenai.ChatMessageRoleAssistant,
		ToolCalls: m.ToolCalls,
		Content:   m.Content,
	}
}

type ToolMessage struct {
	Content    string `json:"content"`
	ToolCallID string `json:"tool_call_id"`
}

func convertToolMessage(part *ai.Part) (*ToolMessage, error) {
	if !part.IsToolResponse() {
		return nil, fmt.Errorf("part is not a tool response: %#v", part)
	}
	return &ToolMessage{
		Content: mapToJSONString(part.ToolResponse.Output),
		// NOTE: Temporarily set its name instead of its ref (i.e. call_xxxxx) since it's not defined in the ai.ToolResponse struct.
		ToolCallID: part.ToolResponse.Name,
	}, nil
}

func (m *ToolMessage) toMessage() goopenai.ChatCompletionMessage {
	return goopenai.ChatCompletionMessage{
		Role:       goopenai.ChatMessageRoleTool,
		ToolCallID: m.ToolCallID,
		Content:    m.Content,
	}
}
