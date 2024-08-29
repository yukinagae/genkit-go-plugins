package openai

import (
	"fmt"

	"github.com/firebase/genkit/go/ai"
	goopenai "github.com/sashabaranov/go-openai"
)

type systemMessage struct {
	Content string `json:"content"`
	Name    string `json:"name,omitempty"`
}

func convertSystemMessage(msg *ai.Message) *systemMessage {
	return &systemMessage{
		Content: msg.Content[0].Text,
	}
}

func (m *systemMessage) toMessage() goopenai.ChatCompletionMessage {
	return goopenai.ChatCompletionMessage{
		Role:    goopenai.ChatMessageRoleSystem,
		Content: m.Content,
	}
}

type userMessage struct {
	Content []goopenai.ChatMessagePart `json:"content"`
	Name    string                     `json:"name,omitempty"`
}

func convertUserMessage(msg *ai.Message) (*userMessage, error) {
	var multiContent []goopenai.ChatMessagePart
	for _, p := range msg.Content {
		part, err := convertPart(p)
		if err != nil {
			return nil, err
		}
		multiContent = append(multiContent, part)
	}
	return &userMessage{
		Content: multiContent,
	}, nil
}

func (m *userMessage) toMessage() goopenai.ChatCompletionMessage {
	return goopenai.ChatCompletionMessage{
		Role:         goopenai.ChatMessageRoleUser,
		MultiContent: m.Content,
	}
}

type assistantMessage struct {
	ToolCalls []goopenai.ToolCall `json:"tool_calls"`
	Content   string              `json:"content"`
	Refusal   string              `json:"refusal,omitempty"`
	Name      string              `json:"name,omitempty"`
}

func convertAssistantMessage(msg *ai.Message) *assistantMessage {
	toolCalls := convertToolCalls(msg.Content)
	return &assistantMessage{
		ToolCalls: toolCalls,
		Content:   msg.Content[0].Text,
	}
}

func (m *assistantMessage) toMessage() goopenai.ChatCompletionMessage {
	return goopenai.ChatCompletionMessage{
		Role:      goopenai.ChatMessageRoleAssistant,
		ToolCalls: m.ToolCalls,
		Content:   m.Content,
	}
}

type toolMessage struct {
	Content    string `json:"content"`
	ToolCallID string `json:"tool_call_id"`
}

func convertToolMessage(part *ai.Part) (*toolMessage, error) {
	if !part.IsToolResponse() {
		return nil, fmt.Errorf("part is not a tool response: %#v", part)
	}
	return &toolMessage{
		Content: mapToJSONString(part.ToolResponse.Output),
		// NOTE: Temporarily set its name instead of its ref (i.e. call_xxxxx) since it's not defined in the ai.ToolResponse struct.
		ToolCallID: part.ToolResponse.Name,
	}, nil
}

func (m *toolMessage) toMessage() goopenai.ChatCompletionMessage {
	return goopenai.ChatCompletionMessage{
		Role:       goopenai.ChatMessageRoleTool,
		ToolCallID: m.ToolCallID,
		Content:    m.Content,
	}
}
