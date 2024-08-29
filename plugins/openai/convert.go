package openai

import (
	"fmt"
	"slices"

	"github.com/firebase/genkit/go/ai"
	goopenai "github.com/sashabaranov/go-openai"
)

func convertRequest(model string, input *ai.GenerateRequest) (goopenai.ChatCompletionRequest, error) {
	messages, err := convertMessages(input.Messages)
	if err != nil {
		return goopenai.ChatCompletionRequest{}, err
	}

	tools, err := convertTools(input.Tools)
	if err != nil {
		return goopenai.ChatCompletionRequest{}, err
	}

	chatCompletionRequest := goopenai.ChatCompletionRequest{
		Model:    model,
		Messages: messages,
		Tools:    tools,
		N:        input.Candidates,
	}

	if c, ok := input.Config.(*ai.GenerationCommonConfig); ok && c != nil {
		if c.MaxOutputTokens != 0 {
			chatCompletionRequest.MaxTokens = c.MaxOutputTokens
		}
		if len(c.StopSequences) > 0 {
			chatCompletionRequest.Stop = c.StopSequences
		}
		if c.Temperature != 0 {
			chatCompletionRequest.Temperature = float32(c.Temperature)
		}
		if c.TopP != 0 {
			chatCompletionRequest.TopP = float32(c.TopP)
		}
	}

	if input.Output != nil &&
		input.Output.Format != "" &&
		slices.Contains(modelsSupportingResponseFormats, model) {
		switch input.Output.Format {
		case ai.OutputFormatJSON:
			chatCompletionRequest.ResponseFormat = &goopenai.ChatCompletionResponseFormat{
				Type: goopenai.ChatCompletionResponseFormatTypeJSONObject,
			}
		case ai.OutputFormatText:
			chatCompletionRequest.ResponseFormat = &goopenai.ChatCompletionResponseFormat{
				Type: goopenai.ChatCompletionResponseFormatTypeText,
			}
		default:
			return goopenai.ChatCompletionRequest{}, fmt.Errorf("unknown output format in a request: %s", input.Output.Format)
		}
	}

	return chatCompletionRequest, nil
}

func convertMessages(messages []*ai.Message) ([]goopenai.ChatCompletionMessage, error) {
	var msgs []goopenai.ChatCompletionMessage
	for _, m := range messages {
		role, err := convertRole(m.Role)
		if err != nil {
			return nil, err
		}
		switch role {
		case goopenai.ChatMessageRoleSystem: // system
			sm := convertSystemMessage(m)
			msgs = append(msgs, sm.toMessage())
		case goopenai.ChatMessageRoleUser: // user
			um, err := convertUserMessage(m)
			if err != nil {
				return nil, err
			}
			msgs = append(msgs, um.toMessage())
		case goopenai.ChatMessageRoleAssistant: // assistant
			am := convertAssistantMessage(m)
			msgs = append(msgs, am.toMessage())
		case goopenai.ChatMessageRoleTool: // tool
			for _, p := range m.Content {
				if !p.IsToolResponse() {
					continue
				}
				tm, err := convertToolMessage(p)
				if err != nil {
					return nil, err
				}
				msgs = append(msgs, tm.toMessage())
			}
		default:
			return nil, fmt.Errorf("Unknown OpenAI Role %s", role)
		}
	}
	return msgs, nil
}

func convertToolCalls(content []*ai.Part) []goopenai.ToolCall {
	var toolCalls []goopenai.ToolCall
	for _, p := range content {
		if !p.IsToolRequest() {
			continue
		}
		toolCall := convertToolCall(p)
		toolCalls = append(toolCalls, toolCall)
	}
	return toolCalls
}

func convertToolCall(part *ai.Part) goopenai.ToolCall {
	arguments := ""
	if len(part.ToolRequest.Input) > 0 {
		arguments = mapToJSONString(part.ToolRequest.Input)
	}
	return goopenai.ToolCall{
		// NOTE: Temporarily set its name instead of its ref (i.e. call_xxxxx) since it's not defined in the ai.ToolRequest struct.
		ID:   part.ToolRequest.Name,
		Type: goopenai.ToolTypeFunction,
		Function: goopenai.FunctionCall{
			Name:      part.ToolRequest.Name,
			Arguments: arguments,
		},
	}
}

func convertRole(aiRole ai.Role) (string, error) {
	switch aiRole {
	case ai.RoleUser: // user -> user
		return goopenai.ChatMessageRoleUser, nil
	case ai.RoleSystem: // system -> system
		return goopenai.ChatMessageRoleSystem, nil
	case ai.RoleModel: // model -> assistant
		return goopenai.ChatMessageRoleAssistant, nil
	case ai.RoleTool: // tool -> tool
		return goopenai.ChatMessageRoleTool, nil
	default:
		return "", fmt.Errorf("Unknown ai.Role: %s", aiRole)
	}
}

func convertPart(part *ai.Part) (goopenai.ChatMessagePart, error) {
	switch {
	case part.IsText():
		return goopenai.ChatMessagePart{
			Type: goopenai.ChatMessagePartTypeText,
			Text: part.Text,
		}, nil
	case part.IsMedia():
		return goopenai.ChatMessagePart{
			Type: goopenai.ChatMessagePartTypeImageURL,
			ImageURL: &goopenai.ChatMessageImageURL{
				URL:    part.Text,
				Detail: goopenai.ImageURLDetailAuto,
			},
		}, nil
	default:
		return goopenai.ChatMessagePart{}, fmt.Errorf("unknown part type in a request: %#v", part)
	}
}

func convertTools(inTools []*ai.ToolDefinition) ([]goopenai.Tool, error) {
	var tools []goopenai.Tool
	for _, t := range inTools {
		tool, err := convertTool(t)
		if err != nil {
			return nil, err
		}
		tools = append(tools, tool)
	}
	return tools, nil
}

func convertTool(t *ai.ToolDefinition) (goopenai.Tool, error) {
	parameters, err := mapToJSONRawMessage(t.InputSchema)
	if err != nil {
		return goopenai.Tool{}, err
	}
	return goopenai.Tool{
		Type: goopenai.ToolTypeFunction,
		Function: &goopenai.FunctionDefinition{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  parameters,
		},
	}, nil
}
