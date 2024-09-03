package openai

import (
	"fmt"
	"slices"

	"github.com/firebase/genkit/go/ai"
	goopenai "github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"
)

func convertRequest(model string, input *ai.GenerateRequest) (goopenai.ChatCompletionNewParams, error) {
	messages, err := convertMessages(input.Messages)
	if err != nil {
		return goopenai.ChatCompletionNewParams{}, err
	}

	tools, err := convertTools(input.Tools)
	if err != nil {
		return goopenai.ChatCompletionNewParams{}, err
	}

	chatCompletionRequest := goopenai.ChatCompletionNewParams{
		Model:    goopenai.F(model),
		Messages: goopenai.F(messages),
	}

	if input.Candidates > 0 {
		chatCompletionRequest.N = goopenai.F(int64(input.Candidates))
	}

	if len(tools) > 0 {
		chatCompletionRequest.Tools = goopenai.F(tools)
	}

	if c, ok := input.Config.(*ai.GenerationCommonConfig); ok && c != nil {
		if c.MaxOutputTokens != 0 {
			chatCompletionRequest.MaxTokens = goopenai.F(int64(c.MaxOutputTokens))
		}
		if len(c.StopSequences) > 0 {
			chatCompletionRequest.Stop = goopenai.F[goopenai.ChatCompletionNewParamsStopUnion](goopenai.ChatCompletionNewParamsStopArray(c.StopSequences))
		}
		if c.Temperature != 0 {
			chatCompletionRequest.Temperature = goopenai.F(c.Temperature)
		}
		if c.TopP != 0 {
			chatCompletionRequest.TopP = goopenai.F(c.TopP)
		}
	}

	if input.Output != nil &&
		input.Output.Format != "" &&
		slices.Contains(modelsSupportingResponseFormats, model) {
		switch input.Output.Format {
		case ai.OutputFormatJSON:
			chatCompletionRequest.ResponseFormat = goopenai.F[goopenai.ChatCompletionNewParamsResponseFormatUnion](goopenai.ChatCompletionNewParamsResponseFormat{
				Type: goopenai.F(goopenai.ChatCompletionNewParamsResponseFormatTypeJSONObject),
			})
		case ai.OutputFormatText:
			chatCompletionRequest.ResponseFormat = goopenai.F[goopenai.ChatCompletionNewParamsResponseFormatUnion](goopenai.ChatCompletionNewParamsResponseFormat{
				Type: goopenai.F(goopenai.ChatCompletionNewParamsResponseFormatTypeText),
			})
		default:
			return goopenai.ChatCompletionNewParams{}, fmt.Errorf("unknown output format in a request: %s", input.Output.Format)
		}
	}

	return chatCompletionRequest, nil
}

func convertMessages(messages []*ai.Message) ([]goopenai.ChatCompletionMessageParamUnion, error) {
	var msgs []goopenai.ChatCompletionMessageParamUnion

	for _, m := range messages {
		role, err := convertRole(m.Role)
		if err != nil {
			return nil, err
		}
		switch role {
		case goopenai.ChatCompletionMessageParamRoleSystem: // system
			sm := goopenai.SystemMessage(m.Content[0].Text)
			msgs = append(msgs, sm)
		case goopenai.ChatCompletionMessageParamRoleUser: // user
			var multiContent []goopenai.ChatCompletionContentPartUnionParam
			for _, p := range m.Content {
				part, err := convertPart(p)
				if err != nil {
					return nil, err
				}
				multiContent = append(multiContent, part)
			}
			um := goopenai.UserMessageParts(multiContent...)
			msgs = append(msgs, um)
		case goopenai.ChatCompletionMessageParamRoleAssistant: // assistant
			am := goopenai.ChatCompletionAssistantMessageParam{
				Role: goopenai.F(goopenai.ChatCompletionAssistantMessageParamRoleAssistant),
			}
			if m.Content[0].Text != "" {
				am.Content = goopenai.F([]goopenai.ChatCompletionAssistantMessageParamContentUnion{
					goopenai.TextPart(m.Content[0].Text),
				})
			}
			toolCalls := convertToolCalls(m.Content)
			if len(toolCalls) > 0 {
				am.ToolCalls = goopenai.F(toolCalls)
			}
			msgs = append(msgs, am)
		case goopenai.ChatCompletionMessageParamRoleTool: // tool
			for _, p := range m.Content {
				if !p.IsToolResponse() {
					continue
				}
				tm := goopenai.ToolMessage(
					// NOTE: Temporarily set its name instead of its ref (i.e. call_xxxxx) since it's not defined in the ai.ToolResponse struct.
					p.ToolResponse.Name,
					mapToJSONString(p.ToolResponse.Output),
				)
				msgs = append(msgs, tm)
			}
		default:
			return nil, fmt.Errorf("Unknown OpenAI Role %s", role)
		}
	}

	return msgs, nil
}

func convertPart(part *ai.Part) (goopenai.ChatCompletionContentPartUnionParam, error) {
	switch {
	case part.IsText():
		return goopenai.TextPart(part.Text), nil
	case part.IsMedia():
		mediaPart := goopenai.ImagePart(part.Text)
		mediaPart.ImageURL.Value.Detail = goopenai.F(goopenai.ChatCompletionContentPartImageImageURLDetailAuto)
		return mediaPart, nil
	default:
		return nil, fmt.Errorf("unknown part type in a request: %#v", part)
	}
}

func convertToolCalls(content []*ai.Part) []goopenai.ChatCompletionMessageToolCallParam {
	var toolCalls []goopenai.ChatCompletionMessageToolCallParam
	for _, p := range content {
		if !p.IsToolRequest() {
			continue
		}
		toolCall := convertToolCall(p)
		toolCalls = append(toolCalls, toolCall)
	}
	return toolCalls
}

func convertToolCall(part *ai.Part) goopenai.ChatCompletionMessageToolCallParam {
	param := goopenai.ChatCompletionMessageToolCallParam{
		// NOTE: Temporarily set its name instead of its ref (i.e. call_xxxxx) since it's not defined in the ai.ToolRequest struct.
		ID:   goopenai.F(part.ToolRequest.Name),
		Type: goopenai.F(goopenai.ChatCompletionMessageToolCallTypeFunction),
		Function: goopenai.F(goopenai.ChatCompletionMessageToolCallFunctionParam{
			Name: goopenai.F(part.ToolRequest.Name),
		}),
	}

	if len(part.ToolRequest.Input) > 0 {
		param.Function.Value.Arguments = goopenai.F(mapToJSONString(part.ToolRequest.Input))
	}

	return param
}

func convertTools(inTools []*ai.ToolDefinition) ([]goopenai.ChatCompletionToolParam, error) {
	var tools []goopenai.ChatCompletionToolParam
	for _, t := range inTools {
		tool, err := convertTool(t)
		if err != nil {
			return nil, err
		}
		tools = append(tools, tool)
	}
	return tools, nil
}

func convertTool(t *ai.ToolDefinition) (goopenai.ChatCompletionToolParam, error) {
	return goopenai.ChatCompletionToolParam{
		Type: goopenai.F(goopenai.ChatCompletionToolTypeFunction),
		Function: goopenai.F(shared.FunctionDefinitionParam{
			Name:        goopenai.F(t.Name),
			Description: goopenai.F(t.Description),
			Parameters:  goopenai.F(goopenai.FunctionParameters(t.InputSchema)),
			Strict:      goopenai.F(false),
		}),
	}, nil
}

func convertRole(aiRole ai.Role) (goopenai.ChatCompletionMessageParamRole, error) {
	switch aiRole {
	case ai.RoleUser: // user -> user
		return goopenai.ChatCompletionMessageParamRoleUser, nil
	case ai.RoleSystem: // system -> system
		return goopenai.ChatCompletionMessageParamRoleSystem, nil
	case ai.RoleModel: // model -> assistant
		return goopenai.ChatCompletionMessageParamRoleAssistant, nil
	case ai.RoleTool: // tool -> tool
		return goopenai.ChatCompletionMessageParamRoleTool, nil
	default:
		return "", fmt.Errorf("Unknown ai.Role: %s", aiRole)
	}
}
