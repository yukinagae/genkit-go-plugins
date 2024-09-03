package openai

import (
	"github.com/firebase/genkit/go/ai"
	goopenai "github.com/openai/openai-go"
)

func translateResponse(resp *goopenai.ChatCompletion, jsonMode bool) *ai.GenerateResponse {
	r := &ai.GenerateResponse{}

	for _, c := range resp.Choices {
		r.Candidates = append(r.Candidates, translateCandidate(c, jsonMode))
	}

	r.Usage = &ai.GenerationUsage{
		InputTokens:  int(resp.Usage.PromptTokens),
		OutputTokens: int(resp.Usage.CompletionTokens),
		TotalTokens:  int(resp.Usage.TotalTokens),
	}
	r.Custom = resp
	return r
}

func translateCandidate(choice goopenai.ChatCompletionChoice, jsonMode bool) *ai.Candidate {
	c := &ai.Candidate{
		Index: int(choice.Index),
	}

	switch choice.FinishReason {
	case goopenai.ChatCompletionChoicesFinishReasonStop, goopenai.ChatCompletionChoicesFinishReasonToolCalls:
		c.FinishReason = ai.FinishReasonStop
	case goopenai.ChatCompletionChoicesFinishReasonLength:
		c.FinishReason = ai.FinishReasonLength
	case goopenai.ChatCompletionChoicesFinishReasonContentFilter:
		c.FinishReason = ai.FinishReasonBlocked
	case goopenai.ChatCompletionChoicesFinishReasonFunctionCall:
		c.FinishReason = ai.FinishReasonOther
	default:
		c.FinishReason = ai.FinishReasonUnknown
	}

	m := &ai.Message{
		Role: ai.RoleModel,
	}

	// handle tool calls
	var toolRequestParts []*ai.Part
	for _, toolCall := range choice.Message.ToolCalls {
		toolRequestParts = append(toolRequestParts, ai.NewToolRequestPart(&ai.ToolRequest{
			Name:  toolCall.Function.Name,
			Input: jsonStringToMap(toolCall.Function.Arguments),
		}))
	}
	if len(toolRequestParts) > 0 {
		m.Content = toolRequestParts
		c.Message = m
		return c
	}

	if jsonMode {
		m.Content = append(m.Content, ai.NewDataPart(choice.Message.Content))
	} else {
		m.Content = append(m.Content, ai.NewTextPart(choice.Message.Content))
	}

	c.Message = m
	return c
}
