package openai

import (
	"github.com/firebase/genkit/go/ai"
	goopenai "github.com/sashabaranov/go-openai"
)

// Translate from a goopenai.ChatCompletionResponse to a ai.GenerateResponse.
func translateResponse(resp goopenai.ChatCompletionResponse, jsonMode bool) *ai.GenerateResponse {
	r := &ai.GenerateResponse{}

	for _, c := range resp.Choices {
		r.Candidates = append(r.Candidates, translateCandidate(c, jsonMode))
	}

	r.Usage = &ai.GenerationUsage{
		InputTokens:  resp.Usage.PromptTokens,
		OutputTokens: resp.Usage.CompletionTokens,
		TotalTokens:  resp.Usage.TotalTokens,
	}
	r.Custom = resp
	return r
}

// translateCandidate translates from a goopenai.ChatCompletionChoice to an ai.Candidate.
func translateCandidate(choice goopenai.ChatCompletionChoice, jsonMode bool) *ai.Candidate {
	c := &ai.Candidate{
		Index: choice.Index,
	}
	switch choice.FinishReason {
	case goopenai.FinishReasonStop, goopenai.FinishReasonToolCalls:
		c.FinishReason = ai.FinishReasonStop
	case goopenai.FinishReasonLength:
		c.FinishReason = ai.FinishReasonLength
	case goopenai.FinishReasonContentFilter:
		c.FinishReason = ai.FinishReasonBlocked
	case goopenai.FinishReasonFunctionCall:
		c.FinishReason = ai.FinishReasonOther
	case goopenai.FinishReasonNull:
		c.FinishReason = ai.FinishReasonUnknown
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
		m.Content = append(m.Content, ai.NewJSONPart(choice.Message.Content))
	} else {
		m.Content = append(m.Content, ai.NewTextPart(choice.Message.Content))
	}

	c.Message = m
	return c
}
