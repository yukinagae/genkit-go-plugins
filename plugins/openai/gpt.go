package openai

import "github.com/firebase/genkit/go/ai"

var (
	// BasicText describes model capabilities for text-only GPT models.
	BasicText = ai.ModelCapabilities{
		Multiturn:  true,
		Tools:      true,
		SystemRole: true,
		Media:      false,
	}

	//  Multimodal describes model capabilities for multimodal GPT models.
	Multimodal = ai.ModelCapabilities{
		Multiturn:  true,
		Tools:      true,
		SystemRole: true,
		Media:      true,
	}
)
