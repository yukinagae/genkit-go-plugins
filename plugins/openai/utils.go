package openai

import (
	"encoding/json"
	"fmt"
)

func jsonStringToMap(jsonString string) map[string]any {
	var result map[string]any
	if err := json.Unmarshal([]byte(jsonString), &result); err != nil {
		panic(fmt.Errorf("unmarshal failed to parse json string %s: %w", jsonString, err))
	}
	return result
}

func mapToJSONString(data map[string]any) string {
	jsonBytes, err := json.Marshal(data)
	if err != nil {
		panic(fmt.Errorf("failed to marshal map to JSON string: data, %#v %w", data, err))
	}
	return string(jsonBytes)
}

func mapToJSONRawMessage(data map[string]any) (json.RawMessage, error) {
	jsonBytes, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal map to JSON string: data, %#v %w", data, err)
	}
	return json.RawMessage(jsonBytes), nil
}
