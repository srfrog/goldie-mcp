// Package minilm provides text embeddings using the all-MiniLM-L6-v2 model via ONNX runtime.
package minilm

import (
	"bytes"
	"fmt"
	"os"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	ort "github.com/yalue/onnxruntime_go"
)

const (
	// Dimensions is the output embedding dimension for all-MiniLM-L6-v2
	Dimensions = 384
)

// MiniLM provides text embeddings using the all-MiniLM-L6-v2 ONNX model.
type MiniLM struct {
	tokenizer tokenizer.Tokenizer
	session   *ort.DynamicAdvancedSession
}

// New creates a new MiniLM embedder.
// runtimePath is optional - if empty, uses ONNXRUNTIME_LIB_PATH env var.
func New(runtimePath string) (*MiniLM, error) {
	// Load tokenizer
	tk, err := pretrained.FromReader(bytes.NewBuffer(tokenizerData))
	if err != nil {
		return nil, fmt.Errorf("loading tokenizer: %w", err)
	}

	// Set ONNX Runtime library path
	if runtimePath != "" {
		ort.SetSharedLibraryPath(runtimePath)
	} else if path := os.Getenv("ONNXRUNTIME_LIB_PATH"); path != "" {
		ort.SetSharedLibraryPath(path)
	}

	// Initialize ONNX Runtime
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("initializing ONNX runtime: %w", err)
	}

	// Create session with dynamic input shapes
	inputNames := []string{"input_ids", "attention_mask", "token_type_ids"}
	outputNames := []string{"sentence_embedding"}

	session, err := ort.NewDynamicAdvancedSessionWithONNXData(modelData, inputNames, outputNames, nil)
	if err != nil {
		return nil, fmt.Errorf("creating ONNX session: %w", err)
	}

	return &MiniLM{
		tokenizer: *tk,
		session:   session,
	}, nil
}

// Embed generates an embedding vector for a single text.
func (m *MiniLM) Embed(text string) ([]float32, error) {
	results, err := m.EmbedBatch([]string{text})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, nil
	}
	return results[0], nil
}

// EmbedBatch generates embedding vectors for multiple texts.
func (m *MiniLM) EmbedBatch(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	// Tokenize
	inputs := make([]tokenizer.EncodeInput, len(texts))
	for i, t := range texts {
		inputs[i] = tokenizer.NewSingleEncodeInput(tokenizer.NewRawInputSequence(t))
	}

	encodings, err := m.tokenizer.EncodeBatch(inputs, true)
	if err != nil {
		return nil, fmt.Errorf("tokenizing: %w", err)
	}

	return m.inferFromEncodings(encodings)
}

// inferFromEncodings runs ONNX inference on tokenized inputs.
func (m *MiniLM) inferFromEncodings(encodings []tokenizer.Encoding) ([][]float32, error) {
	batchSize := len(encodings)
	seqLength := len(encodings[0].Ids)

	inputShape := ort.NewShape(int64(batchSize), int64(seqLength))

	// Prepare input tensors
	inputIDs := make([]int64, batchSize*seqLength)
	attentionMask := make([]int64, batchSize*seqLength)
	tokenTypeIDs := make([]int64, batchSize*seqLength)

	for b := range batchSize {
		for i, id := range encodings[b].Ids {
			inputIDs[b*seqLength+i] = int64(id)
		}
		for i, mask := range encodings[b].AttentionMask {
			attentionMask[b*seqLength+i] = int64(mask)
		}
		for i, typeID := range encodings[b].TypeIds {
			tokenTypeIDs[b*seqLength+i] = int64(typeID)
		}
	}

	// Create input tensors
	inputIDsTensor, err := ort.NewTensor(inputShape, inputIDs)
	if err != nil {
		return nil, fmt.Errorf("creating input_ids tensor: %w", err)
	}
	defer inputIDsTensor.Destroy()

	attentionMaskTensor, err := ort.NewTensor(inputShape, attentionMask)
	if err != nil {
		return nil, fmt.Errorf("creating attention_mask tensor: %w", err)
	}
	defer attentionMaskTensor.Destroy()

	tokenTypeIDsTensor, err := ort.NewTensor(inputShape, tokenTypeIDs)
	if err != nil {
		return nil, fmt.Errorf("creating token_type_ids tensor: %w", err)
	}
	defer tokenTypeIDsTensor.Destroy()

	// Create output tensor
	outputShape := ort.NewShape(int64(batchSize), int64(Dimensions))
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("creating output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Run inference
	inputs := []ort.Value{inputIDsTensor, attentionMaskTensor, tokenTypeIDsTensor}
	outputs := []ort.Value{outputTensor}

	if err := m.session.Run(inputs, outputs); err != nil {
		return nil, fmt.Errorf("running inference: %w", err)
	}

	// Extract results
	flatOutput := outputTensor.GetData()
	expectedSize := batchSize * Dimensions
	if len(flatOutput) != expectedSize {
		return nil, fmt.Errorf("unexpected output size: got %d, expected %d", len(flatOutput), expectedSize)
	}

	results := make([][]float32, batchSize)
	for i := range batchSize {
		results[i] = make([]float32, Dimensions)
		copy(results[i], flatOutput[i*Dimensions:(i+1)*Dimensions])
	}

	return results, nil
}

// Close releases ONNX resources.
func (m *MiniLM) Close() error {
	if m.session != nil {
		m.session.Destroy()
	}
	return ort.DestroyEnvironment()
}
