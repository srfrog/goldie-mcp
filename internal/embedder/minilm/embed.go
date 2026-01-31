package minilm

import _ "embed"

//go:embed tokenizer.json
var tokenizerData []byte

//go:embed model.onnx
var modelData []byte
