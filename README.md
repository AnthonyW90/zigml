# zigml

`zigml` is a learning project for working through _Build a Large Language Model_ in Zig instead of the book's Python implementation.

The goal is not to build the fastest or most production-ready GPT implementation. The goal is to understand the machinery behind large language models by recreating the pieces from scratch in a language that makes memory, allocation, data layout, and performance choices explicit.

This means a lot of things that Python libraries usually give us for free — tokenizers, tensors, matrix operations, model layers, training loops, and eventually backpropagation — will be built directly in Zig.

## Why Zig?

Zig makes this project a different kind of learning exercise than following the book directly in Python:

- explicit allocator usage
- manual control over data layout
- no default tensor library to hide the details
- straightforward systems-level performance experiments
- a good excuse to build small, understandable ML primitives from the ground up

This is mostly a learning adventure. Correctness, clarity, and understanding come before speed.

## Roadmap

### Stage 1 — Working with text

Based on chapter 2.

This stage is pure Zig and does not require math libraries. It focuses on the text-processing foundation needed before any neural network code exists.

Planned pieces:

- byte-pair encoding tokenizer
- vocabulary construction and storage
- text encoding and decoding
- sliding-window data loader
- allocator-friendly string and token handling
- hashmap-heavy tokenizer internals

This should be a good warm-up for Zig data structures, memory ownership, and API design.

### Stage 2 — Attention

Based on chapter 3.

This is where the project needs a tensor story. Before implementing attention, we need a small matrix/tensor module — likely starting with 2D float slices and growing only as needed.

Planned pieces:

- simple tensor/matrix representation
- matrix multiplication
- transpose
- elementwise operations
- softmax
- causal masking
- scaled dot-product attention
- multi-head attention

The intent is to keep the tensor layer small and understandable rather than immediately building a full numerical framework.

### Stage 3 — GPT architecture

Based on chapter 4.

Once attention works, the project can assemble the core GPT forward pass.

Planned pieces:

- token embeddings
- positional embeddings
- layer normalization
- GELU activation
- feed-forward blocks
- residual connections
- transformer blocks
- GPT model wiring
- basic text generation from an untrained model

At this point the model may generate text, but it will be untrained and therefore mostly nonsense. That is expected.

### Stage 4 — Pretraining

Based on chapter 5.

This is the hard part in Zig because training requires gradients and parameter updates.

There is an important fork in the road here:

1. Hand-derive gradients for each layer.
   - More educational.
   - Closer to a GPT-from-scratch-in-C style project, such as `llm.c`.
   - Less general, but very explicit.

2. Build a small reverse-mode autograd engine first.
   - More general.
   - More upfront infrastructure.
   - Potentially useful beyond this one model.

This stage also includes loading OpenAI GPT-2 weights so the model can produce meaningful text before or alongside training experiments.

Planned pieces:

- loss calculation
- backpropagation strategy
- optimizer implementation
- training loop
- checkpoint or weight loading
- GPT-2 weight import

### Stage 5 — Classification fine-tuning

Based on chapter 6.

This stage reuses the model and training machinery from stage 4, adapting it for classification tasks.

Planned pieces:

- classification head
- dataset formatting
- fine-tuning loop
- evaluation metrics

### Stage 6 — Instruction fine-tuning

Based on chapter 7.

This stage adapts the model for instruction-following behavior.

Planned pieces:

- instruction dataset handling
- prompt/response formatting
- supervised fine-tuning loop
- generation and evaluation utilities

## Current status

The project is currently at the beginning: a Zig package scaffold with a library module and executable. The next major milestone is stage 1: tokenizer, vocabulary, encoding/decoding, and sliding-window data loading.

## Building and testing

This project currently targets Zig `0.16.0` or newer.

Build the project:

```sh
zig build
```

Run the executable:

```sh
zig build run
```

Run tests:

```sh
zig build test
```

## Guiding principles

- Build the important pieces from scratch.
- Prefer simple, inspectable code over clever abstractions.
- Keep allocation and ownership explicit.
- Add abstractions only when repeated code proves they are needed.
- Treat this as a notebook in code: experiments are welcome, but understanding is the point.
