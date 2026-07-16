# Distributed Inference Cluster: Project Architecture & Execution Plan

This document outlines the engineering blueprint and deployment strategy for building a decentralized, privacy-preserving **Pipeline-Parallel Inference Cluster**. The architecture pools memory across disparate consumer hardware to host large-scale AI models, leveraging a "U-Shaped" split inference pattern for complete end-to-end user privacy.

---

## 1. Project Overview & Objectives

* **Goal:** Enable a group of friends to collectively host large language models (e.g., 70B parameter models) by pooling local hardware resources over a distributed network.
* **Core Principle:** **Memory Pooling, Not Compute Acceleration.** The architecture aims to bypass individual VRAM/RAM constraints, allowing larger models to run via distributed layer allocation.
* **Design Paradigm:** A custom asynchronous pipeline-parallel queue coupled with a "U-Shaped" layer-splitting strategy to guarantee that no participating node can inspect or decode user text payloads.

---

## 2. Cluster Architecture & Layer Topology

The cluster utilizes a **U-Shaped Split Inference Architecture** to isolate raw text inputs and outputs on the client machine, delegating the compute-heavy hidden layers to the shared cluster infrastructure.

```
       +-------------------------------------------------------+
       |                  CLIENT DEVICE                        |
       |  [Raw Input] --> [Head Layers]    [Tail + LM Head] --> [Decoded Output]
       +------------------------|------------------^-----------+
                                |                  |
     ===========================|==== Network =====|===========================
                                v                  |
       +-------------------------------------------|-----------+
       |               DISTRIBUTED CLUSTER (TRUNK)             |
       |  [Node A: Layers 5-15] -> [Node B: Layers 16-25] -> ...|
       +-------------------------------------------------------+
```

### Layer Segmentation Breakdown
1.  **The Head (Client / Local):** Hosts the *Text Embedding Layer* and initial Transformer layers (Layers 1–4). Converts plaintext string tokens into an abstracted hidden-state tensor.
2.  **The Trunk (Distributed Network Cluster):** Hosts the bulk of processing layers (Layers 5 to $N-4$). Handed off sequentially among friends' machines via a streaming binary protocol.
3.  **The Tail (Client / Local):** Hosts the final Transformer layers and the *Language Model (LM) Head*. Receives the final hidden-state vector from the cluster and decodes it back into text tokens.

---

## 3. Communication Protocol Specification

To minimize network overhead during layer transitions, the cluster communicates via a lightweight, binary serialized custom protocol using **gRPC** over TCP.

### 3.1 Network Packet Schema
Every packet sent across the network between cluster nodes must conform to the following schema:

| Field Name | Data Type | Description |
| :--- | :--- | :--- |
| `request_id` | `string` / `uint64` | Unique global identifier tying the payload to an active client generation context. |
| `layer_index` | `uint32` | The specific model layer destination index expected to receive and process the payload. |
| `tensor_shape` | `uint32[]` | Dimensions of the activation tensor matrix (e.g., `[batch, sequence_length, hidden_dim]`). |
| `tensor_data` | `bytes` | Raw IEEE 754 float binary stream representing the activation layer matrix. |

### 3.2 Key-Value (KV) Cache Isolation
* The KV Cache corresponding to specific model layers remains static within the memory architecture of the machine currently hosting those layers.
* Nodes maintain an in-memory map tracking state indices: `Map<RequestID, KVCachePointer>`.
* When a node receives an activation packet, it pulls the matching `request_id` context out of VRAM to execute the forward pass, updating it before pushing the resulting tensor to the next target node.

---

## 4. Host Node Token Pipelining & Queueing Strategy

To resolve the inherent "Hot Potato" problem where upstream and downstream nodes sit idle waiting for sequential calculations, the Host node manages a high-throughput **Token Pipelining Queue**.

```
Cycle 1:  [Node A: Prompt 1]  ->  [Node B: Idle]      ->  [Node C: Idle]
Cycle 2:  [Node A: Prompt 2]  ->  [Node B: Prompt 1]  ->  [Node C: Idle]
Cycle 3:  [Node A: Prompt 3]  ->  [Node B: Prompt 2]  ->  [Node C: Prompt 1]
```

### Queue Implementation Steps
1.  **Global Buffer:** The Host maintains an asynchronous scheduling queue (`asyncio.Queue`) for inbound client requests.
2.  **Staggered Ingestion:** Requests are ingested continuously. As soon as Node A finishes computing its assigned chunk of layers for Request 1 and hands off the packet, it immediately pulls Request 2 from the buffer.
3.  **Throughput Optimization:** This maximizes aggregate hardware utilization, shifting the cluster metric from low-latency single-stream generation to high-throughput multi-agent execution.

---

## 5. Implementation Roadmap

### Phase 1: Environment & Tooling Setup
* Ensure all cluster machines utilize an identical GGUF quantization baseline (e.g., `Q4_K_M` or `Q5_K_M`) to guarantee uniform vector scaling and dramatically lower data payloads.
* Establish static local IP routes over Gigabit Ethernet connections; configure firewall exceptions for target cluster communication ports.

### Phase 2: Core Network Layer Construction
* Define the protobuf file structure encompassing the `Request ID`, `Layer Index`, and `Tensor Data` schemas.
* Build an asynchronous gRPC streaming server in Python capable of routing arbitrary dimensional arrays using `NumPy` buffers.

### Phase 3: Model Splitting & Runtime Hooking
* Implement custom execution hooks wrapping `llama.cpp` or a `transformers` runtime framework to load restricted slice subsets of model architectures.
* Verify that intermediate activations can be cleanly parsed, packed, transmitted, and injected into the sequential execution block of a downstream worker.

### Phase 4: Security Validation & Stress Testing
* Deploy debugging tools (e.g., memory dumps, network sniffers) to confirm that no node within the shared Trunk layer can reconstruct human-readable data from passing packets.
* Simulate concurrent agent swarms to test edge-case conditions within the host pipeline coordinator, tuning asynchronous task loop timeouts for optimal network throughput.

---

## 6. Optimization Rules & Edge Cases

* **The Weakest Link Constraint:** The overall decoding cycle speed will natively throttle down to the speed of the node with the lowest internal memory bus bandwidth. Avoid routing layers through system memory (CPU RAM) if remaining nodes are running dedicated VRAM (GPUs).
* **Network Jitter Mitigation:** Avoid running the Trunk layers over Wi-Fi networks. High jitter introduces dropped packets that invalidate the tight token synchronization windows needed by the host coordinator queue.
