# AI Inference Server · C++20

A high-performance HTTP inference server built from scratch in **C++20** using ONNX Runtime and Crow. Designed as a portfolio project targeting AI/ML backend engineering roles.

**~132 000 req/s · p50 < 0.05 ms · p99 < 0.2 ms** (Apple M-series, tiny model, loopback)

---

## Demo

```bash
chmod +x demo.sh && ./demo.sh
```

[`demo.sh`](demo.sh) — builds the project, generates the test model, starts the server, and exercises all three endpoints (`/health`, `/predict`, `/benchmark`).
