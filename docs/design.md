# Kindred — Design Document

Draft for review. Target Zig version: **0.16.0** (already set in `build.zig.zon`).
Code blocks are **API sketches**, not committed code.

---

## 1. Goals and non-goals

**Goals.** (G1) Describe a model in Zig; the forward pass becomes a graph IR. (G2) Reverse-mode autodiff over that IR, with gradients as *graph nodes*. (G3) Lower to a single StableHLO function `(params, batch) -> (loss, grads...)`. (G4) Compile that function **once** via PJRT and replay it. (G5) Host-side optimizer (SGD, Adam). (G6) Every XLA result is comparable to a pure-Zig CPU reference.

**Non-goals (v1).** Hand-written or JIT kernels — all compute comes from XLA's own lowering. Dynamic shapes. Multi-device / SPMD. `CustomCall`, FFI, Triton, Pallas. Quantized dtypes and mixed-precision training. Rematerialisation.

Each excluded item is a large independent subsystem. The thesis is that compiled training is achievable from a small auditable core; the fastest way to falsify it is to keep the core small.

---

## 2. Architecture overview

```
        user Zig code                  Kindred core (pure Zig)         XLA (external)
 ┌───────────────────────┐
 │ model.forward(b, x)   │──────────┐
 └───────────────────────┘          │
                          ┌─────────▼──────────┐
                          │  src/graph         │  arena of Nodes,
                          │  Graph, Builder,   │  static Shape, DType
                          │  Value             │
                          └─────────┬──────────┘
                                    │  gradients = new nodes
                          ┌─────────▼──────────┐
                          │  src/autodiff      │  reverse pass, per-op
                          │  gradients(g,loss)│  rules, adjoints
                          └─────────┬──────────┘
                                    │
                          ┌─────────▼──────────┐        ┌───────────────────┐
                          │  src/interp        │        │ PJRT_Client       │
                          │  Interpreter       │        │  .Compile         │
                          │  ← oracle, no XLA  │        │  .Execute         │
                          └─────────┬──────────┘        └─────────┬─────────┘
                                    │                            │
                          ┌─────────▼──────────┐                  │
                          │  src/emit          │── StableHLO ────▶│
                          │  lower() → MLIR    │     text         │
                          └────────────────────┘                  │
                                                                   │
                                                  ┌────────────────▼──────────┐
                                                  │ src/nn    (init, forward)│
                                                  │ src/optim (SGD, Adam)    │ ← host, per step
                                                  └───────────────────────────┘
```

Two independent consumers of `src/graph`: the emitter (→ XLA) and the interpreter (→ oracle). The interpreter exists so that "the XLA path is wrong" and "the graph is wrong" are distinguishable failures.

```
src/graph  src/autodiff  src/interp  src/emit  src/runtime  src/nn  src/optim
tests/grad_check  tests/reference
examples/mlp_mnist  examples/tiny_transformer
vendor/pjrt_c_api.h   (pinned commit)
```

`src/interp` is an **addition** to the originally planned layout, which had no home for the oracle.

---

## 3. Graph IR

`DType` = `f32 | f16 | bf16 | f64 | i8 | i32 | i64 | u8 | pred`; only `f32` is complete in v1, but the rest exist so dtype plumbing is exercised early. `Shape` is a static list of `i64`; v1 rejects `-1`. Dims are always slice-borrowed from a shape arena owned by the `Graph`.

```zig
pub const Value = struct { id: NodeId, ty: TensorType };

pub const Node = struct {
    op: Op,
    ty: TensorType,
    inputs: []const NodeId,   // slice into the graph's input buffer
};
```

`Op` is a `union(enum)` over payload structs (`add`, `mul`, `matmul`, `relu`, `reduceSum`, …) rather than a tagged struct with a fixed payload: it keeps the arena small and makes every consumer's `switch` exhaustive-checked.

**Ownership.** A traced graph is append-only and never mutated after tracing, so one `std.heap.ArenaAllocator` over a `GeneralPurposeAllocator` backs all `Node` records, shape dims, and attribute slices; none are freed individually. Constant tensor data is **borrowed**, not copied, so the caller must keep it alive for the graph's lifetime. Parameters are the exception: the arena holds the *node*, while values live in a separate `ParamStore` that the optimizer mutates in place. The accepted consequence is that a `Graph` cannot be freed piecewise, so bucketed graphs all live as long as the program — at a few thousand nodes this is not measurable, and is only revisited if it shows up.

**Construction.** Shape and dtype inference happen in `Builder`, not in the emitter, so an invalid op is a Zig error at trace time — the cheapest failure mode available.

```zig
pub const Builder = struct {
    g: *Graph,
    pub fn input(self: *Builder, name: []const u8, ty: TensorType) !Value;
    pub fn param(self: *Builder, name: []const u8, ty: TensorType, init: Init) !Value;
    pub fn constant(self: *Builder, data: []const f32, ty: TensorType) !Value;
    pub fn add(self: *Builder, a: Value, b: Value) !Value;   // broadcast rules checked here
    pub fn matmul(self: *Builder, a: Value, b: Value) !Value;
    pub fn relu(self: *Builder, x: Value) !Value;
    pub fn logSoftmax(self: *Builder, x: Value, axis: i64) !Value;
    pub fn reduceSum(self: *Builder, x: Value, dims: []const i64) !Value;
};
```

---

## 4. Autodiff

Reverse mode, recompute-based: we walk the *existing* graph backwards from the loss, so no separate tape is needed and the IR stays single. **Gradient accumulation happens in the graph, not in a mutable buffer** — when a node receives a second adjoint, we materialise an explicit `add` node joining the two. This is deliberate: XLA can fuse and CSE a DAG, but it cannot reason about a host-side mutable accumulator, and one would force a custom kernel.

```zig
pub fn gradients(g: *Graph, loss: Value) ![]Value  // one Value per trainable param, in declaration order
```

With `g` the upstream adjoint and `⊕` graph-level accumulation:

| Op | Forward | ∂L/∂inputs |
|---|---|---|
| `add` | `y = a + b` | `∂a ⊕= g`; `∂b ⊕= g` |
| `mul` | `y = a * b` | `∂a ⊕= g * b`; `∂b ⊕= g * a` |
| `sub` | `y = a - b` | `∂a ⊕= g`; `∂b ⊕= -g` |
| `div` | `y = a / b` | `∂a ⊕= g / b`; `∂b ⊕= -g * a / b²` |
| `neg` | `y = -x` | `∂x ⊕= -g` |
| `relu` | `y = max(x,0)` | `∂x ⊕= g * (x > 0 ? 1 : 0)` |
| `tanh` | `y = tanh x` | `∂x ⊕= g * (1 - y²)` — reuses the forward output, no recompute |
| `exp` / `log` | `y = eˣ` / `y = log x` | `∂x ⊕= g * y`; `∂x ⊕= g / x` |
| `matmul` | `y = a @ bᵀ` | `∂a ⊕= g @ b`; `∂b ⊕= gᵀ @ a` |
| `reduceSum` | `y = Σ_axes x` | `∂x ⊕= broadcast(g, x.shape)` |
| `broadcast` | `y = broadcast(x)` | `∂x ⊕= reduceSum(g, broadcast_axes)` |
| `logSoftmax` | `s = logSoftmax(x)` | `∂x ⊕= g - s * Σg` — stable, avoids `exp` overflow |
| `softmax` | `s = softmax(x)` | `∂x ⊕= s ⊙ (g - Σ(g ⊙ s))` |
| `reshape` / `transpose` | — | apply the inverse; no arithmetic |
| `reduceWindowMax` | `y[i] = max_j x[i+j]` | route `g[i]` to argmax `j`; **ties → `g / count`, not 0** |
| `param`, `input`, `constant`, `iota` | — | leaf, no rule |

`logSoftmax` rather than `softmax` is the v1 primitive: it is numerically stable and its derivative is a single fused expression, which is what makes the cross-entropy gradient cheap. Jalon 1 covers the first six rows plus `logSoftmax` and `reduceSum`, one `tests/grad_check` case each.

---

## 5. StableHLO emission

`emit.lower(g, spec) !Program` walks the graph topologically and writes MLIR **text** into an `ArrayList(u8)`. Text rather than bytecode because `PJRT_Client_Compile` accepts MLIR strings, and text is diffable in tests and pipeable into `mlir-opt`.

| Kindred op | StableHLO |
|---|---|
| `param` / `input` | `func.func` parameters, in `ProgramSpec` order |
| `constant` | `stablehlo.constant dense<...> : tensor<...>` |
| `add`/`sub`/`mul`/`div` | `stablehlo.add` / `.subtract` / `.multiply` / `.divide` |
| `matmul` | `stablehlo.dot_general` with `contracting_dims` |
| `relu` | `stablehlo.maximum(x, zeros)` |
| `tanh` / `exp` / `log` | `stablehlo.tanh` / `.exponential` / `.log` |
| `reduceSum` | `stablehlo.reduce` + `stablehlo.add` |
| `broadcast` / `reshape` / `transpose` | `stablehlo.broadcast_in_dim` / `.reshape` / `.transpose` |
| `logSoftmax` | `reduce`(max) → `subtract` → `exponential` → `reduce`(sum) → `log` |
| `reduceWindowMax` | `stablehlo.reduce_window` + `stablehlo.reduce_max` |
| `crossEntropy` | composed from `logSoftmax` + `iota` + `reduceSum` — no dedicated op needed |

**Conventions.** One `func.func @main` *(entry-name requirement unconfirmed, see À VÉRIFIER #1)*. Arguments are params first in declaration order, then non-param inputs in declaration order: `(%p0: tensor<784x256xf32>, ..., %batch: tensor<128x784xf32>, %labels: tensor<128xi32>)`. Results are positional: scalar loss first, then one gradient per trainable param in the same order. There is no implicit f32 upcast anywhere; `i32` labels stay `i32`. `ProgramSpec` is the single source of truth for arg/result order, shared by the emitter, the runtime wrapper, and the oracle, so the order is never written down twice.

---

## 6. PJRT runtime

### Header provenance

Vendor `xla/pjrt/c/pjrt_c_api.h` into `vendor/` at a **pinned commit SHA** — openxla/xla publishes no GitHub Releases, so there is no version number to pin to. Record the SHA and its `PJRT_API_MINOR` in `vendor/PINNED`, and have CI re-download and diff. At time of writing: `PJRT_API_MAJOR = 0`, `PJRT_API_MINOR = 116`, `PJRT_Api_STRUCT_SIZE = 1144`.

### Binding from Zig — measured, not assumed

```zig
const tc = b.addTranslateC(.{
    .root_source_file = b.path("vendor/pjrt_c_api.h"),
    .target = target, .optimize = optimize,
});
const c = tc.addModule("c");   // then: .imports = &.{.{ .name = "c", .module = c }}
```

I ran this against the real header under Zig 0.16.0. It translates cleanly — anonymous unions, `offsetof`, and the `PJRT_DEFINE_STRUCT_TRAITS` constants all survive. The consequences below are observed, not inferred:

| C construct | translate-c produces | Kindred must |
|---|---|---|
| `typedef enum {…} PJRT_Buffer_Type;` | `pub const PJRT_Buffer_Type = c_uint;` + file-scope `pub const PJRT_Buffer_Type_F32: c_int = 11;` | Declare our own `enum(DType)` mirror and `@intCast`. The generated file contains **zero** Zig enums. |
| anonymous `union` in `PJRT_NamedValue` | field `unnamed_0: union_unnamed_9` | Always initialise via `.unnamed_0 = .{ .int64_value = … }` |
| `enum { X_STRUCT_SIZE = … }` | file-scope `pub const X_STRUCT_SIZE: c_int` — **not** a struct member | `@intCast` when assigning to `struct_size: usize` |
| `struct X_Args { size_t struct_size; … }` | `X_Args = struct_X_Args` with `= 0` / `= null` defaults | `struct_size = @sizeOf(c.X_Args)` |

The last row is the ABI strategy. The API's own convention is that the *caller* declares how much of the struct it knows and the plugin checks before reading each field, so passing our own `@sizeOf` is correct and forward-compatible because the API only ever appends fields. Verified: `@sizeOf(PJRT_Client_Create_Args) == PJRT_Client_Create_Args_STRUCT_SIZE == 88`.

### Loading the plugin

`GetPjrtApi` is **not declared in `pjrt_c_api.h`** — it lives in the per-plugin headers (`pjrt_c_api_cpu.h`, …). We declare it ourselves and `dlopen` the plugin, which is never linked at build time:

```zig
extern "C" fn GetPjrtApi() *const c.PJRT_Api;

pub fn loadPlugin(path: []const u8) !std.DynLib {
    var lib = try std.DynLib.open(path);
    _ = lib.lookup(*const fn () callconv(.c) *const c.PJRT_Api, "GetPjrtApi") orelse
        return error.MissingGetPjrtApi;
    return lib;
}
```

I compiled a stub plugin against the real header, exported `GetPjrtApi`, and loaded it from Zig via `std.DynLib`: the symbol is unmangled, the lookup succeeds, and calling through the returned `PJRT_Api` table works.

**Discovery order:** `-Dplugin=<path>` build option → `KINDRED_PJRT_PLUGIN` env var → a candidate list (À VÉRIFIER #4). No auto-download: a library must not silently fetch a compiler at build time.

### Lifecycle

```
DynLib.open → GetPjrtApi() → check pjrt_api_version.major/minor
  → PJRT_Plugin_Initialize
  → PJRT_Plugin_Attributes          (xla_version, stablehlo_current/minimum_version)
  → PJRT_Client_Create(create_options)      [single-process, no KV callbacks]
  → PJRT_Client_Devices / AddressableDevices
  → per shape bucket:
       PJRT_Client_Compile(PJRT_Program{format="mlir"}, compile_options) → LoadedExecutable
       verify: LoadedExecutable_GetExecutable → OutputElementTypes / OutputDimensions
       per step:
         PJRT_Client_BufferFromHostBuffer  (params, batch) → PJRT_Buffer*
         PJRT_LoadedExecutable_Execute    → PJRT_Buffer* + device_complete_events
         PJRT_Buffer_ToHostBuffer         → read loss + grads back
         destroy output buffers, destroy events
```

**Compile options** are a *serialized `CompileOptionsProto`*. An empty string takes XLA's defaults, which is what v1 wants; any non-default option would mean hand-encoding protobuf for that field. That is a real but bounded cost, and it is why we add no protobuf dependency for it.

**Teardown order** is strict — output buffers → events → loaded executable → executable → client → `DynLib.close()`. A `deinit` that leaves buffers alive is the likeliest leak, so `DeviceBuffer` is an owned handle with a `deinit` and the suite runs under leak detection.

### Compile cache

Two layers, both keyed on `hash(StableHLO text ‖ plugin xla_version ‖ stablehlo_current_version ‖ device description ‖ compile_options)`: in-process on the `Client`, and on disk via `PJRT_Executable_Serialize` / `PJRT_Executable_DeserializeAndLoad` (both in the header). The device description and both StableHLO version bounds are in the key because serialised executables are not portable across them.

### PJRT-specific risks

**API drift is the headline risk.** On the header checked, `PJRT_Buffer_CopyToHost` **does not exist** — device-to-host is `PJRT_Buffer_ToHostBuffer` (size query with `dst = nullptr`, then copy, each returning a `PJRT_Event`) or `PJRT_Buffer_CopyRawToHost` (added in API 0.56). The official `CHANGELOG.md` has no entry for this rename, so the header is the only source of truth; every binding is therefore pinned to a vendored commit, never to a distribution header.

**Plugin/header skew.** A plugin built against newer XLA may report a higher `PJRT_API_MINOR` than our vendored header. We accept that direction silently (trailing fields we never read) and refuse to run when the plugin's major/minor is below our minimum. **Plugin availability:** no supported standalone "install the XLA CPU PJRT plugin" package was found; see À VÉRIFIER #4. **Zig churn:** `zig build` APIs moved substantially in 0.15/0.16 (`addModule`/`createModule`, `b.addTranslateC`, `std.Io`, `std.DynLib`), so a Zig release will break `build.zig`; all of it is isolated in one file.

---

## 7. `nn` and `optim`

`src/nn` is deliberately thin: a layer is a struct of `Value`s plus a `forward(*Builder, Value) !Value`. Layers hold no device state, so a model is fully reproducible from `(graph, params)` — which is what makes oracle-vs-XLA comparison possible at all.
```zig
// examples/mlp_mnist/main.zig — esquisse d'API
const std = @import("std");
const kd = @import("kindred");

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    var g = kd.Graph.init(arena);
    var b = kd.Builder.init(&g);

    const x  = b.input("batch",  kd.tensorType(.{ 128, 784 }, .f32));
    const y  = b.input("labels", kd.tensorType(.{ 128 },      .i32));
    const w1 = b.param("w1", kd.tensorType(.{ 784, 256 }, .f32), .xavierUniform);
    const b1 = b.param("b1", kd.tensorType(.{ 256 },      .f32), .zeros);
    const w2 = b.param("w2", kd.tensorType(.{ 256, 10 },  .f32), .xavierUniform);
    const b2 = b.param("b2", kd.tensorType(.{ 10 },       .f32), .zeros);

    const h      = b.relu(b.add(b.matmul(x, w1), b1));
    const logits = b.add(b.matmul(h, w2), b2);
    const loss   = b.reduceMean(b.crossEntropy(logits, y), &.{});

    const grads = try kd.gradients(&g, loss);              // 1. reverse mode

    const program = try kd.emit.lower(&g, .{               // 2. one StableHLO function
        .params = g.trainableParams(),
        .batch  = &.{ x, y },
        .loss   = loss,
        .grads  = grads,
    });

    var client = try kd.runtime.Client.init(arena, pluginPath);   // 3. compile once
    defer client.deinit();
    const exe = try client.compile(&program);
    var state = try exe.allocParams(arena);

    var trainer = kd.optim.Adam.init(.{ .lr = 1e-3, .b1 = 0.9, .b2 = 0.999, .eps = 1e-8 });

    while (try nextBatch()) |batch| {                     // 4. host-side loop
        const out = try exe.execute(&state.params, batch);
        defer out.deinit();
        try trainer.update(&state, out.grads);            // f32 storage, f64 arithmetic
        try log("loss={d:.5}", .{out.loss});
    }
}
```

Params are written out explicitly above because declaration order fixes argument and gradient order; the equivalent layer form, `fc1.forward(&b, x)` returning a `Value`, is the intended ergonomics once `nn.Linear` exists.

The optimizer is host-side and stateful by design. A step is microseconds of f32 work; moving it into the graph would mean re-deriving it and would tie Adam's `step` counter to the compiled program. The boundary is a documented line: **XLA owns `params → grads`, the host owns `params → params`.**

---

## 8. Test strategy

Three oracles, because any one can be wrong. (1) **Pure-Zig CPU interpreter** (`src/interp`) evaluates any `Graph` internally in f64, returns f32; it runs on every `zig build test` with no XLA and no network, and validates the builder, the rules, and the emitter's graph semantics. (2) **Finite differences** (`tests/grad_check`): central difference, `eps = 1e-3`, random inputs in `[-1, 1]`. (3) **JAX** (`tests/reference`) re-expresses each graph in JAX; `jax.grad` / `value_and_grad` outputs are committed as JSON fixtures regenerated only by a manual, reviewed step, so an accidental semantic change surfaces as a review diff rather than a silently-updated baseline.

| Comparison | Metric | Threshold | Note |
|---|---|---|---|
| Finite diff vs analytic, f32 | max relative | `1e-2` | Loose by design — f32 differencing is the noisy party |
| Finite diff vs analytic, f64 internal | max relative | `1e-5` | Small tensors only, to catch real rule errors |
| Oracle (f64→f32) vs PJRT, single op | max abs | `atol 1e-5, rtol 1e-4` | Values in `[-1, 1]` |
| Oracle vs PJRT, full model step | max abs | `atol 1e-3, rtol 1e-3` | XLA fuses and reassociates |
| Kindred vs JAX, loss over 200 steps | max abs on the curve | `1e-3` | Identical init, data order, hyper-parameters |
| Kindred vs JAX, gradients | max relative | `1e-3` | |

Bitwise reproducibility between the oracle and XLA is explicitly **not** a goal; chasing it would mean disabling fusion and misrepresenting production behaviour.

---

## 9. Milestones

Each ends with a criterion a stranger can run in a clean checkout.

**Jalon 0 — PJRT smoke test.** Vendor the header, `dlopen` the CPU plugin, compile a hand-written StableHLO module adding two `f32[2,2]` tensors, execute, read back. *Acceptance:* the four returned floats equal `a + b` computed in Zig for fixed literal `a`, `b`, under `std.testing.expectEqual` (bitwise). On failure the `PJRT_Error_Message` is printed.

**Jalon 1 — IR, oracle, autodiff.** `Graph`/`Builder`, the CPU interpreter, and rules for `add`, `mul`, `relu`, `matmul`, `reduceSum`, `logSoftmax`. *Acceptance:* every op has a `tests/grad_check` case passing at the f64 threshold (`1e-5` max relative); `zig build test` runs the set under `std.testing.allocator` with zero leaks; `relu` additionally has a hand-written expected-value test at exactly 0, since finite differences cannot resolve the kink.

**Jalon 2 — Emission + PJRT execution.** `emit.lower`, `runtime.Client`, `Executable.execute`. *Acceptance:* for every grad-check case the PJRT gradients match the oracle within `atol 1e-4`; the round trip emits exactly one `func.func`; compiling the same program twice hits the in-process cache, asserted via a compile counter.

**Jalon 3 — MLP MNIST + Adam.** *Acceptance:* committed `loss_curve.json` from Kindred and from JAX, identical seed and data order, whose 200-step curves agree within `1e-3` and both reach a final loss below `2.4` on a fixed 10k subset; reproducible from a clean checkout with no network.

**Jalon 4 — Tiny transformer.** `Embedding`, masked `MultiHeadAttention`, pre-norm `LayerNorm`, `GELU`, learned positional embedding, cross-entropy on a small fixed corpus. *Acceptance:* attention and layernorm each have a grad-check case against JAX `value_and_grad`; a 2-layer model trains below a committed loss threshold on a fixed seed; a masking test proves the gradient of position *i*'s loss is zero w.r.t. all positions `> i`.

**Jalon 5 — safetensors.** *Acceptance:* weights from a safetensors file load into `ParamStore` and yield an MNIST loss within `1e-4` of the pre-save training loss; save→load→predict round-trips bitwise.

---

## 10. Risks and open questions

| Risk | Impact | Mitigation |
|---|---|---|
| Activations retained across the reverse pass | OOM outside toy models | Rematerialisation (`remat` as a first-class `Builder` op) is the main lever and needs its own milestone after Jalon 4. XLA's buffer assignment decides what actually stays resident; we only control the graph. |
| No hand-written kernels | Slower than a tuned runtime; no flash-attention or fused RMSNorm+matmul | Accepted for v1; revisit with `CustomCall` only if measurements demand it. |
| Dynamic shapes | Every new batch size is a recompile | Bucketing in `src/nn`. `PJRT_Buffer_UnpaddedDimensions` / `DynamicDimensionIndices` exist in the API; v1 ignores them. |
| PJRT API drift (§6) | Run-time breakage on plugin upgrade | Pinned vendored header + CI diff job. |
| Zig language/stdlib churn | `build.zig` breaks on upgrade | Isolated in one file; CI pins one Zig version. |
| Plugin availability | Blocks Jalon 0 | Resolve before writing any emitter code. |
| StableHLO version skew | Emitted text rejected by the plugin | Read `stablehlo_current_version` / `stablehlo_minimum_version` from `PJRT_Plugin_Attributes` at startup and refuse to run outside the interval. |
| In-graph gradient accumulation | Graph can grow superlinearly with fan-out | Cap adjoints per node; measure on the transformer model. |
| Non-deterministic XLA reductions | Loss curves not bitwise reproducible | Accept; compare curves, not individual steps. |

**Open questions.** Does the optimizer belong in the graph? Currently no (§7); reconsider if host-device sync dominates on GPU. Should `Builder` expose a function-style transform API (`fn f(*Builder, []Value) !Value`) so models compose without classes? Leaning yes. Is `remat` needed before Jalon 4 or after? Do we need non-default `PJRT_Client_Compile` options in v1? Leaning no.

### À VÉRIFIER

1. Whether PJRT/XLA **requires the entry function to be named `main`** — not stated in the header or in `pjrt_integration.md`; assumed from HLO convention. Must be settled before the emitter is written.
2. Whether `PJRT_Executable_Serialize` / `DeserializeAndLoad` are supported and stable on the CPU and GPU plugins.
3. Whether the CPU plugin supports `PJRT_HostBufferSemantics_kMutableZeroCopy` (in-place parameter donation), which would remove one H2D copy per step.
4. **Exact distribution channel for a prebuilt PJRT CPU plugin.** No official standalone package was found; candidates include plugins shipped inside `jaxlib` wheels, whose on-disk layout was not confirmed. **Blocks Jalon 0.**
5. `PJRT_Buffer_ToHostBuffer` vs the older `PJRT_Buffer_CopyToHost` naming across XLA revisions — the official `CHANGELOG.md` has no entry for the rename, so the introducing minor version is unconfirmed.
6. The StableHLO version to target. Resolved at runtime from plugin attributes; the value to pin in `vendor/PINNED` is unconfirmed.
7. The minimum `PJRT_API_MINOR` asserted at startup — candidate: the version at which `PJRT_Buffer_ToHostBuffer` exists (currently ≤ 116).
8. Whether `PJRT_Client_Compile` with `format = "mlir"` accepts StableHLO *text* on the target plugin (the header says "MLIR module bytecode (or string)", implying yes).
9. Whether `PJRT_Executable_OutputElementTypes` and `OutputDimensions` are mandatory on all plugins or may be null.
10. GPU `PJRT_Client_Create` `create_options` keys (preallocation, platform) for CUDA and ROCm — typed `PJRT_NamedValue*` in the header, but the keys are undocumented.
11. Whether translate-c over a 3 200-line header is fast enough to stay in the default build graph, or should be opt-in. Fast locally; unmeasured on CI.

---

## 11. Alternatives rejected

| Alternative | Why rejected |
|---|---|
| **ZML** (code or build dependency) | The strongest candidate, worth stating plainly: OpenXLA's PJRT examples page lists ZML as a framework-level PJRT consumer, so the glue from §6 partly exists there. We build it ourselves because (a) a project meant to demonstrate a design should not hide its runtime behind another project's API — reviewers cannot audit what they cannot see; (b) its build drags in Bazel-era plumbing, against an explicit Zig-only-build constraint; (c) we need ~15 PJRT entry points, and binding to another framework's buffer and module abstractions means tracking its churn. Accepted cost: re-deriving a few hundred lines. ZML stays a useful independent cross-check. |
| Bazel, to build against XLA | Explicitly out of scope, and the dominant source of friction in C++ ML projects. Reinforces vendoring the header rather than building XLA. |
| Emitting StableHLO directly from user code, no intermediate IR | Autodiff needs a differentiable representation. StableHLO is a stable serialisation format, not a tape-friendly IR, and its op set is far larger than what needs differentiating. |
| Forward-mode AD | Right for Jacobians, wrong for training: reverse mode is O(1) in graph depth and yields parameter gradients in one sweep, which is the whole workload. |
| MLIR-XLA dialect instead of StableHLO | Explicitly unstable; StableHLO is the versioned interop contract. |
| `jaxlib` / Python embedding instead of raw PJRT | Requires a Python interpreter inside a Zig library and buys nothing over the C API. Kept only as a source of ground-truth values. |
| StableHLO reference interpreter as the runtime | Correct but far too slow to train. Its role in Kindred is tests-only, and the OpenXLA examples page describes the upstream version as not yet available. |
| Mutable gradient accumulators + donated buffers | XLA cannot fuse host-side mutation, so gradients would not be compilable. Revisit only if in-graph accumulation is measured to be a problem. |
| Custom kernels, Triton, `CustomCall` | Deferred; see §10. |

---

## Décisions à valider

1. **Minimum Zig version.** The repo is on 0.16.0. Floor there (matches today, but forces churn on every release) or hold at 0.15.x for a wider audience?
2. **Plugin acquisition** (À VÉRIFIER #4). Vendor a pinned prebuilt plugin in-repo, resolve it from `jaxlib` at runtime, or require a user-supplied path? This gates Jalon 0 and I would rather not guess.
3. **`src/interp` added to the layout.** Confirm the addition, and whether the oracle is a full interpreter or covers only the ops with grad-check cases.
4. **Optimizer stays host-side** (§7), or in the compiled graph from the start?
5. **Scope of Jalon 4** — 2-layer, 4-head, 128-dim decoder on a fixed small corpus, or larger? This sets the memory pressure that decides whether `remat` lands in v1 or v2.
6. **dtypes.** Is `f32`-only acceptable through Jalon 4, or does `bf16` need to work for the transformer milestone to be meaningful on GPU?
7. **API freeze.** Is the sketched surface (`Graph`/`Builder`/`gradients`/`emit.lower`/`runtime.Client`/`nn`/`optim`) close enough to freeze, or should I produce a second, narrower proposal before implementation starts?
