+++
title = "GSoC 2026: Portable, fast data-parallel primitives for the Julia GPU stack"
author = "Shreyas Hegde"
abstract = """
  My Google Summer of Code 2026 project filled in the data-parallel primitives that
  the Julia GPU stack was missing and made them fast enough to serve as the shared
  implementation for every backend. The work landed as 15 merged pull requests in
  AcceleratedKernels.jl 0.5, and a benchmarking campaign on an RTX 5080 shows the
  portable kernels matching or beating the vendor and Base paths across the board."""
+++

{{abstract}}

This post is my Google Summer of Code 2026 final report for [The Julia
Language](https://julialang.org/) (JuliaGPU), mentored by Tim Besard and Christian
Guinard. It describes what I set out to do, what got merged, the benchmark results,
and what is left for the future.

## The problem

The Julia GPU ecosystem lets you write array programs that run across NVIDIA, AMD,
and Intel GPUs. That portability relies on `GPUArrays.jl`, which defines the common
interface every vendor package implements. Several foundational operations, though,
lacked a shared implementation: `reverse`, `findall`, `accumulate!`, and `mapreduce`
either crashed on backends with no vendor override, or silently degraded to CPU
execution by copying data back and forth. A source audit across `GPUArrays.jl`,
`CUDA.jl`, `AMDGPU.jl`, `oneAPI.jl`, and `Metal.jl` confirmed the gaps: `reverse` and
`findall` were absent from the shared fallback, `accumulate!` existed only as a scalar
CPU fallback, and `mapreducedim!` was an `error("Not implemented")` stub, forcing every
backend to carry its own vendor-specific copy.

The project resolves these gaps by moving the critical GPU kernels upstream into
[`AcceleratedKernels.jl`](https://github.com/JuliaGPU/AcceleratedKernels.jl) (AK),
which is written once against `KernelAbstractions.jl` and runs on every backend, and
by refining `GPUArrays.jl` into a thin delegation layer. Because Julia's multiple
dispatch prefers the most specific method, routing the `AnyGPUArray` fallback to AK
means backends that already have an optimized vendor method keep it, while backends
missing the functionality automatically get a high-performance GPU implementation
instead of a crash or a silent slowdown.

## Project goals

Together with my mentors, I set these goals:

- Build out the missing data-parallel primitives (`reverse`, `findall`, `accumulate`,
  `mapreduce`, and a full `sort` family) as portable AK kernels, so `GPUArrays.jl` can
  delegate to one implementation instead of each backend maintaining its own.
- Add dimension-wise (`dims`) support across these operations, since real array code
  reduces, scans, reverses, and sorts along an axis, not only over a flat vector.
- Optimize the kernels so the portable path matches or beats the vendor and `Base`
  implementations, and add new sort algorithms where a single strategy does not fit
  every input shape.
- Run a benchmarking campaign across every primitive to prove there are no regressions
  from the portable implementations, and to find where tuning is still needed.

## What I did

The work landed as **15 merged pull requests** and shipped in the `AcceleratedKernels.jl`
0.5 release. AK 0.5 also reworked the host API (algorithms as values, an `Auto()`
selector that picks a strategy per input, and explicit workspaces), so every primitive
now presents one uniform calling convention across backends.

**Filling in the primitive suite.** Reductions and scans gained N-dimensional axis
handling, `reverse` was added and then extended to arbitrary `dims`, `findall` was
implemented from scratch as a scan-based stream compaction, and `map`/`map!` were
generalized to several source arrays. These are exactly the operations that were
previously absent or CPU-only in the shared layer.

**Sorting needed more than one algorithm.** A single comparison sort is not the right
choice for every shape: it is excellent on short slices but scales as `L log L` on long
ones. I added an opt-in LSD `RadixSort` and optimized it with portable, capability-gated
tricks (a larger block size, atomic histograms, chunked scatter), added a
vendor-agnostic `BitonicSort` network for short slices, and built a segmented
`RadixSort` path for sorting along `dims` whose cost is largely independent of slice
length. The `Auto()` selector then chooses between them, so users get the fastest
strategy for their data without picking one by hand.

**A full benchmarking campaign.** Every primitive I touched was measured as a pure AK
kernel against the `Base`/vendor path on identical data on an NVIDIA RTX 5080,
device-timed with `CUDA.@elapsed`, min-of-12, warmed, refreshing the input each
iteration for mutating operations, using AK's default settings with no per-device
tuning. The headline is that the portable kernels match or beat the vendor path on
every primitive:

{{img "speedups.png" "Speedup of AcceleratedKernels over the Base/vendor path across primitives on an RTX 5080"}}

The pattern is clean. Compute-shaped primitives (`sort`, `sortperm`, `scan`, `findall`)
win by large margins because the portable kernels use work-efficient algorithms, while
the bandwidth-bound primitives (`reduce`, `reverse`, `map`) read the input a constant
number of times and already sit at the memory ceiling, where parity is the best any
implementation can do. Sorting is where the gap is widest, and it grows with size as
`Base`'s comparison sort falls behind the work-efficient radix path:

{{img "sort_scaling.png" "Flat Int32 sort time versus array size: Base grows faster than AK Auto and RadixSort"}}

## What got merged upstream

Everything below is public and attributable. The table lists every pull request from
the project and its exact status.

| Repo | PR | What it does | Status |
|------|----|--------------|--------|
| `AcceleratedKernels.jl` | [#83](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/83) | Dimensional `mapreduce` / `reduce` | Merged (0.5) |
| `AcceleratedKernels.jl` | [#90](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/90) | Opt-in `RadixSort` via `alg` | Merged (0.5) |
| `AcceleratedKernels.jl` | [#97](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/97) | Optimize radix sort | Merged (0.5) |
| `AcceleratedKernels.jl` | [#102](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/102) | `reverse!` / `reverse` | Merged (0.5) |
| `AcceleratedKernels.jl` | [#105](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/105) | Vectorized by-block reduce loads | Merged (0.5) |
| `AcceleratedKernels.jl` | [#107](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/107) | `items_per_thread` for reductions | Merged (0.5) |
| `AcceleratedKernels.jl` | [#108](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/108) | `items_per_thread` for scans | Merged (0.5) |
| `AcceleratedKernels.jl` | [#114](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/114) | `dims` for `reverse` | Merged (0.5) |
| `AcceleratedKernels.jl` | [#115](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/115) | `findall` (stream compaction) | Merged (0.5) |
| `AcceleratedKernels.jl` | [#116](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/116) | Device-scope fence for DecoupledLookback | Merged (0.5) |
| `AcceleratedKernels.jl` | [#117](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/117) | `dims` for `sort` / `sortperm` | Merged (0.5) |
| `AcceleratedKernels.jl` | [#126](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/126) | `BitonicSort` | Merged (0.5) |
| `AcceleratedKernels.jl` | [#129](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/129) | Segmented `RadixSort` along `dims` | Merged (0.5) |
| `AcceleratedKernels.jl` | [#130](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/130) | Multi-source `map` / `map!` | Merged (0.5) |
| `AMDGPU.jl` | [#1069](https://github.com/JuliaGPU/AMDGPU.jl/pull/1069) | Guard empty-array `reverse` launch | Merged |
| `AcceleratedKernels.jl` | [#149](https://github.com/JuliaGPU/AcceleratedKernels.jl/issues/149) · [#150](https://github.com/JuliaGPU/AcceleratedKernels.jl/issues/150) | `Int32`-indexing and device-aware tuning measurements | Open issues (data recorded) |
| `GPUArrays.jl` | #786 / #787 / #788 | Delegation layer (my first attempt) | Closed, superseded by #790 |
| `GPUArrays.jl` | [#790](https://github.com/JuliaGPU/GPUArrays.jl/pull/790) | Route `sort` / `reduce` / `scan` / `reverse` / `findall` through AK | Open (Tim's, builds on my AK work) |

The primitive work is fully merged and released in `AcceleratedKernels.jl` 0.5. The
`GPUArrays.jl` delegation that consumes it is the one piece still open, and it is
landing through Tim's PR rather than mine by design, since it depends on the AK 0.5
host rework.

## The one place the default is wrong: device-aware tuning

The campaign surfaced a real follow-up. On AMD RDNA4 the whole-array `reduce` is
bandwidth-starved at the default `items_per_thread = 2` (about 158 GB/s). Sweeping that
knob shows the optimum is a property of the *device*, not the input: on AMD, raising
`items_per_thread` toward 8 to 16 recovers up to 1.5x more bandwidth, while on NVIDIA
the default of 2 is already at the knee, which is why the RTX 5080 `reduce` sits at
parity.

{{img "reduce_tuning.png" "Reduce bandwidth versus items_per_thread: NVIDIA is saturated at 2, AMD keeps climbing to 16"}}

A single global default is therefore wrong for a portable library. AK already exposes
the hook for the fix (`reduce_tuning(::Backend, ::Type)` and
`sort_tuning(::Backend, ::Type)`, following the precedent of the `oneAPI.jl` extension's
`predicate_tuning`), so the correct default per device is a small per-backend override
rather than a rewrite. The measurements are recorded in
[#150](https://github.com/JuliaGPU/AcceleratedKernels.jl/issues/150).

## Current state

Measured against the stated goals:

- **Missing primitives, as portable AK kernels** — done and merged. `reverse`,
  `findall`, `accumulate`, `mapreduce`/`reduce`, and the full `sort`/`sortperm` family
  all exist in AK 0.5 and run on every backend.
- **`dims` support** — done. Reductions, scans, `reverse`, and `sort` all take `dims`.
- **Performance parity or better** — done, and verified by the RTX 5080 campaign.
- **No regressions / benchmarking** — done; the full sweep backs the tables above.

The `GPUArrays.jl` delegation itself is landing through Tim's PR
[#790](https://github.com/JuliaGPU/GPUArrays.jl/pull/790), which depends on the AK 0.5
host rework; once it merges, backends missing a vendor method get the AK implementation
automatically.

## Challenges and lessons learned

**Correctness on a single-pass scan needs a real device-scope fence.** The
decoupled-lookback scan that makes `accumulate` fast is a single pass in which blocks
read each other's partial results, so it only produces correct output if every backend
guarantees a device-scope memory fence between the write and the read. The naive version
passed on one backend and silently produced wrong results on another. Fixing it
([#116](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/116)) meant implementing a
genuine device-scope fence on all backends. The lesson: on a portable stack, a
memory-ordering assumption that holds on one vendor is not portable until you have
verified it on each.

**GPU predicates have to stay type-stable.** Building `findall` as a stream compaction
([#115](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/115)) hit GPU-compilation
failures whenever the predicate closure captured a `Type` or produced a type-unstable
result. Portable GPU code is less forgiving than CPU code here, and the fix is discipline
about what a device closure may capture.

**Edge cases only show up on real hardware.** The out-of-place `reverse` crashed when
launched on an empty array, but only on AMD, and only once I ran the tests on an actual
RDNA4 card ([AMDGPU.jl #1069](https://github.com/JuliaGPU/AMDGPU.jl/pull/1069)).
Cross-backend portability work genuinely requires running on each backend.

**The ecosystem's own version constraints shape what you can measure.** AK 0.5 (on
KernelAbstractions 0.9) was not co-installable with the RDNA4-capable AMDGPU 2.8 (on
KernelAbstractions 0.10), so the full AMD sweep had to be dropped and an RTX 5080 became
the primary benchmark box. In a fast-moving package ecosystem the dependency graph is
part of the problem.

**Benchmark honestly, and trust the profiler over intuition.** Two optimizations that
sounded obviously good, a `shfl_down` subgroup reduction and `Int32` index arithmetic,
gave nothing once measured, because both operations are memory-bandwidth-bound rather
than limited by the thing I was optimizing. Measuring them early is what let the project
decide not to carry that complexity. Writing down a negative result is as valuable as
shipping a positive one.

**The headline lesson:** portability and performance are not in tension once you pick the
right algorithm. The compute-shaped primitives beat the vendor code precisely because one
carefully written `KernelAbstractions.jl` kernel can use a work-efficient algorithm that
each backend would otherwise reimplement, while the bandwidth-bound primitives already
sit at the memory ceiling where parity is the honest best.

## What's left

- **Device-aware `Auto` tuning** — the main follow-up contribution. The per-backend
  override belongs in a package extension through the `reduce_tuning`/`sort_tuning` hooks,
  and the cleanest moment to add it is once the AK host-API redesign has landed. Data in
  [#150](https://github.com/JuliaGPU/AcceleratedKernels.jl/issues/150).
- **Narrow-integer (`Int32`) indexing** — being handled at the GPU compiler level (Tim) by
  adding `llvm.assume` range hints so the compiler narrows `i64` to `i32` with no source
  changes. In a memory-bound regime the gain is marginal; measurements in
  [#149](https://github.com/JuliaGPU/AcceleratedKernels.jl/issues/149).
- **Small- and whole-array reductions** — a known slowdown tracked in
  [#135](https://github.com/JuliaGPU/AcceleratedKernels.jl/issues/135), which I am looking
  into.
- **GPUArrays on top of AK** — the larger direction is to build `GPUArrays.jl` on top of
  AK so backends no longer depend on it directly; the initial PR
  [#790](https://github.com/JuliaGPU/GPUArrays.jl/pull/790) is under evaluation.

## Links

- `AcceleratedKernels.jl` PRs: [#83](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/83),
  [#90](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/90),
  [#97](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/97),
  [#102](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/102),
  [#105](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/105),
  [#107](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/107),
  [#108](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/108),
  [#114](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/114),
  [#115](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/115),
  [#116](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/116),
  [#117](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/117),
  [#126](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/126),
  [#129](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/129),
  [#130](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/130).
- `AMDGPU.jl`: [#1069](https://github.com/JuliaGPU/AMDGPU.jl/pull/1069).
- `GPUArrays.jl` delegation: [#790](https://github.com/JuliaGPU/GPUArrays.jl/pull/790) (Tim's, builds on the AK 0.5 host rework).

Thanks to Tim Besard and Christian Guinard for their mentorship throughout, and to the
JuliaGPU community.
