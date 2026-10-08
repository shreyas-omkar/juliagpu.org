+++
title = "GSoC 2026: Portable data-parallel primitives for the Julia GPU stack"
author = "Shreyas Hegde"
abstract = """
  This Google Summer of Code 2026 project added the data-parallel primitives that
  the Julia GPU stack was missing and tuned them to serve as the shared
  implementation for every backend. The work was merged as 15 pull requests in
  AcceleratedKernels.jl 0.5. A benchmark compares each operation against
  single-threaded CPU Julia and the current CUDA.jl GPU implementation on an
  NVIDIA RTX 5080."""
+++

{{abstract}}

This post is my Google Summer of Code 2026 final report for [The Julia
Language](https://julialang.org/) (JuliaGPU), mentored by Tim Besard and Christian
Guinard. It describes the project goals, what was done, what was merged, the
benchmark results for each operation, and the work that remains.

## The problem

The Julia GPU ecosystem lets users write array programs that run on NVIDIA, AMD, and
Intel GPUs. That portability relies on `GPUArrays.jl`, which defines the common
interface every vendor package implements. Several common operations did not have a
shared implementation. `reverse`, `findall`, `accumulate!`, and `mapreduce` either
failed on backends with no vendor-specific method, or fell back to CPU execution by
copying data to the host and back. A source review of `GPUArrays.jl`, `CUDA.jl`,
`AMDGPU.jl`, `oneAPI.jl`, and `Metal.jl` confirmed the gaps: `reverse` and `findall`
were missing from the shared fallback, `accumulate!` existed only as a scalar CPU
fallback, and `mapreducedim!` was an `error("Not implemented")` stub, so each backend
carried its own copy.

The project addresses these gaps by moving the kernels into
[`AcceleratedKernels.jl`](https://github.com/JuliaGPU/AcceleratedKernels.jl) (AK),
which is written once against `KernelAbstractions.jl` and runs on every backend, and
by keeping `GPUArrays.jl` as a thin layer that delegates to it. Because Julia's
multiple dispatch selects the most specific method, routing the `AnyGPUArray` fallback
to AK means a backend that already has a vendor method keeps it, while a backend
without one receives the AK implementation instead of an error or a host round trip.

## Project goals

The goals I set with my mentors were:

- Add the missing data-parallel primitives (`reverse`, `findall`, `accumulate`,
  `mapreduce`, and the `sort` family) as portable AK kernels, so `GPUArrays.jl` can
  use one implementation instead of each backend maintaining its own.
- Add dimension-wise (`dims`) support across these operations, since array code
  reduces, scans, reverses, and sorts along an axis, not only over a flat vector.
- Tune the kernels so the portable path is at least as fast as the vendor and `Base`
  implementations, and add further sort algorithms where one strategy does not fit
  every input shape.
- Run a benchmark across every operation to check for regressions and to find where
  tuning is still required.

## What I did

The work was merged as 15 pull requests and released in `AcceleratedKernels.jl` 0.5.
AK 0.5 also reworked the host API (algorithms passed as values, an `Auto()` selector
that chooses a strategy per input, and explicit workspaces), so every operation now
has one calling convention across backends.

**Primitives and `dims` support.** Reductions and scans gained N-dimensional axis
handling, `reverse` was added and then extended to arbitrary `dims`, `findall` was
implemented as a scan-based stream compaction, and `map` and `map!` were generalized
to several source arrays.

**Sorting.** A single comparison sort is not the best choice for every shape, since it
is good on short slices but grows as `L log L` on long ones. I added an opt-in LSD
`RadixSort` and tuned it with portable, capability-gated changes (a larger block size,
atomic histograms, chunked scatter), added a `BitonicSort` network for short slices,
and added a segmented `RadixSort` path for sorting along `dims` whose cost depends
little on slice length. The `Auto()` selector chooses between them per input.

**Benchmarking.** Each operation was measured three ways on the same data: as an AK
kernel, as the CUDA.jl GPU path, and as single-threaded CPU Julia. The GPU runs were
on an NVIDIA RTX 5080, timed on the device with `CUDA.@elapsed`, taking the minimum of
12 runs, warmed, refreshing the input each iteration for mutating operations, and using
AK's default settings with no per-device tuning. The CPU runs used single-threaded
Julia (`Base`) on the host, timed with the minimum of a few warmed runs. The results
per operation follow.

## Results by operation

In every chart, lower time is better, and both axes use a logarithmic scale unless
noted. The three series are:

- **Base (CPU):** single-threaded CPU Julia (`sort!`, `sortperm`, `cumsum!`, `sum`,
  `findall`, `reverse!`, `map!`) on a host `Array`. This is the slowest reference and
  sits above the two GPU lines.
- **CUDA.jl:** the GPU method that runs today when the same standard function is called
  on a `CuArray`, which goes through CUDA.jl and GPUArrays.jl.
- **AK:** the AcceleratedKernels.jl kernel.

The comparison that matters for this project is `AK` against `CUDA.jl`, since both run
on the GPU. `Base (CPU)` is included for context.

### Sort

For a flat `Int32` array, `AK Auto` is about 5 to 7 times faster than the CUDA.jl sort
across the mid to large sizes, and the explicit `RadixSort` reaches about 15 times
faster at 128M elements. The gap grows with size, because the comparison sort in
CUDA.jl grows faster than the radix path.

{{img "sort.png" "Flat Int32 sort time versus array size for CUDA.jl, AK Auto, and AK RadixSort"}}

### sortperm

The CUDA.jl `sortperm` is slow, so the portable path shows a large difference here,
roughly 10 to 25 times faster depending on size.

{{img "sortperm.png" "Flat Int32 sortperm time versus array size for CUDA.jl and AK Auto"}}

### accumulate

For a `Float32` cumulative sum, AK is about 1.6 to 9 times faster than the CUDA.jl
scan, with the larger difference at the mid sizes.

{{img "accumulate.png" "Float32 cumsum time versus array size for CUDA.jl and AK"}}

### reduce and mapreduce

A whole-array reduction reads the input a fixed number of times, so it is limited by
memory bandwidth. AK and CUDA.jl are at parity here, and within measurement noise AK is
slightly slower at a few mid sizes.

{{img "reduce.png" "Whole-array Float32 reduce time versus array size for CUDA.jl and AK"}}

Reduction along a dimension has more room, since the work can be organized to read
memory in a coalesced order. For `dims=2` AK is faster on most shapes, by up to about
3 times, and at parity or slightly slower on a few.

{{img "mapreduce_dims.png" "mapreduce along dims=2 time by array shape for CUDA.jl and AK"}}

### findall

Built as a scan-based stream compaction, AK is about 2 to 5 times faster than the
CUDA.jl `findall` across result densities.

{{img "findall.png" "findall time versus array size at density 0.5 for CUDA.jl and AK"}}

### reverse

`reverse` is bandwidth-limited, so AK and the CUDA.jl `reverse!` are at parity.

{{img "reverse.png" "Flat Int32 reverse time versus array size for CUDA.jl and AK"}}

### map

`map` is also bandwidth-limited, and AK is at parity with the CUDA.jl `map!`.

{{img "map.png" "Float32 map time versus array size for CUDA.jl and AK"}}

The pattern across the operations is consistent. The compute-bound operations (`sort`,
`sortperm`, `accumulate`, `findall`, and reduction along a dimension) are faster
because a work-efficient algorithm helps, while the bandwidth-bound operations
(`reduce`, `reverse`, `map`) are at parity, which is the expected result when the
kernel already reads the input at the memory bandwidth limit.

## What got merged upstream

The table lists the pull requests from the project and their status. All of the
primitive work is merged and released in `AcceleratedKernels.jl` 0.5.

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

Two measurement write-ups are recorded as open issues so the data lives in the repo:
`Int32` indexing in [#149](https://github.com/JuliaGPU/AcceleratedKernels.jl/issues/149)
and device-aware tuning in [#150](https://github.com/JuliaGPU/AcceleratedKernels.jl/issues/150).
The `GPUArrays.jl` delegation that consumes this work is still open as
[#790](https://github.com/JuliaGPU/GPUArrays.jl/pull/790); it is maintained by Tim and
builds on the AK 0.5 host rework.

## Current state

Measured against the stated goals:

- **Missing primitives as portable AK kernels:** done and merged. `reverse`,
  `findall`, `accumulate`, `mapreduce` or `reduce`, and the `sort` and `sortperm`
  family are in AK 0.5 and run on every backend.
- **`dims` support:** done. Reductions, scans, `reverse`, and `sort` take `dims`.
- **Performance at least at parity:** done, as the per-operation charts show.
- **Regression check:** done, through the full benchmark run.

The `GPUArrays.jl` delegation is landing through
[#790](https://github.com/JuliaGPU/GPUArrays.jl/pull/790), which depends on the AK 0.5
host rework. Once it merges, a backend without a vendor method uses the AK
implementation.

## Challenges and lessons learned

**A single-pass scan needs a real device-scope fence.** The decoupled-lookback scan
that makes `accumulate` fast is a single pass in which blocks read each other's partial
results, so it is correct only if every backend provides a device-scope memory fence
between the write and the read. The first version passed on one backend and produced
wrong results on another. The fix
([#116](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/116)) added a real
device-scope fence on every backend. A memory-ordering assumption that holds on one
vendor is not portable until it is checked on each.

**GPU predicates have to stay type-stable.** Building `findall` as a stream compaction
([#115](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/115)) failed to compile
whenever the predicate closure captured a `Type` or returned a type-unstable value. The
kernels had to be written so the predicate is type-stable. GPU code is less forgiving
than CPU code here, and the fix is care about what a device closure captures.

**Edge cases show up only on real hardware.** The out-of-place `reverse` failed when
launched on an empty array, but only on AMD, and only when the tests ran on an actual
RDNA4 card ([AMDGPU.jl #1069](https://github.com/JuliaGPU/AMDGPU.jl/pull/1069)).
Portability work needs to run on each backend.

**Package version constraints affect what can be measured.** AK 0.5 (on
KernelAbstractions 0.9) was not co-installable with the RDNA4-capable AMDGPU 2.8 (on
KernelAbstractions 0.10), so the full AMD sweep was set aside and an RTX 5080 became
the main benchmark machine. In a fast-moving ecosystem the dependency graph is part of
the problem.

**Measure before adding complexity.** Two changes that looked promising, a `shfl_down`
subgroup reduction and `Int32` index arithmetic, gave no improvement once measured,
because both operations are limited by memory bandwidth rather than by the part being
changed. Measuring them early kept that complexity out of the code. A recorded negative
result is as useful as a positive one.

The main lesson is that portability and performance are compatible once the algorithm
fits the operation. The compute-bound operations improve because one portable kernel can
use a work-efficient algorithm that each backend would otherwise reimplement, and the
bandwidth-bound operations reach parity, which is the expected ceiling.

## What is left

- **Device-aware `Auto` tuning.** The per-backend override belongs in a package
  extension through the `reduce_tuning` and `sort_tuning` hooks. The cleanest time to
  add it is after the AK host-API redesign lands. Data in
  [#150](https://github.com/JuliaGPU/AcceleratedKernels.jl/issues/150).
- **Narrow-integer (`Int32`) indexing.** This is being handled at the GPU compiler
  level (by Tim) by adding `llvm.assume` range hints so the compiler narrows `i64` to
  `i32` with no source changes. In a memory-bound regime the effect is small.
  Measurements in [#149](https://github.com/JuliaGPU/AcceleratedKernels.jl/issues/149).
- **Small and whole-array reductions.** A known slowdown is tracked in
  [#135](https://github.com/JuliaGPU/AcceleratedKernels.jl/issues/135), which I am
  looking into.
- **GPUArrays on top of AK.** The larger plan is to build `GPUArrays.jl` on top of AK
  so backends no longer depend on it directly. The first PR
  [#790](https://github.com/JuliaGPU/GPUArrays.jl/pull/790) is under review.

## Links

- `AcceleratedKernels.jl` pull requests: [#83](https://github.com/JuliaGPU/AcceleratedKernels.jl/pull/83),
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
- `GPUArrays.jl` delegation: [#790](https://github.com/JuliaGPU/GPUArrays.jl/pull/790).

Thanks to Tim Besard and Christian Guinard for their mentorship, and to the JuliaGPU
community.
