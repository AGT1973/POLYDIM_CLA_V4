# Scientific Verification Contract

## Gate A — Geometry

A sphere point is a finite vector satisfying `||x|| = 1` within an explicitly configured tolerance. Zero vectors and non-sphere vectors are not silently accepted as valid manifold points.

For `y != -x`:

`Log_x(y) = theta * (y - <x,y>x) / ||y - <x,y>x||`

with `theta = 2 atan2(||x-y||, ||x+y||)`.

`Exp_x(v) = cos(||v||)x + sin(||v||)v/||v||` for tangent `v`.

At `y = -x`, the log is multi-valued. The library must either reject it or expose an explicit branch policy. A finite arbitrary vector must never be reported as the unique smooth logarithm.

## Gate B — Householder

For nonzero `v`,

`H_v(x) = x - 2(v^T x)/(v^T v) v`.

Properties:

- `H_v(H_v(x)) = x`
- `||H_v(x)|| = ||x||`
- `H_{c v}(x) = H_v(x)` for any nonzero finite scalar `c`
- zero/NaN/Inf inputs follow explicit policy

## Gate C — QR

For a full-rank tall-skinny matrix `A`:

- `Q^T Q ≈ I`
- `A ≈ Q R`
- conditioning sweep is reported

Rank deficiency is not hidden by adding a positive shift. A valid status/diagnostic is required.

The suite does not inherit theorem-level guarantees from a paper unless the implementation actually matches the paper's algorithm and assumptions.

## Gate D — Cayley-SMW

For `X^T X = I` and `W = G X^T - X G^T`:

`Y = (I + a/2 W)^(-1) (I - a/2 W) X`

must agree with the independently derived low-rank SMW form.

For valid inputs:

`Y^T Y ≈ I`.

The native kernel must not reduce to an Euler-like stub.

## Gate E — FFI

ABI contract must specify:

- dtype
- rank/shape
- layout/strides
- byte lengths
- pointer validity
- alias policy
- batching
- autodiff policy

A loaded shared library is not proof of execution. The tests must observe and compare actual native output.

## Gate F — PMTP

The binary format must have one canonical size and one canonical byte layout.

Required tests:

- header size exact
- encode/decode round-trip
- HMAC tamper rejection
- payload covered by authentication
- sequence replay rejection
- timestamp window policy
- shape overflow checks
- payload-size check before allocation
- malformed lengths rejected

## Gate G — Benchmark Integrity

Benchmarks must:

- split PRNG keys
- warm up compilation separately from steady-state timing
- report device and dtype
- avoid hidden synchronization artifacts
- run multiple scales
- report variance
- compare against an independent reference

A benchmark is not a correctness test.
