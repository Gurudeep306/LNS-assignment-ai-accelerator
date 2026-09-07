# lns_lib -- A Logarithmic Number System library for DNN computation

A small, `numpy`-only Python package implementing fixed-point Logarithmic Number System (LNS)
arithmetic in two formats, LNS16 and LNS8, with conversion to/from IEEE FP32/FP16 and an
error-analysis harness comparing LNS results against those references.

## Why LNS?

In a Logarithmic Number System, a real number `x` is stored as `(sign(x), log2(|x|))` instead of
`(sign, exponent, mantissa)`. Multiplication becomes an exact addition of the log fields
(`log(ab) = log(a) + log(b)`), which is why LNS suits multiply-accumulate-heavy workloads like DNN
inference/training: the multiplier in a MAC unit collapses into a cheap fixed-point adder. The
trade-off is addition, which has no closed-form log-domain identity and needs the classical
Gaussian-logarithm correction functions (see `lns_lib/arithmetic.py`).

## Repository layout

| File | Purpose |
|---|---|
| `lns_lib/format.py` | `LNSFormat` (bit layout: total/int/frac bits, zero sentinel, overflow/underflow thresholds, representable range) and the two standard instances `LNS16` (1 sign + 8 int + 7 frac) and `LNS8` (1 sign + 4 int + 3 frac). |
| `lns_lib/convert.py` | `LNSValue` (sign + integer log-magnitude code), `float_to_lns` / `lns_to_float` conversion to and from FP32/FP16, and `to_bits`/`from_bits` for the hardware bit-packed representation. |
| `lns_lib/arithmetic.py` | `lns_add`, `lns_mul`, `lns_sub`, `lns_mac`, `lns_dot` -- arithmetic on the log-domain codes directly, never by reconstructing operands as linear floats. Also `lns_add_lut` / `build_fplus_fminus_lut`, a lookup-table implementation of LNS addition used as a correctness cross-check. |
| `lns_lib/metrics.py` | `error()` (relative error, or absolute error when the reference is zero) and `ErrorStats`, an accumulator reporting mean/median/max error. |
| `lns_lib/__init__.py` | Public package API. |
| `pyproject.toml` | Packaging metadata -- `pip install -e .` installs `lns_lib`. |
| `tests/test_lns.py` | Correctness tests: round-trip precision bounds, exact multiplication, addition (same-sign, opposite-sign, exact cancellation, identity with zero), subtraction, MAC, overflow/underflow, LUT-vs-closed-form agreement, bit-packing round-trip. Run with `python tests/test_lns.py` or `pytest tests/`. |
| `analysis.py` | Generates zero/small/large/positive/negative/random test vectors, evaluates conversion and add/mul/mac error against FP32 and FP16, reports mean/max relative error tables, characterises overflow/underflow incidence across the FP32 range, and saves plots to `figures/`. Run with `python analysis.py`. |
| `LNS_Library_Assignment.ipynb` | The notebook this README was generated from -- builds the package with `%%writefile`, installs it, runs the tests and analysis, and discusses the results. Runs top-to-bottom in Colab. |
| `figures/` | Plots: `range_comparison.png`, `conversion_error_scatter.png`, `error_summary_bars.png`. |

## Quick start

```bash
pip install -e .
python tests/test_lns.py
python analysis.py
```

```python
import lns_lib as lns

a, _ = lns.float_to_lns(6.0, lns.LNS16)
b, _ = lns.float_to_lns(2.5, lns.LNS16)

prod, flags = lns.lns_mul(a, b)      # exact log-domain multiplication
summ, flags = lns.lns_add(a, b)      # Gaussian-logarithm addition
mac, flags  = lns.lns_mac(b, a, b)   # b + a*b

print(lns.lns_to_float(prod), lns.lns_to_float(summ), lns.lns_to_float(mac))
```

## Format summary

| Format | Bits | Layout | Representable range | Max quantization step (relative) |
|---|---|---|---|---|
| LNS16 | 16 | 1 sign + 8 int + 7 frac | ~2.96e-39 .. 3.38e+38 | ~0.27% |
| LNS8  | 8  | 1 sign + 4 int + 3 frac | ~4.26e-03 .. 2.35e+02 | ~4.43% |

Zero is a single reserved log-magnitude code (the most negative two's-complement value); results
that overflow the format saturate to its largest magnitude, and results that underflow flush to
zero. See `lns_lib/format.py` and the notebook's Section 1/6 for the full design rationale and a
discussion of range/precision trade-offs against FP32 and FP16.
