# Image tools

## `mask.py`

Creates a 1-bit mask BMP: white background with a black rectangle. Black pixels are masked out by `demo`.

```bash
python3 mask.py <input.bmp> [-o mask.bmp] \
  [--out-width W] [--out-height H] \
  [--x X] [--y Y] [--width W] [--height H]
```

- `input.bmp` — image used to pick the default mask size
- `-o` — where to write the mask (default: `<input>_mask.bmp`)
- `--out-width` / `--out-height` — mask size (default: same as the input)
- `--x` / `--y` — top-left corner of the black rectangle (default: `0 0`)
- `--width` / `--height` — size of the black rectangle (default: half the mask)

## `demo`

Crops the input to the mask size, then runs sharpen → blur → mask → Sobel and writes the result.

The mask width and height must be less than or equal to the input BMP. The output BMP is the same size as the mask (a top-left crop of the input).

### Build

From the `build` folder:

```bash
make
```

### Run

```bash
./bin/demo <input.bmp> <mask.bmp> <output.bmp>
```

Example:

```bash
python3 ../tools/image/mask.py input.bmp -o mask.bmp
./bin/demo input.bmp mask.bmp output.bmp
```

## File navigation

### Public API (`include/image/`)

| File | Role |
|------|------|
| `bmp_io.h` | `BGRImage`, `loadBMP`, `saveBMP` |
| `bmp_crop.h` | `cropImage` |
| `bmp_mask.h` | `maskImage`, `BGRColor` |
| `conv_filter.h` | Convolution filter modes and pipeline API |

### Library (`lib/image/`)

| File | Role |
|------|------|
| `bmp_io.cpp` | BMP load/save; row-padding strip via `FilterByMask` |
| `bmp_crop.cpp` | Rectangular crop via keep/discard mask + `FilterByMask` |
| `bmp_mask.cpp` | Apply 1-bit mask BMP onto a BGR image |
| `bmp_pipeline_internal.h` | Shared pipeline helpers (`AlignedByteBuffer`, materialize, color streams) |
| `conv_filter.cpp` | Conv filter orchestration / shared logic |
| `conv_filter_default.cpp` | Default (dense) convolution |
| `conv_filter_uniform.cpp` | Uniform-weight convolution |
| `conv_filter_low_rank.cpp` | Low-rank separable convolution |
| `conv_filter_frequency.cpp` | Frequency-domain convolution |
| `conv_filter_*_illustration.*` | Illustrator / debug capture for conv modes |
| `conv_filter_common.h` | Shared conv-filter internals |
| `CMakeLists.txt` | Builds the `image` module |

### Tools (`tools/image/`)

| File | Role |
|------|------|
| `demo.cpp` | End-to-end demo: crop → filters → mask → Sobel |
| `benchmark.cpp` | Image pipeline benchmarks (`image_benchmark`) |
| `mask.py` | Generate 1-bit mask BMPs for `demo` |

### Tests

| File | Role |
|------|------|
| `tests/test_bmp.cpp` | BMP load/crop (and related) unit tests |
| `tests/test_conv_filter.cpp` | Convolution filter unit tests |

