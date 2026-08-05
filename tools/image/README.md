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

