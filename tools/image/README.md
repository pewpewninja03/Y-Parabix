# bmp_demo

Crops a rectangle from a BMP, blurs it with a 3×3 box filter, and writes a 24-bit BMP. It is an end-to-end demo of what the `image` library can do: parse an 8-bit indexed BMP, expand its palette into color streams with a Parabix pipeline, crop a region, blur it, and serialize the result back to a 24-bit BMP.

## Build

From the `build` folder:

```bash
make bmp_demo
```

## Run

```bash
./bin/bmp_demo <input.bmp> \
  --crop-width=<w> --crop-height=<h> \
  --crop-x=<x> --crop-y=<y> \
  -o <output.bmp>
```

- `--crop-x` / `--crop-y` — top-left of the crop (use `0 0` for the image origin)
- `--crop-width` / `--crop-height` — size of the crop
- `-o` — output path for the blurred crop

Example (full image, if it is 512×512):

```bash
./bin/bmp_demo ../tools/image/lena_gray.bmp \
  --crop-width=512 --crop-height=512 \
  --crop-x=0 --crop-y=0 \
  -o blurred.bmp
```
