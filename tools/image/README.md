# bmp_demo


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
  [--mask-bright-red] [--mask-bright-green] [--mask-bright-blue] \
  [--no-blur] \
  -o <output.bmp>
```

- `--crop-x` / `--crop-y` — top-left of the crop (use `0 0` for the image origin)
- `--crop-width` / `--crop-height` — size of the crop
- `--mask-bright-red` — optional; before blurring, black out every cropped pixel whose red value is `>= 128`
- `--mask-bright-green` — optional; same, for the green channel (`>= 128`)
- `--mask-bright-blue` — optional; same, for the blue channel (`>= 128`)
- `--no-blur` — optional; skip the 3×3 box blur and write the (masked) crop as-is
- `-o` — output path for the crop


Example (full image, if it is 512×512):

```bash
./bin/bmp_demo ../tools/image/lena_gray.bmp \
  --crop-width=512 --crop-height=512 \
  --crop-x=0 --crop-y=0 \
  --mask-bright-red \
  -o blurred.bmp
```

If BMP image is grayscale (R = G = B), the red, green, and blue masks all black out the same pixels on that fixture.

