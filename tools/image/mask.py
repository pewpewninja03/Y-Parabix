from pathlib import Path
import argparse
from PIL import Image, ImageDraw

here = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Create a 1-bit mask BMP (black rectangle region)")
parser.add_argument("input", type=Path, nargs="?", default=here / "lena_gray.bmp", help="Input BMP (default: lena_gray.bmp)")
parser.add_argument("-o", "--output", type=Path, default=None, help="Output mask BMP (default: <input>_mask.bmp)")
parser.add_argument("--out-width", type=int, default=None, help="Output mask width (default: input width)")
parser.add_argument("--out-height", type=int, default=None, help="Output mask height (default: input height)")
parser.add_argument("--x", type=int, default=0, help="Rectangle x position (default: 0)")
parser.add_argument("--y", type=int, default=0, help="Rectangle y position (default: 0)")
parser.add_argument("--width", type=int, default=None, help="Rectangle width (default: half output width)")
parser.add_argument("--height", type=int, default=None, help="Rectangle height (default: half output height)")
args = parser.parse_args()

img = Image.open(args.input)
img_width, img_height = img.size

out_w = args.out_width if args.out_width is not None else img_width
out_h = args.out_height if args.out_height is not None else img_height
rect_w = args.width if args.width is not None else out_w // 2
rect_h = args.height if args.height is not None else out_h // 2

if out_w < 1 or out_h < 1:
    raise SystemExit(f"output size must be positive, got {out_w}x{out_h}")
if rect_w < 1 or rect_h < 1:
    raise SystemExit(f"rectangle size must be positive, got {rect_w}x{rect_h}")
if args.x < 0 or args.y < 0 or args.x + rect_w > out_w or args.y + rect_h > out_h:
    raise SystemExit(
        f"rectangle ({args.x},{args.y}) {rect_w}x{rect_h} is outside output size {out_w}x{out_h}"
    )

output = args.output if args.output is not None else args.input.with_name(f"{args.input.stem}_mask.bmp")

mask = Image.new("1", (out_w, out_h), 1)
draw = ImageDraw.Draw(mask)
draw.rectangle([args.x, args.y, args.x + rect_w - 1, args.y + rect_h - 1], fill=0)
mask.save(output)
print(f"wrote {output} ({out_w}x{out_h}, rect ({args.x},{args.y}) {rect_w}x{rect_h})")
