from PIL import Image

files = [
    "hist_r.png", "scatter_r.png",
    "hist_lam.png", "scatter_lam.png",
    "hist_eta.png", "scatter_eta.png",
]

imgs = [Image.open(f) for f in files]

col_widths = [max(imgs[r * 2 + c].size[0] for r in range(3)) for c in range(2)]
row_heights = [max(imgs[r * 2 + c].size[1] for c in range(2)) for r in range(3)]
col_offsets = [sum(col_widths[:c]) for c in range(2)]
row_offsets = [sum(row_heights[:r]) for r in range(3)]

grid = Image.new("RGB", (sum(col_widths), sum(row_heights)))
for i, img in enumerate(imgs):
    row, col = divmod(i, 2)
    grid.paste(img, (col_offsets[col], row_offsets[row]))

grid.save("summary.png")
print("Saved summary.png")
