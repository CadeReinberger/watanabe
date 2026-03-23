from PIL import Image

files = [
    "hist_r.png", "scatter_r.png",
    "hist_lam.png", "scatter_lam.png",
    "hist_eta.png", "scatter_eta.png",
]

imgs = [Image.open(f) for f in files]
w, h = imgs[0].size

grid = Image.new("RGB", (2 * w, 3 * h))
for i, img in enumerate(imgs):
    row, col = divmod(i, 2)
    grid.paste(img, (col * w, row * h))

grid.save("summary.png")
print("Saved summary.png")
