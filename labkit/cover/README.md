# Cover image (author only, not shipped)

`cover.svg` is the source (original artwork, Localpane palette and type stacks). `cover.png` is the
1280×720 render. To re-render with headless Chrome on macOS:

```sh
cd labkit/cover
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu \
  --hide-scrollbars --force-device-scale-factor=1 --window-size=1280,720 \
  --screenshot="$PWD/cover.png" "file://$PWD/render.html"
```
