"""Screenshot an HTML file (wrapped in the artifact skeleton) at desktop and phone width, light and dark."""
import sys, asyncio
from pathlib import Path
from playwright.async_api import async_playwright
src, outdir = Path(sys.argv[1]), Path(sys.argv[2]); outdir.mkdir(parents=True, exist_ok=True)
body = src.read_text()
page_html = ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
             '<style>:root{color-scheme:light}body{margin:0;font:14px system-ui}img{max-width:100%}[hidden]{display:none!important}</style>'
             + body + '</head><body></body></html>')
tmp = outdir / "_wrapped.html"; tmp.write_text(page_html)
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        for name, w, scheme in [("desk_light", 1280, "light"), ("desk_dark", 1280, "dark"), ("phone_light", 400, "light")]:
            pg = await b.new_page(viewport={"width": w, "height": 900}, color_scheme=scheme, device_scale_factor=1)
            msgs = []
            pg.on("console", lambda m: msgs.append(m.text)); pg.on("pageerror", lambda e: msgs.append(str(e)))
            await pg.goto(tmp.as_uri()); await pg.wait_for_timeout(1500)
            sw = await pg.evaluate("document.documentElement.scrollWidth")
            await pg.screenshot(path=str(outdir / f"{name}.png"), full_page=True)
            print(name, "scrollWidth", sw, "viewport", w, "console:", msgs[:5])
        await b.close()
asyncio.run(main())
