import sys, asyncio
from pathlib import Path
from playwright.async_api import async_playwright
wrapped, outdir, ids = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(",")
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1280, "height": 900}, color_scheme=sys.argv[4] if len(sys.argv) > 4 else "light")
        await pg.goto(wrapped.as_uri()); await pg.wait_for_timeout(1500)
        for i in ids:
            el = await pg.query_selector(f"#{i}")
            await el.screenshot(path=str(outdir / f"el_{i}.png"))
            print(i, await el.bounding_box())
        await b.close()
asyncio.run(main())
