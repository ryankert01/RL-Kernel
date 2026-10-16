import sys, asyncio
from pathlib import Path
from playwright.async_api import async_playwright
wrapped, outdir, ids, scheme, width = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(","), sys.argv[4], int(sys.argv[5])
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": width, "height": 900}, color_scheme=scheme)
        await pg.goto(wrapped.as_uri()); await pg.wait_for_timeout(1500)
        for i in ids:
            el = await pg.query_selector(i)
            await el.screenshot(path=str(outdir / f"{i.strip('#.').replace(' ', '_')}.png"))
        await b.close()
asyncio.run(main())
