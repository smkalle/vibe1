// Render `flysaver snapshot --html` pages to PNG with headless Chromium.
// usage: NODE_PATH=$(npm root -g) node tools/shoot.cjs out.png < frame.html
const { chromium } = require("playwright");
const { readFileSync } = require("node:fs");

(async () => {
  const html = readFileSync(0, "utf8");
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  await page.setContent(html);
  await (await page.$("pre")).screenshot({ path: process.argv[2] });
  await browser.close();
})();
