// Records docs/media/walkthrough.webm (<= 90 s) and README screenshots with Playwright.
// Needs a running API + built UI, e.g. (test ports, so the live app on 5100/5101 is untouched):
//   API_PORT=5103 WEB_PORT=5104 ./scripts/demo.sh            (in one terminal)
//   NODE_PATH=<dir with playwright installed> WEB_PORT=5104 node scripts/record_walkthrough.cjs
const path = require("path");
const fs = require("fs");
const { chromium } = require("playwright");

const WEB = `http://127.0.0.1:${process.env.WEB_PORT || "5104"}`;
const OUT = path.join(__dirname, "..", "docs", "media");
const pause = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const tmp = fs.mkdtempSync(path.join(require("os").tmpdir(), "mg-video-"));
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 760 }, colorScheme: "light",
    recordVideo: { dir: tmp, size: { width: 1280, height: 760 } } });
  const page = await ctx.newPage();
  await page.goto(WEB, { waitUntil: "networkidle" });
  await page.waitForSelector(".evidence .card");
  await pause(3500); // evidence: curves, held-out result, provenance badges
  await page.getByRole("tab", { name: /Findings/ }).click();
  await pause(3500);
  await page.getByRole("tab", { name: /Serving/ }).click();
  await page.waitForSelector(".serving .card");
  await pause(2500);
  await page.screenshot({ path: path.join(OUT, "serving-kv-cache.png") });
  const center = page.locator(".panel.center");
  for (let i = 0; i < 6; i++) { await center.evaluate((el) => el.scrollBy({ top: 260, behavior: "smooth" })); await pause(1300); }
  const harness = page.locator("section.card", { hasText: "lm-evaluation-harness" });
  if (await harness.count()) {
    await harness.scrollIntoViewIfNeeded();
    await pause(1500);
    await harness.screenshot({ path: path.join(OUT, "serving-harness.png") });
    await pause(2500);
  }
  const live = page.locator("section.card", { hasText: "OpenAI-compatible API" });
  await live.scrollIntoViewIfNeeded();
  await pause(1200);
  await live.getByRole("button", { name: /POST \/v1\/completions/ }).click();
  await live.locator("text=finish_reason").waitFor({ timeout: 60000 });
  await pause(1500);
  await live.screenshot({ path: path.join(OUT, "serving-openai-live.png") });
  await pause(4000);
  await page.getByRole("tab", { name: /Generate/ }).click();
  await pause(2500);
  await ctx.close();
  await browser.close();
  const vid = fs.readdirSync(tmp).find((f) => f.endsWith(".webm"));
  fs.copyFileSync(path.join(tmp, vid), path.join(OUT, "walkthrough.webm"));
  console.log("wrote", fs.readdirSync(OUT).map((f) => `${f} ${fs.statSync(path.join(OUT, f)).size}`).join(", "));
})().catch((e) => { console.error(e); process.exit(1); });
