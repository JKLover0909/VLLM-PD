import asyncio
import json
import urllib.request
import websockets
import subprocess
import time
from pathlib import Path

URL = "http://127.0.0.1:8001/"
REMOTE_PORT = 9226
PROFILE = "/tmp/meibook-mobile-chrome-2"

async def check():
    chrome = subprocess.Popen([
        "/usr/bin/google-chrome", "--headless=new", "--no-sandbox",
        "--disable-gpu", "--disable-dev-shm-usage",
        f"--remote-debugging-port={REMOTE_PORT}",
        f"--user-data-dir={PROFILE}",
        "--window-size=390,844", "about:blank"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    try:
        targets = None
        for _ in range(60):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{REMOTE_PORT}/json", timeout=1) as response:
                    targets = json.load(response)
                break
            except Exception:
                time.sleep(0.1)
                
        page = next(item for item in targets if item["type"] == "page")
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=30_000_000) as ws:
            next_id = 0
            async def command(method, params=None):
                nonlocal next_id
                next_id += 1
                request_id = next_id
                await ws.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))
                while True:
                    message = json.loads(await ws.recv())
                    if message.get("id") == request_id:
                        return message.get("result", {})
                        
            async def evaluate(expression):
                result = await command("Runtime.evaluate", {"expression": expression, "returnByValue": True})
                return result["result"].get("value")
                
            await command("Page.enable")
            await command("Runtime.enable")
            await command("Emulation.setDeviceMetricsOverride", {
                "width": 390, "height": 844, "deviceScaleFactor": 3, "mobile": True,
                "screenWidth": 390, "screenHeight": 844
            })
            await command("Page.navigate", {"url": URL})
            time.sleep(2)
            
            print("Checking UI state on mobile layout...")
            
            # Check overflows
            overflows = await evaluate("""
                (() => {
                    const visible = (el) => {
                        const s = getComputedStyle(el), r = el.getBoundingClientRect();
                        return s.display !== 'none' && s.visibility !== 'hidden' && r.width > 0 && r.height > 0;
                    };
                    const viewportWidth = document.documentElement.clientWidth;
                    return [...document.querySelectorAll('body *')].filter(visible).map(el => {
                        const r = el.getBoundingClientRect();
                        return {tag: el.tagName.toLowerCase(), cls: typeof el.className === 'string' ? el.className.slice(0, 100) : '', r};
                    }).filter(({r}) => r.right > viewportWidth + 1 || r.left < -1).length;
                })()
            """)
            
            # Check small targets
            small_targets = await evaluate("""
                (() => {
                    const visible = (el) => {
                        const s = getComputedStyle(el), r = el.getBoundingClientRect();
                        return s.display !== 'none' && s.visibility !== 'hidden' && r.width > 0 && r.height > 0;
                    };
                    return [...document.querySelectorAll('button, input, textarea, select, a, summary')]
                        .filter(visible)
                        .filter(el => {
                            const r = el.getBoundingClientRect();
                            return r.width > 0 && r.height > 0 && (r.width < 44 || r.height < 44);
                        })
                        .map(el => {
                            const r = el.getBoundingClientRect();
                            return `${el.tagName.toLowerCase()}.${typeof el.className === 'string' ? el.className.slice(0, 20) : ''} (${Math.round(r.width)}x${Math.round(r.height)})`;
                        });
                })()
            """)
            
            print(f"Overflows: {overflows}")
            print(f"Small Targets: {len(small_targets)}")
            if small_targets:
                print("Small targets found:")
                for target in small_targets:
                    print(f"  - {target}")
    finally:
        chrome.terminate()

if __name__ == "__main__":
    asyncio.run(check())
