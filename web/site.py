#!/usr/bin/env python3
"""site.py -- turn the built card into the published site (used by the GitHub workflow).

Copies nfl-dashboard.html to site/index.html and adds what lets an iPhone keep it on the Home Screen as an app:
the web app manifest, the icon, and a small offline worker that keeps the last copy for when there is no signal.
"""
import os, shutil

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB, OUT = os.path.join(BASE, 'web'), os.path.join(BASE, 'site')
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(BASE, 'nfl-dashboard.html'), encoding='utf-8') as fh:
    html = fh.read()
head = ('<link rel="manifest" href="manifest.webmanifest">\n'
        '<link rel="apple-touch-icon" href="apple-touch-icon.png">\n'
        '<link rel="icon" type="image/png" href="icon-192.png">\n'
        '<meta name="apple-mobile-web-app-title" content="NFL card">\n'
        '<meta name="theme-color" content="#F4F3EE" media="(prefers-color-scheme: light)">\n'
        '<meta name="theme-color" content="#14181D" media="(prefers-color-scheme: dark)">\n')
html = html.replace('</head>', head + '</head>', 1)
worker = ("<script>if('serviceWorker' in navigator)addEventListener('load',()=>navigator.serviceWorker"
          ".register('sw.js').catch(()=>{}))</script>\n")
html = html.replace('</body>', worker + '</body>', 1)
with open(os.path.join(OUT, 'index.html'), 'w', encoding='utf-8') as fh:
    fh.write(html)
for f in ('manifest.webmanifest', 'sw.js', 'apple-touch-icon.png', 'icon-192.png', 'icon-512.png'):
    shutil.copy(os.path.join(WEB, f), OUT)
print('site ready:', sorted(os.listdir(OUT)))
