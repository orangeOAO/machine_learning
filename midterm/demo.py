"""Local-only prediction demo, no external service or browser upload destination."""
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from project import ROOT, LABELS, CLASSES, load_model, transform, seed_all

PAGE = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>垃圾圖片分類</title>
<style>body{font-family:system-ui,sans-serif;background:#eef2ed;color:#173c32;margin:0;padding:40px}
main{max-width:1280px;margin:auto}h1{font-size:42px}p{line-height:1.7}button,input{font-size:18px;padding:12px}
button{background:#176b50;color:white;border:0;border-radius:6px;cursor:pointer}button:disabled{opacity:.5}
.grid{display:grid;grid-template-columns:minmax(240px,.9fr) minmax(0,2fr);gap:24px;margin-top:28px}img{width:100%;max-height:420px;object-fit:contain;background:white}
#results{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;align-items:start}
article{background:white;padding:18px;margin-bottom:16px;border-radius:8px}h2{margin-top:0;font-size:22px;min-height:58px}.row{display:flex;justify-content:space-between;margin:12px 0}
progress{width:100%;accent-color:#176b50}small{color:#52635c}@media(max-width:900px){.grid{grid-template-columns:1fr}body{padding:20px}}@media(max-width:650px){#results{grid-template-columns:1fr}}</style>
<main><h1>垃圾圖片分類</h1><p>CNN、ResNet18、DNN-TC 架構比較版</p>
<input id="file" type="file" accept="image/*"><button id="go" disabled>比較三個模型</button>
<p id="status" role="status">請選擇一張只有一個主要物件的圖片。</p>
<div class="grid"><div><img id="preview" alt="待分類的圖片" hidden></div><div id="results"></div></div>
<p><small>分類範圍：紙板、玻璃、金屬、紙張、塑膠、其他垃圾。分數是 softmax 輸出，未經機率校準。<br>
模型一定會選擇六類之一，無法判斷所有未知物品；結果不是當地垃圾回收規定。</small></p></main>
<script>let data;const file=document.querySelector('#file'),go=document.querySelector('#go'),status=document.querySelector('#status');
file.onchange=()=>{go.disabled=true;document.querySelector('#results').replaceChildren();const f=file.files[0];if(!f)return;
if(f.size>8*1024*1024){status.textContent='請選擇小於 8 MB 的圖片';return;}const r=new FileReader();r.onload=()=>{data=r.result;
const p=document.querySelector('#preview');p.src=data;p.hidden=false;go.disabled=false;status.textContent='圖片已載入，可以開始比較。';};r.readAsDataURL(f);};
go.onclick=async()=>{go.disabled=true;status.textContent='模型預測中…';try{const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:data})});
const j=await r.json();if(!r.ok)throw Error(j.error);const box=document.querySelector('#results');box.replaceChildren();
for(const model of j){const card=document.createElement('article');const title=document.createElement('h2');title.textContent=model.model;card.append(title);
for(const item of model.top){const row=document.createElement('div');row.className='row';row.textContent=item.label+'　'+(item.score*100).toFixed(1)+'%';card.append(row);
const bar=document.createElement('progress');bar.max=1;bar.value=item.score;card.append(bar);}box.append(card);}status.textContent='預測完成。觀察三個模型是否一致。';
}catch(e){status.textContent='無法預測：'+e.message;}finally{go.disabled=false;}};</script></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/':
            self.send_error(404); return
        raw = PAGE.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers(); self.wfile.write(raw)

    def do_POST(self):
        if self.path != '/predict':
            self.send_error(404); return
        try:
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 12 * 1024 * 1024:
                raise ValueError('圖片檔案過大或空白')
            payload = json.loads(self.rfile.read(size))
            raw = base64.b64decode(payload['image'].split(',', 1)[1], validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                if image.width * image.height > 25_000_000:
                    raise ValueError('圖片解析度過大')
                x = transform()(ImageOps.exif_transpose(image).convert('RGB')).unsqueeze(0)
            result = []
            with torch.inference_mode():
                for name, net in NETWORKS.items():
                    scores, indices = net(x).softmax(1)[0].topk(3)
                    result.append({'model': name, 'top': [{'label': LABELS[i], 'class': CLASSES[i], 'score': float(s)}
                                                         for s, i in zip(scores, indices)]})
            self.respond(200, result)
        except (ValueError, TypeError, KeyError, IndexError, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            self.respond(400, {'error': str(exc)})

    def respond(self, status, payload):
        raw = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers(); self.wfile.write(raw)


if __name__ == '__main__':
    seed_all(42)
    NETWORKS = {'自建 CNN': load_model('cnn')[0], 'ResNet18 遷移學習': load_model('resnet18')[0],
                'DNN-TC 架構比較版': load_model('dnn_tc')[0]}
    print('Open http://localhost:8765  (Ctrl+C to stop)', flush=True)
    HTTPServer(('127.0.0.1', 8765), Handler).serve_forever()
