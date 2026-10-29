"""Local upload demo for CNN224 and ResNet101. Images stay in memory."""
import argparse
import io
import json
import threading
import time
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
import subprocess
import torch
from torch import nn
from torchvision.models import resnet101, ResNet101_Weights
from PIL import Image, ImageOps, UnidentifiedImageError
from CNN import CNN

ROOT = Path(__file__).resolve().parent
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
LABELS = ['紙板', '玻璃', '金屬', '紙張', '塑膠', '其他垃圾']
MAX_BYTES = 8 * 1024 * 1024
LOCK = threading.Lock()


class Predictor:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        torch.set_num_threads(4)
        self.transform = ResNet101_Weights.IMAGENET1K_V2.transforms()
        self.models, self.info = [], []
        for key, name, folder, build, checkpoint_name in [
            ('cnn224', 'CNN 224', 'cnn224_25epochs', lambda: CNN(6), 'best_cnn_trashnet.pth'),
            ('resnet101', 'ResNet101', 'resnet101_25epochs', lambda: resnet101(weights=None), 'best_resnet101_batch32_AdamW_Dropout.pth'),
        ]:
            path = ROOT / 'result' / folder
            metrics = json.loads((path / 'test.json').read_text())
            state = torch.load(path / checkpoint_name, map_location='cpu', weights_only=True)
            assert state['classes'] == CLASSES and state['image_size'] == 224
            assert metrics['config']['epochs'] == 25 and metrics['input_size'] == [224, 224]
            if key == 'cnn224':
                assert state['architecture'] == 'CNN224'
            model = build()
            if key == 'resnet101':
                model.fc = nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Linear(model.fc.in_features, len(CLASSES))
                )
            model.load_state_dict(state['model_state'], strict=True)
            model.eval().to(self.device)
            self.models.append(model)
            self.info.append({'id': key, 'name': name, 'best_epoch': state['epoch'],
                              'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1'],
                              'test_loss': metrics['loss'], 'trained_epochs': 25})
        split = json.loads((ROOT / 'data/splits.json').read_text())
        self.samples = [
            [r for r in split['test'] if r['label'] == i]
            for i in range(6)
        ]

    def predict(self, raw):
        with Image.open(io.BytesIO(raw)) as im:
            if im.width * im.height > 25_000_000:
                raise ValueError('圖片解析度上限為 2,500 萬像素')
            im = ImageOps.exif_transpose(im).convert('RGB')
            size = list(im.size)
            x = self.transform(im).unsqueeze(0).to(self.device)
        results = []
        with LOCK, torch.inference_mode():
            for meta, model in zip(self.info, self.models):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                probs = model(x).softmax(1)[0].cpu().tolist()
                elapsed = (time.perf_counter() - start) * 1000
                scores = [{'class': c, 'label': label, 'score': float(v)} for c, label, v in zip(CLASSES, LABELS, probs)]
                scores.sort(key=lambda v: v['score'], reverse=True)
                results.append({**meta, 'scores': scores, 'elapsed_ms': round(elapsed, 1)})
        return {'results': results, 'image_size': size, 'input_size': [224, 224],
                'agreement': results[0]['scores'][0]['class'] == results[1]['scores'][0]['class']}


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, data, content_type='application/json; charset=utf-8'):
        raw = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path == '/':
            self.reply(200, (ROOT/'compare_web/dist/index.html').read_bytes(), 'text/html; charset=utf-8')
        elif url.path == '/api/info':
            self.reply(200, {'models': self.server.predictor.info, 'device': str(self.server.predictor.device),
                             'samples': [{'id': i, 'label': LABELS[i]} for i in range(6)]})
        elif url.path == '/api/sample':
            try:
                idx = int(parse_qs(url.query).get('id', [''])[0])
                if not 0 <= idx < 6:
                    raise ValueError()
                sample = random.choice(self.server.predictor.samples[idx])
                print(f"Selected: {sample['path']}", flush=True)
                self.reply(200, (ROOT/'data'/sample['path']).read_bytes(), 'image/jpeg')
            except (ValueError, IndexError):
                self.reply(400, {'error': '找不到此範例'})
        elif url.path == '/doro.mp4':
            video_path = ROOT / 'compare_web/dist/doro.mp4'
            if video_path.is_file():
                self.reply(200, video_path.read_bytes(), 'video/mp4')
            else:
                self.reply(404, {'error': '找不到影片'})
        elif url.path == '/style.css':
            css_path = ROOT / 'compare_web/dist/style.css'

            if css_path.is_file():
                self.reply(
                    200,
                    css_path.read_bytes(),
                    'text/css; charset=utf-8'
                )
            else:
                self.reply(404, {'error': '找不到 CSS 檔案'})
        else:
            self.reply(404, {'error': '找不到此頁面'})

    def do_POST(self):
        if self.path != '/api/predict':
            self.reply(404, {'error': '找不到此端點'})
            return
        try:
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= MAX_BYTES:
                raise ValueError('請選擇 8 MB 以內的圖片')
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError('圖片傳輸不完整')
            self.reply(200, self.server.predictor.predict(raw))
        except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError):
            self.reply(400, {'error': '無法讀取圖片，請使用 8 MB 以內的 JPG、PNG 或 WebP，且不超過 2,500 萬像素。'})
        except Exception:
            import traceback
            traceback.print_exc()
            self.reply(500, {'error': '模型推論失敗，請查看伺服器紀錄後重試。'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()

    predictor = Predictor(args.device)

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.predictor = predictor

    url = f'http://localhost:{args.port}'

    print(f'Ready: {url} ({args.device})', flush=True)

    subprocess.Popen(
        ['cmd.exe', '/c', 'start', '', url],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
