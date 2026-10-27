"""Exercise HTTP upload, prediction response and invalid-image handling."""
import base64
import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
import demo
from project import DATA, OUT, CLASSES, load_model, seed_all

seed_all(42)
demo.NETWORKS = {name: load_model(name)[0] for name in ['cnn', 'resnet18', 'dnn_tc']}
server = HTTPServer(('127.0.0.1', 0), demo.Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
base = f'http://127.0.0.1:{server.server_port}'
try:
    with urllib.request.urlopen(base) as response:
        assert '垃圾圖片分類' in response.read().decode('utf-8')
    split = json.loads((DATA / 'splits.json').read_text())
    image = (DATA / split['test'][0]['path']).read_bytes()
    payload = json.dumps({'image': 'data:image/jpeg;base64,' + base64.b64encode(image).decode()}).encode()
    request = urllib.request.Request(base + '/predict', data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request) as response:
        result = json.load(response)
    assert len(result) == 3
    for model in result:
        assert len(model['top']) == 3
        assert all(item['class'] in CLASSES and 0 <= item['score'] <= 1 for item in model['top'])
        scores = [item['score'] for item in model['top']]
        assert scores == sorted(scores, reverse=True)
        saved = json.loads((OUT / model['model'] / 'test.json').read_text())['probabilities'][0]
        for item in model['top']:
            assert abs(item['score'] - saved[CLASSES.index(item['class'])]) < 1e-4
    malformed = urllib.request.Request(base + '/predict', data=b'{"image":"broken"}', headers={'Content-Type': 'application/json'})
    try:
        urllib.request.urlopen(malformed)
        raise AssertionError('Malformed image should fail')
    except urllib.error.HTTPError as error:
        assert error.code == 400
    print('PASS: page response, real image upload, three predictions, invalid image rejection')
finally:
    server.shutdown()
    server.server_close()
    thread.join()
