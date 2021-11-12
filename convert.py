import requests

def dwg2dxf (from_path, to_path):
    url = 'http://cyborg-dwg2dxf:8001/dwg2dxf/'
    with open(from_path, 'rb') as f:
        files = {'file': f}
        r = requests.post(url, files=files)
    with open(to_path, 'wb') as f:
        f.write(r.content)
        pass
    pass

