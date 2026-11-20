"""Loopback JSON RPC, arrays encoded losslessly as NPY (no pickle network input)."""
import base64
import io
import json
import urllib.request
import urllib.error

class ModelError(RuntimeError): pass

def array_encode(x):
    import numpy as np
    buf=io.BytesIO();np.save(buf,np.asarray(x),allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def array_decode(x):
    import numpy as np
    return np.load(io.BytesIO(base64.b64decode(x,validate=True)),allow_pickle=False)

def call(url, route, data, timeout=1800):
    if not url.startswith("http://127.0.0.1:"):
        raise ValueError("Only loopback model servers are supported")
    req=urllib.request.Request(url+route,data=json.dumps(data,allow_nan=False).encode(),
                               headers={"Content-Type":"application/json"},method="POST")
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        detail=exc.read().decode("utf-8",errors="replace")
        raise ModelError("Model service rejected the request: "+detail) from exc
