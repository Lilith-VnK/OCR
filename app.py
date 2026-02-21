from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
import os
import uuid

app = Flask(__name__)

if not os.path.exists("tmp"):
    os.mkdir("tmp")

@app.post("/ocr")
def read():
    f = request.files.get("image")
    if not f:
        return jsonify({"ok": False, "msg": "no file"})

    name = str(uuid.uuid4()) + ".png"
    path = "tmp/" + name
    f.save(path)

    try:
        im = Image.open(path)
        txt = pytesseract.image_to_string(im)
    except Exception as e:
        os.remove(path)
        return jsonify({"ok": False, "msg": str(e)})

    os.remove(path)
    return jsonify({"ok": True, "text": txt.strip()})


app.run("0.0.0.0",5000)