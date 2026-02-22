import os, json, hashlib
import cv2
import numpy as np
import pytesseract
from flask import Flask, request, jsonify

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

CACHE_FOLDER = "cache"
if not os.path.isdir(CACHE_FOLDER):
    os.makedirs(CACHE_FOLDER)

TEXT_CFG = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
NUM_CFG  = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.,:-'


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _decode(b):
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)


def _norm(txt):
    return "\n".join([i.strip() for i in txt.splitlines() if i.strip()])


def _rotate(img):
    try:
        d = pytesseract.image_to_osd(img)
        rot = [x for x in d.split("\n") if "Rotate" in x][0]
        ang = int(rot.split(":")[1].strip())
        if ang != 0:
            h, w = img.shape[:2]
            m = cv2.getRotationMatrix2D((w//2, h//2), -ang, 1)
            img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except:
        pass
    return img


def _scale(img):
    h, w = img.shape[:2]
    m = max(h, w)
    if m < 900:
        return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if m < 1500:
        return cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    return img


def _illum(gray):
    bg = cv2.medianBlur(gray, 51)
    d = 255 - cv2.absdiff(gray, bg)
    return cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)


def _bin(gray):
    clahe = cv2.createCLAHE(3.0, (8,8))
    g = clahe.apply(gray)
    g = cv2.fastNlMeansDenoising(g, None, 25, 7, 21)
    t1 = cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    t2 = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,2)
    r = np.count_nonzero(t1==0)/t1.size
    return t1 if 0.03 < r < 0.5 else t2


def _deskew(img):
    pts = np.column_stack(np.where(img < 255))
    if len(pts) < 400:
        return img
    ang = cv2.minAreaRect(pts)[-1]
    ang = -(90 + ang) if ang < -45 else -ang
    if abs(ang) < 0.25:
        return img
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w//2,h//2),ang,1)
    return cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _detect(img):
    g = _gray(img)
    lap = cv2.Laplacian(g, cv2.CV_64F).var()
    edge = cv2.Canny(g,80,160)
    dens = np.count_nonzero(edge)/edge.size
    return "screenshot" if (lap>130 and dens>0.045) else "document"


def _pre(img, mode):
    img = _scale(_rotate(img))
    g = _illum(_gray(img))
    b = _bin(g)
    if mode != "number":
        b = _deskew(b)
    return b


def _boxes(binary):
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(25,5))
    d = cv2.dilate(binary,k,1)
    c,_ = cv2.findContours(d,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    out=[]
    for i in c:
        x,y,w,h=cv2.boundingRect(i)
        if w*h>1500 and h>15:
            out.append((x,y,w,h))
    return sorted(out,key=lambda b:(b[1],b[0]))


def _ocr(image,mode):
    cfg = NUM_CFG if mode=="number" else TEXT_CFG
    lang = None if mode=="number" else "ind+eng"
    d = pytesseract.image_to_data(image,lang=lang,config=cfg,output_type=pytesseract.Output.DICT)
    w=[]; c=[]
    for t,cf in zip(d["text"],d["conf"]):
        if t.strip() and cf!="-1":
            w.append(t)
            c.append(int(cf))
    txt=" ".join(w)
    avg=sum(c)/len(c) if c else 0
    return _norm(txt),avg


def _core(img,mode):
    t,c=_ocr(img,mode)
    if c<70:
        inv=255-img
        t2,c2=_ocr(inv,mode)
        if c2>c:
            return t2,round(c2,2)
    return t,round(c,2)


@app.route("/ocr",methods=["POST"])
def api():
    if "image" not in request.files:
        return jsonify({"error":"no image"}),400

    mode=request.form.get("mode","text")
    b=request.files["image"].read()
    if not b:
        return jsonify({"error":"empty file"}),400

    h=_hash(b)
    cf=os.path.join(CACHE_FOLDER,h+".json")

    if os.path.exists(cf):
        with open(cf,"r",encoding="utf-8") as f:
            d=json.load(f)
            d["cached"]=True
            return jsonify(d)

    img=_decode(b)
    if img is None:
        return jsonify({"error":"invalid image"}),400

    t=_detect(img)
    p=_pre(img,mode)

    if t=="document" and mode!="number":
        parts=[]; confs=[]
        for x,y,w,h in _boxes(p):
            crop=p[y:y+h,x:x+w]
            tx,cf=_core(crop,mode)
            if tx:
                parts.append(tx)
                confs.append(cf)
        text="\n".join(parts)
        conf=sum(confs)/len(confs) if confs else 0
    else:
        text,conf=_core(p,mode)

    res={"cached":False,"type":t,"confidence":round(conf,2),"text":text}

    with open(cf,"w",encoding="utf-8") as f:
        json.dump(res,f,ensure_ascii=False)

    return jsonify(res)


if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000,threaded=True)
