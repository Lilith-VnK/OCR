## OCR

Lightweight OCR API built with Flask and Tesseract.
Accepts a single image upload and returns extracted text in JSON format.

---

## Running the Server

Make sure Tesseract is installed and Python dependencies are available.

Run:

```python app.py```

If successful, the server will start at:

```http://127.0.0.1:5000```

The server provides only one endpoint and accepts POST requests only.

---

## Endpoint

```POST /ocr```

Uploads one image and returns the detected text.

---

Request Format

The request must use:

multipart/form-data

Required field:

Key| Type| Required
image| File| Yes

The field name must be "image".
Any other key name will be rejected.

---

## How to Use

1. Using CURL (Terminal)

```
curl -X POST http://127.0.0.1:5000/ocr -F "image=@test.png"
```

---

## 2. Using Postman / Insomnia

1. Method → POST
2. URL → http://127.0.0.1:5000/ocr
3. Open Body tab
4. Select form-data
5. Add:

Key| Type| Value
image| File| choose an image

Press Send.

---

## 3. Using Python

```
import requests

url = "http://127.0.0.1:5000/ocr"

with open("test.png", "rb") as f:
    r = requests.post(url, files={"image": f})

print(r.json())
```
---

## Response

Success
```
{
  "ok": true,
  "text": "HELLO WORLD"
}

No File Provided

{
  "ok": false,
  "msg": "no file"
}

Invalid Image / Read Error

{
  "ok": false,
  "msg": "cannot identify image file"
}
```

---

Limitations

- Only one image per request
- No permanent file storage
- No authentication
- No rate limiting
- Recommended formats: PNG or JPG

---

Accuracy Tips

For best results, use images with:

- clear text
- no blur
- straight orientation
- high contrast (dark text on light background)

---

Works Best For

- chat screenshots
- documents
- exam questions
- subtitles

Not Ideal For

- handwriting
- distant whiteboard photos
- decorative fonts
