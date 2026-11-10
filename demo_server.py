#!/usr/bin/env python3
"""Local API server for the InsureScan browser UI."""
from __future__ import annotations
import json, os, sys, tempfile
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"backend"))
from app.intelligence.pipeline import analyze_image

UI=(ROOT/"frontend"/"index.html").read_bytes()
MAX_UPLOAD=15*1024*1024
ALLOWED={".png",".jpg",".jpeg",".tif",".tiff"}

class Handler(BaseHTTPRequestHandler):
    def send_bytes(self,status,content_type,body):
        self.send_response(status)
        self.send_header("Content-Type",content_type)
        self.send_header("Content-Length",str(len(body)))
        self.send_header("Cache-Control","no-store")
        self.end_headers()
        self.wfile.write(body)
    def send_json(self,status,payload):
        self.send_bytes(status,"application/json; charset=utf-8",json.dumps(payload,ensure_ascii=False).encode())
    def do_GET(self):
        path=urlparse(self.path).path
        if path in ("/","/index.html"):
            self.send_bytes(200,"text/html; charset=utf-8",UI)
        elif path=="/health":
            self.send_json(200,{"status":"ok","service":"insurescan-local"})
        else:
            self.send_json(404,{"error":"Not found"})
    def do_POST(self):
        if urlparse(self.path).path!="/api/analyze":
            self.send_json(404,{"error":"Not found"}); return
        try:
            length=int(self.headers.get("Content-Length","0"))
            if length<=0: raise ValueError("No file was uploaded.")
            if length>MAX_UPLOAD+1024*1024: raise ValueError("File exceeds 15 MB limit.")
            content_type=self.headers.get("Content-Type","")
            raw=self.rfile.read(length)
            message=BytesParser(policy=default).parsebytes(
                b"Content-Type: "+content_type.encode()+b"\r\nMIME-Version: 1.0\r\n\r\n"+raw)
            if not message.is_multipart(): raise ValueError("Upload must use multipart form data.")
            upload=None
            for part in message.iter_parts():
                if part.get_param("name",header="content-disposition")=="document":
                    upload=part; break
            if upload is None: raise ValueError("Select a document image first.")
            filename=upload.get_filename() or "upload.png"
            suffix=Path(filename).suffix.lower()
            if suffix not in ALLOWED: raise ValueError("Only PNG, JPG and TIFF files are supported.")
            data=upload.get_payload(decode=True) or b""
            if not data: raise ValueError("The uploaded file is empty.")
            if len(data)>MAX_UPLOAD: raise ValueError("File exceeds 15 MB limit.")
            with tempfile.NamedTemporaryFile(suffix=suffix,delete=False) as f:
                f.write(data); temp=Path(f.name)
            try:
                result=analyze_image(temp)
                self.send_json(200,result)
            finally:
                temp.unlink(missing_ok=True)
        except Exception as exc:
            self.send_json(400,{"error":str(exc)})
    def log_message(self,fmt,*args):
        print("%s - %s"%(self.address_string(),fmt%args))

if __name__=="__main__":
    host=os.environ.get("HOST","0.0.0.0")
    port=int(os.environ.get("PORT","8000"))
    print(f"InsureScan UI running at http://{host}:{port}",flush=True)
    ThreadingHTTPServer((host,port),Handler).serve_forever()
