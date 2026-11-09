# =========================================================
# SERVER CHAY LOCAL CHO TRANG WEB DU DOAN CREDIT SCORING
# Khong can cai them thu vien ngoai (khong dung Flask) -
# chi dung http.server co san trong Python.
#
# Chay:  python server.py
# Mo trinh duyet: http://localhost:8765/
# =========================================================

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from predict import predict_customer, predict_batch

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "predict_ui.html"
PORT = 8765


class Handler(BaseHTTPRequestHandler):

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        return json.loads(raw.decode("utf-8")) if raw else {}

    def log_message(self, format, *args):
        pass  # im lang, khong in log request ra console

    # ---------------- GET ----------------
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html = INDEX_FILE.read_text(encoding="utf-8")
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    # ---------------- POST ----------------
    def do_POST(self):
        try:
            if self.path == "/api/predict":
                customer = self._read_json_body()
                result = predict_customer(customer)
                self._send_json(200, {"ok": True, "result": result})

            elif self.path == "/api/predict_batch":
                payload = self._read_json_body()
                rows = payload.get("rows", [])
                if not rows:
                    raise ValueError("Khong co du lieu khach hang nao de du doan.")

                ids = [row.pop("Customer_ID", None) for row in rows]
                df = pd.DataFrame(rows)
                results_df = predict_batch(df)

                results = []
                for cid, (_, row) in zip(ids, results_df.iterrows()):
                    results.append({
                        "Customer_ID": cid,
                        "Default_Probability": float(row["Default_Probability"]),
                        "Credit_Score": int(row["Credit_Score"]),
                        "Risk_Group": str(row["Risk_Group"]),
                        "Predicted_Default": int(row["Predicted_Default"]),
                    })
                self._send_json(200, {"ok": True, "results": results})

            else:
                self._send_json(404, {"ok": False, "error": "Khong tim thay API."})

        except Exception as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})


def main():
    server = ThreadingHTTPServer(("localhost", PORT), Handler)
    print(f"Server dang chay tai: http://localhost:{PORT}/")
    print("Nhan Ctrl+C de dung server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDa dung server.")


if __name__ == "__main__":
    main()
