import cgi
import io
import json
import os
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from web_pipeline import (
    analyze_uploaded_data,
    analyze_variable,
    describe_dataset,
    df_to_records,
    load_reference_training_data,
    ordered_numeric_columns,
    predict_new_data,
    train_models_for_website,
)


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
BUILTIN_TRAINING_DATA = ROOT / "data" / "data.csv"


def json_response(handler, payload, status=200):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def file_response(handler, path: Path, content_type: str):
    data = path.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Cache-Control", "no-store, max-age=0")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def read_dataframe_from_form(form, field_name, default_path=None, required=False):
    upload = form[field_name] if field_name in form else None
    if upload is not None and getattr(upload, "file", None) and getattr(upload, "filename", ""):
        raw_bytes = upload.file.read()
        if not raw_bytes:
            raise ValueError("上传文件为空。")
        csv_text = raw_bytes.decode("utf-8-sig")
        return pd.read_csv(io.StringIO(csv_text)), "uploaded"

    if default_path is not None:
        return pd.read_csv(default_path), "builtin"

    if required:
        raise ValueError("请先上传预测集 CSV 文件。")

    return None, "missing"


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            return file_response(self, STATIC_DIR / "index.html", "text/html; charset=utf-8")
        if path == "/static/styles.css":
            return file_response(self, STATIC_DIR / "styles.css", "text/css; charset=utf-8")
        if path == "/static/app.js":
            return file_response(self, STATIC_DIR / "app.js", "application/javascript; charset=utf-8")
        if path == "/api/reference":
            train_df, feature_columns = load_reference_training_data()
            builtin_df = pd.read_csv(BUILTIN_TRAINING_DATA)
            return json_response(
                self,
                {
                    "feature_columns": feature_columns,
                    "model_training_rows": int(len(train_df)),
                    "builtin_training_rows": int(len(builtin_df)),
                    "target_column": "yield",
                    "builtin_data": "data/data.csv",
                },
            )

        return json_response(self, {"error": "Not found"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path not in ["/api/preview", "/api/predict-preview", "/api/predict", "/api/train-models", "/api/analyze", "/api/describe", "/api/variable"]:
            return json_response(self, {"error": "Not found"}, status=404)

        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                },
            )
            env = form.getvalue("environment", "train")
            train_df, train_source = read_dataframe_from_form(
                form,
                "train_file",
                default_path=BUILTIN_TRAINING_DATA,
            )

            if parsed.path == "/api/preview":
                if env == "test":
                    dataframe, source = read_dataframe_from_form(form, "predict_file", required=True)
                else:
                    dataframe, source = train_df, train_source
                cleaned = dataframe.drop(columns=["id"], errors="ignore")
                result = {
                    "dataset": {
                        "rows": int(len(cleaned)),
                        "columns": int(cleaned.shape[1]),
                        "numeric_columns": ordered_numeric_columns(cleaned),
                        "has_target": "yield" in cleaned.columns,
                    },
                    "preview": df_to_records(cleaned, limit=8),
                }
                result["data_source"] = source
            elif parsed.path == "/api/predict-preview":
                predict_df, predict_source = read_dataframe_from_form(
                    form,
                    "predict_file",
                    required=True,
                )
                train_columns = [col for col in train_df.drop(columns=["id"], errors="ignore").columns if col != "yield"]
                pred_cleaned = predict_df.drop(columns=["id", "yield"], errors="ignore")
                missing = [col for col in train_columns if col not in pred_cleaned.columns]
                predict_columns = pred_cleaned.columns.tolist()
                field_check = [
                    {
                        "feature": col,
                        "required": True,
                        "present": col in pred_cleaned.columns,
                    }
                    for col in train_columns
                ]
                result = {
                    "dataset": {
                        "rows": int(len(pred_cleaned)),
                        "columns": int(pred_cleaned.shape[1]),
                        "required_features": train_columns,
                        "present_features": [col for col in train_columns if col in pred_cleaned.columns],
                        "missing_features": missing,
                        "extra_features": [col for col in predict_columns if col not in train_columns],
                        "ready": len(missing) == 0,
                    },
                    "field_check": field_check,
                    "preview": df_to_records(predict_df, limit=8),
                    "normalized_preview": df_to_records(pred_cleaned[[col for col in train_columns if col in pred_cleaned.columns]], limit=8),
                    "predict_source": predict_source,
                }
            elif parsed.path == "/api/analyze":
                result = analyze_uploaded_data(train_df)
            elif parsed.path == "/api/train-models":
                result = train_models_for_website(
                    training_data_df=train_df,
                    use_builtin_model=train_source == "builtin",
                    model_choice=form.getvalue("model_choice", "both"),
                    xgb_params_text=form.getvalue("xgb_params", ""),
                    stacking_params_text=form.getvalue("stacking_params", ""),
                    run_xgb=form.getvalue("run_xgb", "true") == "true",
                    run_stacking=form.getvalue("run_stacking", "true") == "true",
                )
            elif parsed.path == "/api/describe":
                if env == "test":
                    dataframe, source = read_dataframe_from_form(form, "predict_file", required=True)
                else:
                    dataframe, source = train_df, train_source
                result = describe_dataset(dataframe)
                result["data_source"] = source
            elif parsed.path == "/api/variable":
                if env == "test":
                    dataframe, source = read_dataframe_from_form(form, "predict_file", required=True)
                else:
                    dataframe, source = train_df, train_source
                variable = form.getvalue("variable", "")
                if not variable:
                    numeric_cols = ordered_numeric_columns(dataframe.drop(columns=["id"], errors="ignore"))
                    if not numeric_cols:
                        raise ValueError("当前数据没有可分析的数值变量。")
                    variable = numeric_cols[0]
                result = analyze_variable(dataframe, variable)
                result["data_source"] = source
            else:
                predict_df, predict_source = read_dataframe_from_form(
                    form,
                    "predict_file",
                    required=True,
                )
                result = predict_new_data(
                    new_data_df=predict_df,
                    training_data_df=train_df,
                    use_builtin_model=train_source == "builtin",
                    model_choice=form.getvalue("model_choice", "stacking"),
                    xgb_params_text=form.getvalue("xgb_params", ""),
                    stacking_params_text=form.getvalue("stacking_params", ""),
                    run_xgb=form.getvalue("run_xgb", "true") == "true",
                    run_stacking=form.getvalue("run_stacking", "true") == "true",
                    xgb_use_pca=False,
                    stacking_use_pca=False,
                )
                result["predict_source"] = predict_source

            result["train_source"] = train_source
            return json_response(self, {"ok": True, "data": result})
        except Exception as exc:
            return json_response(
                self,
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                status=400,
            )

    def log_message(self, format, *args):
        return


def run(host="127.0.0.1", port=8000):
    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"Server running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run(host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8000")))
