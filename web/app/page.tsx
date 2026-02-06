"use client";

import { useState } from "react";
import * as ort from "onnxruntime-web";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<null | {
    label: string;
    probability: number;
  }>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) {
      setError("Please select an image.");
      return;
    }
    setError(null);
    setResult(null);
    setLoading(true);

    try {
      ort.env.wasm.wasmPaths =
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";

      const session = await ort.InferenceSession.create(
        "/models/flulens.onnx"
      );

      const bitmap = await createImageBitmap(file);
      const canvas = document.createElement("canvas");
      canvas.width = 224;
      canvas.height = 224;
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Canvas unavailable");
      ctx.drawImage(bitmap, 0, 0, 224, 224);
      const imageData = ctx.getImageData(0, 0, 224, 224).data;

      const mean = [0.485, 0.456, 0.406];
      const std = [0.229, 0.224, 0.225];
      const floatData = new Float32Array(1 * 3 * 224 * 224);

      for (let i = 0; i < 224 * 224; i++) {
        const r = imageData[i * 4] / 255;
        const g = imageData[i * 4 + 1] / 255;
        const b = imageData[i * 4 + 2] / 255;
        floatData[i] = (r - mean[0]) / std[0];
        floatData[i + 224 * 224] = (g - mean[1]) / std[1];
        floatData[i + 2 * 224 * 224] = (b - mean[2]) / std[2];
      }

      const input = new ort.Tensor("float32", floatData, [1, 3, 224, 224]);
      const output = await session.run({ input });
      const logits = Array.from(output.logits.data as Float32Array);
      const max = Math.max(...logits);
      const exps = logits.map((v) => Math.exp(v - max));
      const sum = exps.reduce((a, b) => a + b, 0);
      const probs = exps.map((v) => v / sum);

      const label = probs[1] >= probs[0] ? "High Infection" : "Low Infection";
      const probability = Math.max(...probs);
      setResult({ label, probability });
    } catch (err) {
      setError("Prediction failed. Please try another image.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <div className="container">
        <header className="header fade-in">
          <span className="badge">FluLens Screening</span>
          <h1>Influenza Infection Intensity Lens</h1>
          <p className="sub">
            Upload a fluorescence microscopy image. The model screens for
            infection intensity based on a self-supervised attention backbone and
            returns a high vs. low infection score.
          </p>
        </header>

        <section className="grid fade-in">
          <div className="card upload">
            <h2>Run a Screen</h2>
            <p>Accepted formats: PNG, JPG, or TIFF converted to PNG.</p>
            <form className="input" onSubmit={handleSubmit}>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
              <button type="submit" disabled={loading}>
                {loading ? "Analyzing…" : "Predict"}
              </button>
            </form>
            {error && <p style={{ color: "#f2a3a3" }}>{error}</p>}
            {result && (
              <div className="result">
                <strong>{result.label}</strong>
                <span>Confidence: {(result.probability * 100).toFixed(1)}%</span>
              </div>
            )}
          </div>

          <div className="card">
            <h2>Model Snapshot</h2>
            <p>
              Backbone: ResNeSt-50 + channel & spatial attention. Self-supervised
              pretraining aligned to BYOL strategy.
            </p>
            <p>
              Output classes: High infection (top 20%) vs. Low infection (bottom
              20%).
            </p>
          </div>

          <div className="card">
            <h2>Evidence Trail</h2>
            <p>
              Dataset: IDR idr0128 influenza screen (Georgi et al.). The app
              consumes an exported ONNX model for server-side inference.
            </p>
          </div>
        </section>

        <footer className="footer fade-in">
          <span>FluLens • AI Influenza Screen</span>
          <span>Research prototype</span>
        </footer>
      </div>
    </main>
  );
}
