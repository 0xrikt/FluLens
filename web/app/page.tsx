"use client";

import { useState } from "react";

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

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        throw new Error("Prediction failed");
      }
      const data = await res.json();
      setResult({ label: data.label, probability: data.probability });
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
