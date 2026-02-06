import { NextResponse } from "next/server";
import sharp from "sharp";
import path from "path";

export const runtime = "nodejs";

const modelPath = path.join(process.cwd(), "..", "models", "flulens.onnx");
let session: any = null;

async function getSession() {
  // Load onnxruntime-node at runtime to avoid bundling native binaries
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const ort = require("onnxruntime-node");
  if (!session) {
    session = await ort.InferenceSession.create(modelPath);
  }
  return session;
}

function softmax(logits: number[]) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

export async function POST(req: Request) {
  try {
    const formData = await req.formData();
    const file = formData.get("file") as File | null;
    if (!file) {
      return NextResponse.json({ error: "No file" }, { status: 400 });
    }

    const buffer = Buffer.from(await file.arrayBuffer());
    const image = await sharp(buffer)
      .resize(224, 224)
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: false });

    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    const floatData = new Float32Array(1 * 3 * 224 * 224);
    for (let i = 0; i < 224 * 224; i++) {
      const r = image[i * 3] / 255;
      const g = image[i * 3 + 1] / 255;
      const b = image[i * 3 + 2] / 255;
      floatData[i] = (r - mean[0]) / std[0];
      floatData[i + 224 * 224] = (g - mean[1]) / std[1];
      floatData[i + 2 * 224 * 224] = (b - mean[2]) / std[2];
    }

    const ort = require("onnxruntime-node");
    const tensor = new ort.Tensor("float32", floatData, [1, 3, 224, 224]);
    const sess = await getSession();
    const output = await sess.run({ input: tensor });
    const logits = Array.from(output.logits.data as Float32Array);
    const probs = softmax(logits);

    const label = probs[1] >= probs[0] ? "High Infection" : "Low Infection";
    const probability = Math.max(...probs);

    return NextResponse.json({ label, probability });
  } catch (err) {
    return NextResponse.json({ error: "Inference failed" }, { status: 500 });
  }
}
