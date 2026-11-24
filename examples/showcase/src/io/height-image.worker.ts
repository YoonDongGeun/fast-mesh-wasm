// The height image Worker: encodes height maps as PNGs and reads images as
// grids of heights, off the main thread (see `height-image.ts`).
import { HeightImageEngine, type HeightImageReply, type HeightImageRequest } from "./height-image";

const engine = new HeightImageEngine();

async function run(req: HeightImageRequest): Promise<unknown> {
  switch (req.op) {
    case "encode":
      return engine.encode(req.heights, req.width, req.height, req.range);
    case "open":
      return engine.open(req.blob);
    case "grid":
      return engine.grid(req.id, req.w, req.h, req.options);
    case "close":
      engine.close(req.id);
      return undefined;
  }
}

self.onmessage = async (event: MessageEvent<HeightImageRequest>) => {
  const req = event.data;
  let reply: HeightImageReply;
  const transfer: Transferable[] = [];
  try {
    const value = await run(req);
    if (ArrayBuffer.isView(value) && value.buffer instanceof ArrayBuffer) {
      transfer.push(value.buffer);
    }
    reply = { n: req.n, ok: true, value };
  } catch (error) {
    reply = { n: req.n, ok: false, message: error instanceof Error ? error.message : String(error) };
  }
  self.postMessage(reply, { transfer });
};
