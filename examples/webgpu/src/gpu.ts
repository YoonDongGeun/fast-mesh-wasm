// The WebGPU side: device, pipelines, per-mesh GPU buffers kept in sync with
// the scene's change records, and drawing. Everything scene-specific goes
// through `applyChange`: buffers come from `scene.views(mesh)` and only dirty
// ranges are written with `queue.writeBuffer`.
import type { MeshChange, MeshHandle, Range, SceneLike } from "fast-mesh-wasm";
import shaderCode from "./shader.wgsl?raw";

/** What one draw needs besides the mesh's buffers. */
export interface DrawItem {
  mesh: MeshHandle;
  /** RGB in 0..1, used while the scene does not colour the mesh. */
  baseColor: readonly [number, number, number];
  opacity: number;
  picked: boolean;
}

export interface FrameParams {
  viewProj: Float32Array;
  eye: readonly [number, number, number];
  /** `[nx, ny, nz, -dot(n, origin)]`: the half-space cut away, or `null`. */
  clip: readonly [number, number, number, number] | null;
}

/** One mesh's GPU buffers. Vertex buffers are slot-aligned with the views. */
interface GpuMesh {
  positions: GPUBuffer;
  normals: GPUBuffer;
  /**
   * RGBA8 per vertex slot. Always allocated (4 bytes per vertex), so one
   * pipeline fits every mesh: while colouring is off the shader ignores it
   * (`params.y = 0`) and uses the base colour instead.
   */
  colors: GPUBuffer;
  indices: GPUBuffer;
  indexCount: number;
  hasColors: boolean;
  uniform: GPUBuffer;
  bindGroup: GPUBindGroup;
}

const DEPTH_FORMAT: GPUTextureFormat = "depth24plus";
const FRAME_BYTES = 112; // mat4 + 3 vec4
const OBJECT_BYTES = 96; // mat4 + 2 vec4
// Buffer usages, read lazily: `GPUBufferUsage` only exists where WebGPU does.
const VERTEX = () => GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST;
const INDEX = () => GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST;

export class Renderer {
  readonly #device: GPUDevice;
  readonly #context: GPUCanvasContext;
  readonly #canvas: HTMLCanvasElement;
  readonly #opaque: GPURenderPipeline;
  readonly #transparent: GPURenderPipeline;
  readonly #lines: GPURenderPipeline;
  readonly #objectLayout: GPUBindGroupLayout;
  readonly #frameUniform: GPUBuffer;
  readonly #frameBindGroup: GPUBindGroup;
  readonly #meshes = new Map<MeshHandle, GpuMesh>();
  #depth: GPUTexture | undefined;
  /** Reused every frame for the uniforms, so rendering allocates no arrays. */
  readonly #frameScratch = new Float32Array(FRAME_BYTES / 4);
  readonly #objectScratch = new Float32Array(OBJECT_BYTES / 4);
  #lineBuffer: GPUBuffer | undefined;
  #lineVertices = 0;

  /** A renderer for `canvas`, or `null` when WebGPU is unavailable. */
  static async create(canvas: HTMLCanvasElement): Promise<Renderer | null> {
    // `navigator.gpu` is missing without WebGPU (or outside a secure context).
    if (!navigator.gpu) {
      return null;
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter === null) {
      return null;
    }
    const device = await adapter.requestDevice();
    const context = canvas.getContext("webgpu");
    if (context === null) {
      return null;
    }
    return new Renderer(device, context, canvas);
  }

  private constructor(device: GPUDevice, context: GPUCanvasContext, canvas: HTMLCanvasElement) {
    this.#device = device;
    this.#context = context;
    this.#canvas = canvas;
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: "opaque" });

    const visibility = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    const uniformLayout = () =>
      device.createBindGroupLayout({ entries: [{ binding: 0, visibility, buffer: {} }] });
    const frameLayout = uniformLayout();
    this.#objectLayout = uniformLayout();
    this.#frameUniform = device.createBuffer({
      size: FRAME_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.#frameBindGroup = device.createBindGroup({
      layout: frameLayout,
      entries: [{ binding: 0, resource: { buffer: this.#frameUniform } }],
    });

    const module = device.createShaderModule({ code: shaderCode });
    const meshLayout = device.createPipelineLayout({
      bindGroupLayouts: [frameLayout, this.#objectLayout],
    });
    const meshPipeline = (transparent: boolean) =>
      device.createRenderPipeline({
        layout: meshLayout,
        vertex: {
          module,
          entryPoint: "vs_mesh",
          buffers: [
            { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] },
            { arrayStride: 12, attributes: [{ shaderLocation: 1, offset: 0, format: "float32x3" }] },
            { arrayStride: 4, attributes: [{ shaderLocation: 2, offset: 0, format: "unorm8x4" }] },
          ],
        },
        fragment: {
          module,
          entryPoint: "fs_mesh",
          targets: [
            {
              format,
              blend: transparent
                ? {
                    color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
                    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
                  }
                : undefined,
            },
          ],
        },
        primitive: { topology: "triangle-list", cullMode: "none" },
        // Transparent meshes test depth but do not write it.
        depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: !transparent, depthCompare: "less" },
      });
    this.#opaque = meshPipeline(false);
    this.#transparent = meshPipeline(true);
    this.#lines = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [frameLayout] }),
      vertex: {
        module,
        entryPoint: "vs_line",
        buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] }],
      },
      fragment: { module, entryPoint: "fs_line", targets: [{ format }] },
      primitive: { topology: "line-list" },
      // Drawn last and on top, so contours on the cut stay visible.
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: "always" },
    });
  }

  /**
   * Brings one mesh's GPU buffers up to date with a change record from
   * `scene.commitFrame()`. Views are read fresh here and passed through
   * `.subarray()` (debug-mode views are Proxies, which writeBuffer rejects).
   */
  applyChange(scene: SceneLike, change: MeshChange): void {
    const old = this.#meshes.get(change.mesh);
    if (change.removed) {
      if (old !== undefined) {
        destroy(old);
        this.#meshes.delete(change.mesh);
      }
      return;
    }
    if (old === undefined || change.added || change.capacityChanged) {
      // New slot counts: new buffers, uploaded whole.
      if (old !== undefined) {
        destroy(old);
      }
      this.#meshes.set(change.mesh, this.#createMesh(scene, change.mesh));
      return;
    }
    const views = scene.views(change.mesh);
    const positions = views.positions();
    old.positions = this.#write(old.positions, VERTEX(), positions, change.positions, 3, positions.byteLength);
    const normals = views.normals();
    if (normals !== null) {
      // Normals can be short (lazy meshes): the buffer stays as big as the positions.
      old.normals = this.#write(old.normals, VERTEX(), normals, change.normals, 3, positions.byteLength);
    }
    const colors = views.colors();
    old.hasColors = colors !== null;
    if (colors !== null) {
      old.colors = this.#write(old.colors, VERTEX(), colors, change.colors, 4, colors.byteLength);
    }
    const indices = views.indices();
    old.indices = this.#write(old.indices, INDEX(), indices, change.indices, 3, indices.byteLength);
    old.indexCount = indices.length;
  }

  /** Sets the section contour lines: pairs of world points, 6 floats per segment. */
  setLines(segments: Float32Array): void {
    this.#lineBuffer?.destroy();
    this.#lineBuffer = undefined;
    this.#lineVertices = segments.length / 3;
    if (this.#lineVertices > 0) {
      this.#lineBuffer = this.#device.createBuffer({ size: segments.byteLength, usage: VERTEX() });
      this.#device.queue.writeBuffer(this.#lineBuffer, 0, segments);
    }
  }

  /** The canvas aspect ratio, after matching its size to the page. */
  resize(): number {
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(this.#canvas.clientWidth * dpr));
    const height = Math.max(1, Math.round(this.#canvas.clientHeight * dpr));
    if (this.#canvas.width !== width || this.#canvas.height !== height || this.#depth === undefined) {
      this.#canvas.width = width;
      this.#canvas.height = height;
      this.#depth?.destroy();
      this.#depth = this.#device.createTexture({
        size: [width, height],
        format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }
    return width / height;
  }

  /**
   * Draws the items: opaque ones first, then transparent ones back to front
   * by world centre, then the section lines. Each item's model matrix is
   * `scene.matrix4(mesh)` converted to f32.
   */
  render(scene: SceneLike, frame: FrameParams, items: readonly DrawItem[]): void {
    const device = this.#device;
    const f = this.#frameScratch;
    f.fill(0);
    f.set(frame.viewProj, 0);
    f.set(frame.eye, 16);
    if (frame.clip !== null) {
      f.set(frame.clip, 20);
      f[24] = 1;
    }
    device.queue.writeBuffer(this.#frameUniform, 0, f);

    const opaque: DrawItem[] = [];
    const transparent: { item: DrawItem; depth: number }[] = [];
    for (const item of items) {
      const gpu = this.#meshes.get(item.mesh);
      if (gpu === undefined) {
        continue;
      }
      const matrix = scene.matrix4(item.mesh);
      const u = this.#objectScratch;
      u.fill(0);
      u.set(matrix, 0); // f64 → f32
      u.set(item.baseColor, 16);
      u[19] = 1;
      u[20] = item.opacity;
      u[21] = gpu.hasColors ? 1 : 0;
      u[22] = item.picked ? 1 : 0;
      device.queue.writeBuffer(gpu.uniform, 0, u);
      if (item.opacity >= 1) {
        opaque.push(item);
      } else {
        // The last column of matrix4 is the world position of the local origin,
        // the mesh's bounding-box centre.
        const dx = matrix[12] - frame.eye[0];
        const dy = matrix[13] - frame.eye[1];
        const dz = matrix[14] - frame.eye[2];
        transparent.push({ item, depth: dx * dx + dy * dy + dz * dz });
      }
    }
    transparent.sort((a, b) => b.depth - a.depth);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.#context.getCurrentTexture().createView(),
          clearValue: { r: 0.11, g: 0.12, b: 0.14, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: this.#depth!.createView(),
        depthClearValue: 1,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });
    pass.setBindGroup(0, this.#frameBindGroup);
    const draw = (item: DrawItem) => {
      const gpu = this.#meshes.get(item.mesh)!;
      if (gpu.indexCount === 0) {
        return;
      }
      pass.setBindGroup(1, gpu.bindGroup);
      pass.setVertexBuffer(0, gpu.positions);
      pass.setVertexBuffer(1, gpu.normals);
      pass.setVertexBuffer(2, gpu.colors);
      pass.setIndexBuffer(gpu.indices, "uint32");
      // Removed triangles are (0, 0, 0): degenerate, so they draw nothing.
      pass.drawIndexed(gpu.indexCount);
    };
    pass.setPipeline(this.#opaque);
    opaque.forEach(draw);
    pass.setPipeline(this.#transparent);
    transparent.forEach(({ item }) => draw(item));
    if (this.#lineBuffer !== undefined) {
      pass.setPipeline(this.#lines);
      pass.setVertexBuffer(0, this.#lineBuffer);
      pass.draw(this.#lineVertices);
    }
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  /** Buffers sized from the views and uploaded whole. */
  #createMesh(scene: SceneLike, mesh: MeshHandle): GpuMesh {
    const views = scene.views(mesh);
    const positions = views.positions();
    const slots = positions.length / 3;
    const colors = views.colors();
    const indices = views.indices();
    const uniform = this.#device.createBuffer({
      size: OBJECT_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    return {
      positions: this.#upload(VERTEX(), positions.byteLength, positions),
      // Normals can be off or short (lazy meshes): keep the buffer full size.
      normals: this.#upload(VERTEX(), positions.byteLength, views.normals()),
      colors: this.#upload(VERTEX(), 4 * slots, colors),
      indices: this.#upload(INDEX(), indices.byteLength, indices),
      indexCount: indices.length,
      hasColors: colors !== null,
      uniform,
      bindGroup: this.#device.createBindGroup({
        layout: this.#objectLayout,
        entries: [{ binding: 0, resource: { buffer: uniform } }],
      }),
    };
  }

  /** A new buffer of `size` bytes holding `data` (or zeros). */
  #upload(usage: number, size: number, data: Float32Array | Uint8Array | Uint32Array | null): GPUBuffer {
    // A buffer may not be empty; writeBuffer sizes are multiples of 4 here.
    const buffer = this.#device.createBuffer({ size: Math.max(4, Math.ceil(size / 4) * 4), usage });
    if (data !== null && data.byteLength > 0) {
      const bytes = Math.min(data.byteLength, buffer.size);
      this.#device.queue.writeBuffer(buffer, 0, data.subarray(0, bytes / data.BYTES_PER_ELEMENT));
    }
    return buffer;
  }

  /**
   * Writes a change's ranges of `view` into `buffer`. `itemSize` is elements
   * per slot (3 floats, 4 colour bytes, 3 indices), so a range `[start, count]`
   * starts at byte `start · itemSize · bytesPerElement`. `"all"` writes the
   * whole view (at most `size` bytes), recreating the buffer if it is not
   * `size` bytes. `size` is fixed by the caller, not taken from the view, so a
   * short view never shrinks the buffer below the vertex count.
   */
  #write(
    buffer: GPUBuffer,
    usage: number,
    view: Float32Array | Uint8Array | Uint32Array,
    ranges: Range[] | "all",
    itemSize: number,
    size: number,
  ): GPUBuffer {
    if (ranges === "all") {
      if (buffer.size !== Math.max(4, Math.ceil(size / 4) * 4)) {
        buffer.destroy();
        return this.#upload(usage, size, view);
      }
      const bytes = Math.min(view.byteLength, size);
      if (bytes > 0) {
        this.#device.queue.writeBuffer(buffer, 0, view.subarray(0, bytes / view.BYTES_PER_ELEMENT));
      }
      return buffer;
    }
    const bytesPerSlot = itemSize * view.BYTES_PER_ELEMENT;
    for (const [start, count] of ranges) {
      if (count > 0) {
        const data = view.subarray(start * itemSize, (start + count) * itemSize);
        this.#device.queue.writeBuffer(buffer, start * bytesPerSlot, data);
      }
    }
    return buffer;
  }
}

function destroy(mesh: GpuMesh): void {
  mesh.positions.destroy();
  mesh.normals.destroy();
  mesh.colors.destroy();
  mesh.indices.destroy();
  mesh.uniform.destroy();
}
