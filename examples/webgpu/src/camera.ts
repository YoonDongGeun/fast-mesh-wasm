// A small orbit camera: view-projection matrix (column-major, WebGPU depth
// 0..1) and pointer rays for picking. Plus the few vector helpers main.ts uses.
import type { Vec3 } from "fast-mesh-wasm";

export const add = (a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a: Readonly<Vec3>, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a: Readonly<Vec3>, b: Readonly<Vec3>): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const normalize = (a: Readonly<Vec3>): Vec3 => scale(a, 1 / (Math.hypot(a[0], a[1], a[2]) || 1));

/** Where the camera is and which way its axes point (world space). */
export interface CameraBasis {
  eye: Vec3;
  forward: Vec3;
  right: Vec3;
  up: Vec3;
}

export class OrbitCamera {
  target: Vec3 = [0, 0, 0];
  distance = 8;
  yaw = 0.5;
  pitch = 0.5;
  readonly fovY = Math.PI / 4;

  basis(): CameraBasis {
    const c = Math.cos(this.pitch);
    const offset: Vec3 = [c * Math.sin(this.yaw), Math.sin(this.pitch), c * Math.cos(this.yaw)];
    const eye = add(this.target, scale(offset, this.distance));
    const forward = scale(offset, -1);
    const right = normalize(cross(forward, [0, 1, 0]));
    return { eye, forward, right, up: cross(right, forward) };
  }

  /** `projection · view`, column-major. */
  viewProj(aspect: number): Float32Array {
    const { eye, forward: f, right: r, up: u } = this.basis();
    // prettier-ignore
    const view = [
      r[0], u[0], -f[0], 0,
      r[1], u[1], -f[1], 0,
      r[2], u[2], -f[2], 0,
      -dot(r, eye), -dot(u, eye), dot(f, eye), 1,
    ];
    const near = this.distance * 0.01;
    const far = this.distance * 100;
    const fy = 1 / Math.tan(this.fovY / 2);
    const proj = new Array<number>(16).fill(0);
    proj[0] = fy / aspect;
    proj[5] = fy;
    proj[10] = far / (near - far);
    proj[11] = -1;
    proj[14] = (near * far) / (near - far);
    const out = new Float32Array(16);
    for (let col = 0; col < 4; col++) {
      for (let row = 0; row < 4; row++) {
        let s = 0;
        for (let k = 0; k < 4; k++) {
          s += proj[k * 4 + row] * view[col * 4 + k];
        }
        out[col * 4 + row] = s;
      }
    }
    return out;
  }

  /** The world ray through pixel `(x, y)` of a `width × height` view (CSS pixels). */
  ray(x: number, y: number, width: number, height: number): { origin: Vec3; direction: Vec3 } {
    const { eye, forward, right, up } = this.basis();
    const t = Math.tan(this.fovY / 2);
    const nx = ((2 * x) / width - 1) * t * (width / height);
    const ny = (1 - (2 * y) / height) * t;
    return { origin: eye, direction: normalize(add(forward, add(scale(right, nx), scale(up, ny)))) };
  }
}
