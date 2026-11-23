// A camera-facing text label for the object overlays (#257): the angle and
// the bounding box dimensions. The canvas redraws only when the text
// changes, so a label that moves every frame costs no texture upload.
import * as THREE from "three";

export interface LabelSprite {
  sprite: THREE.Sprite;
  /** Shows `text` at `at`, `height` world units tall. */
  set: (text: string, at: readonly [number, number, number], height: number) => void;
  dispose: () => void;
}

/** A label in `color` (CSS) on a dark rounded plate, drawn over everything. */
export function makeLabelSprite(color: string): LabelSprite {
  const canvas = document.createElement("canvas");
  const texture = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false, transparent: true }));
  sprite.renderOrder = 999;
  sprite.raycast = () => {}; // A label is never a click target, also as a child of a mesh that is.
  let shown: string | null = null;
  return {
    sprite,
    set: (text, at, height) => {
      if (text !== shown) {
        const ctx = canvas.getContext("2d");
        if (ctx === null) {
          return;
        }
        shown = text;
        const font = "600 44px system-ui, sans-serif";
        ctx.font = font;
        canvas.width = Math.max(Math.ceil(ctx.measureText(text).width) + 36, 2);
        canvas.height = 72;
        ctx.font = font;
        ctx.textBaseline = "middle";
        ctx.fillStyle = "rgba(20, 22, 26, 0.85)";
        ctx.beginPath();
        ctx.roundRect(0, 0, canvas.width, canvas.height, 14);
        ctx.fill();
        ctx.fillStyle = color;
        ctx.fillText(text, 18, canvas.height / 2 + 2);
        texture.needsUpdate = true;
      }
      sprite.position.set(at[0], at[1], at[2]);
      sprite.scale.set((height * canvas.width) / canvas.height, height, 1);
    },
    // Frees the GPU side only: StrictMode runs cleanups once and keeps the objects, which three uploads again.
    dispose: () => {
      texture.dispose();
      (sprite.material as THREE.Material).dispose();
    },
  };
}
