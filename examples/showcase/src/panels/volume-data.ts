// Volume display values shared by the 3D and slice views.
/**
 * The scalars to show: real volumes pass through (Hounsfield units),
 * phantom SDF maps to pseudo-HU for the CT presets (air past 5 mm out).
 */
export function displayValues(values: Float32Array, pseudoHu: boolean): Float32Array {
  if (!pseudoHu) {
    return values;
  }
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    out[i] = v > 5 ? -1000 : 300 - 40 * v;
  }
  return out;
}
