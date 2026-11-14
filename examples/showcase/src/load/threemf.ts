// 3MF container importer: delegates archive extraction, Production Extension
// references, transforms and material colors to `fast-mesh-wasm/io`, then
// validates each resulting mesh into showcase `NamedMeshData` parts.
import { import3MF, type ImportedMesh } from "fast-mesh-wasm/io";
import type { NamedMeshData } from "./gltf";
import { checkMesh } from "./mesh-data";

/**
 * Parses a `.3mf` archive asynchronously into named showcase meshes.
 *
 * @throws When the ZIP/XML archive is malformed or contains no valid triangle meshes.
 */
export async function parse3mf(name: string, buffer: ArrayBuffer): Promise<NamedMeshData[]> {
  const parts = await import3MF(buffer);
  const valid = parts.filter((p) => p.indices.length > 0);
  if (valid.length === 0) {
    throw new Error(`3MF "${name}": the file contains no triangle meshes`);
  }
  return nameParts(name, valid);
}

function nameParts(name: string, parts: ImportedMesh[]): NamedMeshData[] {
  if (parts.length === 1) {
    const only = parts[0];
    return [
      {
        name,
        data: checkMesh({ positions: only.positions, indices: only.indices }, "3MF"),
      },
    ];
  }
  const stem = name.replace(/\.[^./\\]+$/, "") || name;
  return parts.map((part) => ({
    name: `${stem} · ${part.name}`,
    data: checkMesh({ positions: part.positions, indices: part.indices }, "3MF"),
  }));
}
