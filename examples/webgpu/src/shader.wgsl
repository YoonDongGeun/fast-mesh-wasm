// Meshes: a headlight-lit shader with per-vertex RGBA8 colour (when the
// scene colours the mesh) or the object's base colour, per-object opacity,
// and an optional clip plane for the section view.
// Lines: section contours in one flat colour.

struct Frame {
  viewProj: mat4x4f,
  eye: vec4f,
  // Plane normal in xyz, w = -dot(normal, origin): points with
  // dot(normal, p) + w < 0 are cut away while flags.x is 1, so the side the
  // normal points to stays (as with bindSection in three.js).
  clip: vec4f,
  flags: vec4f,
};

struct Object {
  // scene.matrix4(mesh) as f32. Placements are rigid, so it also turns normals.
  model: mat4x4f,
  baseColor: vec4f,
  // x: opacity, y: 1 = use the vertex colour attribute, z: 1 = picked.
  params: vec4f,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<uniform> object: Object;

struct MeshIn {
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) color: vec4f, // unorm8x4
};

struct MeshOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) color: vec4f,
};

@vertex
fn vs_mesh(in: MeshIn) -> MeshOut {
  let world = object.model * vec4f(in.position, 1.0);
  var out: MeshOut;
  out.clip = frame.viewProj * world;
  out.world = world.xyz;
  out.normal = (object.model * vec4f(in.normal, 0.0)).xyz;
  out.color = select(object.baseColor, in.color, object.params.y > 0.5);
  return out;
}

@fragment
fn fs_mesh(in: MeshOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  if (frame.flags.x > 0.5 && dot(frame.clip.xyz, in.world) + frame.clip.w < 0.0) {
    discard;
  }
  let toEye = normalize(frame.eye.xyz - in.world);
  var n = in.normal;
  if (dot(n, n) < 1e-12) {
    n = toEye; // a vertex of degenerate triangles only
  }
  n = normalize(n);
  if (!front) {
    n = -n; // two-sided: back faces show through cuts and transparency
  }
  let diffuse = max(dot(n, toEye), 0.0);
  let specular = pow(diffuse, 48.0) * 0.25;
  var rgb = in.color.rgb * (0.3 + 0.7 * diffuse) + vec3f(specular);
  if (object.params.z > 0.5) {
    rgb = mix(rgb, vec3f(1.0, 0.6, 0.2), 0.25);
  }
  return vec4f(rgb, in.color.a * object.params.x);
}

@vertex
fn vs_line(@location(0) position: vec3f) -> @builtin(position) vec4f {
  return frame.viewProj * vec4f(position, 1.0);
}

@fragment
fn fs_line() -> @location(0) vec4f {
  return vec4f(1.0, 0.85, 0.1, 1.0);
}
