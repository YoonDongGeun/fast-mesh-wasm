// The colormap legend of the inspect panel's colourings: the surface field's
// and the inspect maps'. A gradient bar with its two end labels, and for a
// map with a band of values in one colour, that colour's swatch and meaning.
import type { SurfaceColormap } from "../../../app/state/app-state";
import { COLORMAPS } from "../ux/use-surface-field";

export function FieldLegend(props: {
  map: SurfaceColormap;
  lo: string;
  hi: string;
  band?: { color: string; label: string } | null;
}) {
  const { map, lo, hi, band } = props;
  const meta = COLORMAPS.find((c) => c.id === map) ?? COLORMAPS[0];
  return (
    <div className="legend">
      <div className="legend-bar" style={{ background: meta.gradient }} />
      <div className="legend-labels">
        <span>{lo}</span>
        <span>{hi}</span>
      </div>
      {band != null ? (
        <div className="legend-band" data-legend-band>
          <span className="legend-swatch" style={{ background: band.color }} />
          {band.label}
        </div>
      ) : null}
    </div>
  );
}
