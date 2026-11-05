// A typed length: holds millimetres, shows and takes the display unit.
import type { InputHTMLAttributes } from "react";
import { fromDisplayLength, lengthInputValue, type LengthUnit } from "./units";

export function LengthInput(
  props: { mm: number; unit: LengthUnit; onMm: (mm: number) => void } & Omit<
    InputHTMLAttributes<HTMLInputElement>,
    "type" | "value" | "onChange"
  >,
) {
  const { mm, unit, onMm, ...rest } = props;
  return (
    <input
      type="number"
      step="any"
      {...rest}
      value={lengthInputValue(mm, unit)}
      onChange={(e) => onMm(fromDisplayLength(Number(e.currentTarget.value) || 0, unit))}
    />
  );
}
