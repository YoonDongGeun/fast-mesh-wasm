// The help overlay's and the tour card's React side: whether help shows,
// the tour's step (null when it is not running) and its first-run start,
// and the ring on the anchor of the step that shows.
import { useCallback, useEffect, useRef, useState } from "react";
import { markTourSeen, resolveTourStart, type TourStep, tourBack, tourNext, tourSeen } from "./tour";

export function useHelpAndTour() {
  const [helpOpen, setHelpOpen] = useState(false);
  const [tourStep, setTourStep] = useState<number | null>(null);

  // Ending the tour records it, so first-run only runs once.
  const endTour = useCallback(() => {
    markTourSeen();
    setTourStep(null);
  }, []);
  const nextTour = useCallback(() => {
    if (tourStep === null) {
      return;
    }
    const next = tourNext(tourStep);
    if (next === null) {
      markTourSeen();
    }
    setTourStep(next);
  }, [tourStep]);
  const backTour = useCallback(() => {
    setTourStep((s) => (s === null ? s : tourBack(s)));
  }, []);

  // ---- First run: the tour, unless seen, forced off, or headless ----
  // `?tour=1` forces it on (even headless); `?tour=0` / `?notour` and
  // headless Chrome (`showcase-check.mjs`) keep it off, so the scripted
  // session never sees a card over the canvas.
  const onboarded = useRef(false);
  useEffect(() => {
    if (onboarded.current) {
      return;
    }
    onboarded.current = true;
    const start = resolveTourStart(location.search, navigator.userAgent, navigator.webdriver, tourSeen());
    if (start !== "off") {
      setTourStep(0);
    }
  }, []);

  const openHelp = () => setHelpOpen(true);
  const closeHelp = () => setHelpOpen(false);
  /** Help's "Take the tour": closes help and starts from the first step. */
  const startTourFromHelp = () => {
    setHelpOpen(false);
    setTourStep(0);
  };

  return { helpOpen, setHelpOpen, openHelp, closeHelp, tourStep, nextTour, backTour, endTour, startTourFromHelp };
}

/** Rings the step's `[data-tour]` anchor (`.tour-spot`) while it shows, scrolled into view. */
export function useTourSpot(step: TourStep): void {
  useEffect(() => {
    const el = document.querySelector(`[data-tour="${step.target}"]`);
    el?.classList.add("tour-spot");
    el?.scrollIntoView({ block: "nearest" });
    return () => el?.classList.remove("tour-spot");
  }, [step]);
}
