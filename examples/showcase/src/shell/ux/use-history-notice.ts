// The status bar's notice that an action is permanent (#281): shown as it
// comes, until it is dismissed or has been there a while.
import { useEffect } from "react";
import { type AppState, useApp } from "../../app/state/app-state";
import { HISTORY_NOTICE_MS } from "../../app/ux/history-notice";

export function useHistoryNotice(app: AppState): { notice: string | null; dismiss: () => void } {
  const shown = useApp(app, (s) => s.historyNotice);
  useEffect(() => {
    if (shown === null) {
      return;
    }
    const timer = setTimeout(() => app.setHistoryNotice(null), HISTORY_NOTICE_MS);
    return () => clearTimeout(timer);
  }, [app, shown]);
  return { notice: shown?.text ?? null, dismiss: () => app.setHistoryNotice(null) };
}
