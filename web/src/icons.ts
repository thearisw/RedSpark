/** Dark gray SVG icons for the Play/Pause and Reset buttons, with text kept as aria-label and tooltip. */

const svg = (body: string) =>
  `<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false">${body}</svg>`;

const ICONS = {
  play: svg(`<path d="M4 2.5v11l9-5.5z" fill="currentColor"/>`),
  pause: svg(`<rect x="3.5" y="2.5" width="3" height="11" rx="0.5" fill="currentColor"/><rect x="9.5" y="2.5" width="3" height="11" rx="0.5" fill="currentColor"/>`),
  reset: svg(
    `<path d="M3.5 8a4.5 4.5 0 1 0 1.4-3.3" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>` +
      `<path d="M2.5 1.8v3.9h3.9z" fill="currentColor"/>`,
  ),
};

function setIcon(btn: HTMLElement, icon: keyof typeof ICONS, label: string): void {
  btn.innerHTML = ICONS[icon];
  btn.classList.add("icon");
  btn.setAttribute("aria-label", label);
  btn.title = label;
}

/** Shows Play or Pause on a play/pause toggle button. */
export const setPlayIcon = (btn: HTMLElement, playing: boolean) =>
  setIcon(btn, playing ? "pause" : "play", playing ? "Pause" : "Play");

export const setResetIcon = (btn: HTMLElement) => setIcon(btn, "reset", "Reset");
