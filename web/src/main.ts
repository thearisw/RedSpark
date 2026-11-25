import { SimSocket } from "./net";
import { CircuitView } from "./render";
import { TrainMode } from "./train";
import type { ServerMessage } from "./types";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const select = $<HTMLSelectElement>("circuit");
const playBtn = $<HTMLButtonElement>("play");
const tps = $<HTMLInputElement>("tps");
const statusEl = $("status");
let playing = false;

const setStatus = (text: string, isError = false) => {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
};

const view = new CircuitView(document.body, (pos) => socket.send({ type: "toggle", pos }));

const socket = new SimSocket(
  (msg: ServerMessage) => {
    if (msg.type === "error") return setStatus(msg.message, true);
    if (train.handle(msg)) return;
    if (msg.type !== "state") return;
    playing = msg.playing;
    if (train.active) return; // the train tab owns the 3D view
    playBtn.textContent = playing ? "Pause" : "Play";
    if (msg.circuit) {
      view.show(msg.circuit);
      $("ticks").textContent = String(msg.circuit.ticks);
    }
  },
  setStatus,
  () => {
    train.disconnected();
    loadList();
  },
);
const train = new TrainMode(view, socket);

async function loadList(): Promise<void> {
    const names: string[] = await (await fetch("/api/circuits")).json();
    const previous = select.value;
    select.innerHTML = names.map((n) => `<option value="${n}">${n.replace(/_/g, " ")}</option>`).join("");
    const fromUrl = new URLSearchParams(location.search).get("circuit") ?? "";
    const preferred = [previous, fromUrl, "not_gate"].find((n) => names.includes(n));
    select.value = preferred ?? names[0];
    socket.send({ type: "load", name: select.value });
    if (playing) socket.send({ type: "play", on: true, tps: Number(tps.value) });
}

function setTab(trainTab: boolean): void {
  $("tabView").classList.toggle("on", !trainTab);
  $("tabTrain").classList.toggle("on", trainTab);
  $("viewMode").hidden = trainTab;
  $("trainMode").hidden = !trainTab;
  $("side").hidden = !trainTab;
  if (trainTab) {
    if (playing) socket.send({ type: "play", on: false });
    train.enter();
  } else {
    train.leave();
    loadList(); // picks up circuits saved from training
  }
}
$("tabView").addEventListener("click", () => setTab(false));
$("tabTrain").addEventListener("click", () => setTab(true));

select.addEventListener("change", () => socket.send({ type: "load", name: select.value }));
playBtn.addEventListener("click", () => socket.send({ type: "play", on: !playing, tps: Number(tps.value) }));
$("step").addEventListener("click", () => socket.send({ type: "step" }));
$("reset").addEventListener("click", () => socket.send({ type: "reset" }));
tps.addEventListener("input", () => {
  $("tpsLabel").textContent = tps.value;
  if (playing) socket.send({ type: "play", on: true, tps: Number(tps.value) });
});
