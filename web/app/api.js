// One door to the kHop server. Two builds use it:
// - the app (`make ui`): every call goes to the FastAPI server on the same port;
// - the recorded demo (`make demo-site`, NEXT_PUBLIC_REPLAY=1): calls read real runs recorded
//   from the app (`make record-demo`, files in public/replay/), and answers stream back at the
//   speed they were recorded. Nothing is generated in the browser.

const API = process.env.NEXT_PUBLIC_API_BASE || "";
export const REPLAY = process.env.NEXT_PUBLIC_REPLAY === "1";
const BASE = process.env.NEXT_PUBLIC_BASE_PATH || "";
export const NOT_RECORDED =
  "This recorded demo can replay only its example questions, listed below. To ask your own, run kHop on your computer (see the README).";

let index = null;
async function replayIndex() {
  if (!index) {
    index = fetch(`${BASE}/replay/index.json`).then((r) => (r.ok ? r.json() : Promise.reject(new Error("no index"))));
  }
  return index;
}

/** The recorded examples ({kind, q}) and when they were recorded, or null in the app. */
export async function recordedExamples() {
  if (!REPLAY) return null;
  const i = await replayIndex();
  return { examples: i.examples, recorded: i.recorded };
}

const json = (body, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

const wait = (ms, signal) =>
  new Promise((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => { clearTimeout(t); reject(new DOMException("aborted", "AbortError")); });
  });

/** A recorded run -> a Server-Sent Events body that arrives at the recorded times. */
function replayStream(events, signal) {
  const encoder = new TextEncoder();
  return new ReadableStream({
    async start(controller) {
      let last = 0;
      try {
        for (const e of events) {
          await wait(Math.max(0, e.t - last), signal);
          last = e.t;
          controller.enqueue(encoder.encode(`event: ${e.event}\ndata: ${JSON.stringify(e.data)}\n\n`));
        }
        controller.close();
      } catch (err) {
        controller.error(err);
      }
    },
  });
}

async function replay(path, signal) {
  const url = new URL(path, "http://replay");
  const q = url.searchParams.get("q") || "";
  const i = await replayIndex();
  const file = (name) => fetch(`${BASE}/replay/${name}`, { signal }).then((r) => r.json());
  if (url.pathname === "/api/meta") return json(await file("meta.json"));
  if (url.pathname === "/api/about") return json(await file("about.json"));
  if (url.pathname === "/api/reframe") {
    const id = i.reframe[q.trim()];
    return json(id ? await file(`reframe/${id}.json`) : { original: q, question: q, changed: false, changes: [] });
  }
  if (url.pathname === "/api/ask") {
    const id = i.ask[q.trim()];
    if (!id) return json({ message: NOT_RECORDED }, 404);
    const events = await file(`ask/${id}.json`);
    return new Response(replayStream(events, signal), { headers: { "Content-Type": "text/event-stream" } });
  }
  return json({ message: "Not part of the recorded demo." }, 404);
}

export function apiFetch(path, options = {}) {
  return REPLAY ? replay(path, options.signal) : fetch(`${API}${path}`, options);
}
