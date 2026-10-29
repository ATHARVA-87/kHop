"use client";

import { useEffect, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_BASE || "";

const EXAMPLES = [
  { kind: "Two companies", q: "What do Impinj and Allegro MicroSystems each rely on TSMC for?" },
  { kind: "Two companies", q: "How do Lam Research and KLA describe their exposure to Huawei under export restrictions?" },
  { kind: "One company", q: "What natural disaster risks does NXP face at its factories in places like Singapore, Taiwan, Malaysia and Thailand?" },
  { kind: "Whole sector", q: "How do all semiconductor companies describe the risks that tariffs and trade policy pose to their costs and supply chains?" },
];

const STEP_ORDER = ["guard", "route", "retrieve", "answer", "verify"];
const HISTORY_KEY = "plexus-history"; // this browser tab only (sessionStorage), never sent anywhere
const HISTORY_MAX = 20;

function loadHistory() {
  try {
    const h = JSON.parse(sessionStorage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(h) ? h : [];
  } catch {
    return [];
  }
}

function saveHistory(h) {
  try {
    sessionStorage.setItem(HISTORY_KEY, JSON.stringify(h.slice(0, HISTORY_MAX)));
  } catch {
    // storage full or blocked: history just is not kept
  }
}

function stepsOf(data) {
  const steps = Object.fromEntries((data.trace || []).map((t) => [t.step, t]));
  if (data.from_cache) steps.cache = { step: "cache" };
  return steps;
}
const NOT_ENOUGH = "Not enough evidence in the filings to answer this.";
const REASONS = {
  advice: "It asks for investment advice.",
  scope: "It is not about the filings Plexus covers.",
  jailbreak: "It asks Plexus to ignore its rules.",
  too_long: "It is longer than 1000 characters.",
};
const ROUTES = {
  local: "One company: searched only its filings",
  multi_hop: "Several companies: followed the links between them",
  global: "Whole sector: one passage per company first",
  fixed: "Searched all filings",
};

function WarnIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M12 3 2 21h20L12 3z" />
      <path d="M12 10v5M12 18h.01" />
    </svg>
  );
}

function ExternalIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M14 4h6v6M20 4l-9 9M18 14v6H4V6h6" />
    </svg>
  );
}

// ---- reading the event stream (fetch, so error messages from the server can be shown) ----
async function streamQuestion(question, signal, on) {
  const res = await fetch(`${API}/api/ask?q=${encodeURIComponent(question)}`, { signal });
  if (!res.ok) {
    let message = "Plexus could not take this question. Try again.";
    try {
      const body = await res.json();
      if (typeof body.message === "string") message = body.message;
      else if (res.status === 422) message = "Type a question between 3 and 1000 characters.";
    } catch {}
    throw new Error(message);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      let event = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) {
        let parsed;
        try { parsed = JSON.parse(data); } catch { throw new Error("Plexus sent a reply it could not read. Ask again."); }
        on(event, parsed);
      }
    }
  }
}

// ---- answer text: plain text only (never HTML), citations become buttons ----
const norm = (s) => s.replace(/\[\d+\]/g, "").replace(/\*\*/g, "").replace(/\s+/g, " ").trim().toLowerCase();

function blocksOf(text) {
  const out = [];
  for (const raw of text.split("\n")) {
    const line = raw.replace(/\*\*/g, "").replace(/^#+\s*/, "").trim();
    if (!line) continue;
    const bullet = /^([-*•]|\d+\.)\s+/.test(line);
    out.push({ bullet, text: line.replace(/^([-*•]|\d+\.)\s+/, "") });
  }
  return out;
}

function Inline({ text, nSources, current, onCite }) {
  return text.split(/(\[\d+\])/g).map((part, i) => {
    const m = part.match(/^\[(\d+)\]$/);
    if (!m) return <span key={i}>{part}</span>;
    const n = Number(m[1]);
    if (!nSources || n < 1 || n > nSources) return <span key={i} className="cite" aria-hidden="true">{n}</span>;
    return (
      <button key={i} type="button" className="cite" aria-current={current === n ? "true" : undefined}
        onClick={() => onCite(n)} aria-label={`Source ${n}`}>
        {n}
      </button>
    );
  });
}

function Answer({ text, busy, result, current, onCite }) {
  const blocks = blocksOf(text);
  const unsupported = (result?.checks || []).filter((c) => !c.supported);
  const nSources = result?.sources?.length || 0;
  const items = [];
  let list = [];
  const flush = () => {
    if (list.length) items.push(<ul key={`ul${items.length}`}>{list}</ul>);
    list = [];
  };
  blocks.forEach((b, i) => {
    const flagged = unsupported.some((c) => norm(b.text).includes(norm(c.sentence).slice(0, 60)));
    const body = (
      <>
        <Inline text={b.text} nSources={nSources} current={current} onCite={onCite} />
        {flagged && (
          <span className="flag">
            <WarnIcon />
            Not backed by its source. Check the cited passage before relying on this.
          </span>
        )}
      </>
    );
    if (b.bullet) list.push(<li key={i} className="line">{body}</li>);
    else {
      flush();
      items.push(<p key={i} className="line">{body}</p>);
    }
  });
  flush();
  return (
    <div className="answer" aria-busy={busy ? "true" : "false"}>
      {items.length ? items : <p className="status">Searching the filings…</p>}
    </div>
  );
}

function formatDate(iso) {
  if (!iso) return "";
  const d = new Date(`${iso}T00:00:00`);
  return d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

function Source({ s, cited, current }) {
  return (
    <article id={`source-${s.n}`} tabIndex={-1}
      className={`source${cited ? " is-cited" : ""}${current === s.n ? " is-current" : ""}`}>
      <div className="source-head">
        <span className="source-n">[{s.n}]</span>
        <span className="source-title">{s.company}, {s.form}</span>
        <span className="source-meta">filed {formatDate(s.filed)}, {s.section}</span>
      </div>
      <p className="source-text">{s.text}</p>
      {s.url && (
        <a className="source-link" href={s.url} target="_blank" rel="noopener noreferrer">
          Open the filing on SEC.gov <ExternalIcon />
          <span className="visually-hidden">(opens in a new tab)</span>
        </a>
      )}
    </article>
  );
}

function stepView(name, step, running, result) {
  if (!step) {
    const waiting = {
      guard: "Check the question", route: "Choose how to search", retrieve: "Search the filings",
      answer: "Write the answer", verify: "Check each sentence",
    };
    return { title: waiting[name], detail: running ? "Working on it…" : "Waiting", state: running ? "is-running" : "" };
  }
  const sec = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.max(1, Math.round(ms))} ms`);
  switch (name) {
    case "cache":
      return {
        title: "Reused a recent answer",
        detail: "A very similar question about the same companies and dates was answered recently and fully fact-checked, so its answer is shown at once.",
        state: "is-done",
      };
    case "guard":
      return step.refused
        ? { title: "Declined the question", detail: REASONS[step.reason] || "Outside what Plexus answers.", state: "is-flagged" }
        : { title: "Checked the question", detail: "It is about the filings Plexus covers.", state: "is-done" };
    case "route":
      return { title: "Chose how to search", detail: ROUTES[step.route] || step.route, state: "is-done" };
    case "retrieve":
      return { title: `Found ${step.n_hits} passages`, detail: `The best ${Math.min(5, step.n_hits)} went to the writer, in ${sec(step.ms)}.`, state: "is-done" };
    case "answer":
      return {
        title: step.cached ? "Reused a saved answer" : "Wrote the answer",
        detail: step.cached
          ? `This question was answered before, citing ${new Set(step.cited || []).size} passages.`
          : `${sec(step.ms)}, citing ${new Set(step.cited || []).size} passages.`,
        state: "is-done",
      };
    case "verify": {
      if (!step.sentences) return { title: "Checked each sentence", detail: "Nothing to check: the answer makes no claims.", state: "is-done" };
      const bad = step.sentences - step.supported;
      return {
        title: "Checked each sentence",
        detail: bad ? `${step.supported} of ${step.sentences} backed by their source; ${bad} flagged in the answer.` : `All ${step.sentences} backed by their source.`,
        state: bad ? "is-flagged" : "is-done",
      };
    }
    default:
      return { title: name, detail: "", state: "is-done" };
  }
}

function Trace({ steps, phase, queued, result }) {
  const refused = steps.guard?.refused;
  const names = steps.cache ? ["cache"] : refused ? ["guard"] : STEP_ORDER;
  const nextName = names.find((n) => !steps[n]);
  return (
    <aside className="trace" aria-labelledby="trace-title">
      <h2 id="trace-title">How this answer was made</h2>
      {queued > 0 && phase === "asking" && !steps.guard && (
        <p className="status">Waiting for {queued === 1 ? "1 question" : `${queued} questions`} ahead to finish.</p>
      )}
      <ol className="steps">
        {names.map((n) => {
          const v = stepView(n, steps[n], phase === "asking" && n === nextName, result);
          return (
            <li key={n} className={`step ${v.state}${steps[n] ? " step-new" : ""}`}>
              <span className="step-title">{v.title}</span>
              <span className="step-detail">{v.detail}</span>
            </li>
          );
        })}
      </ol>
    </aside>
  );
}

export default function AskPage() {
  const [question, setQuestion] = useState("");
  const [asked, setAsked] = useState("");
  const [fieldError, setFieldError] = useState("");
  const [phase, setPhase] = useState("start"); // start | asking | done | error
  const [text, setText] = useState("");
  const [steps, setSteps] = useState({});
  const [queued, setQueued] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [current, setCurrent] = useState(null);
  const [meta, setMeta] = useState(null);
  const [history, setHistory] = useState([]);
  const [metaError, setMetaError] = useState(false);
  const abort = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    setHistory(loadHistory());
    fetch(`${API}/api/meta`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setMeta)
      .catch(() => setMetaError(true));
    return () => abort.current?.abort();
  }, []);

  async function ask(q) {
    const clean = q.trim();
    if (clean.length < 3) {
      setFieldError("Type at least 3 characters.");
      inputRef.current?.focus();
      return;
    }
    setFieldError("");
    abort.current?.abort(); // a new question replaces the old one (the server frees its slot)
    const ctrl = new AbortController();
    abort.current = ctrl;
    setAsked(clean);
    setPhase("asking");
    setText("");
    setSteps({});
    setQueued(0);
    setResult(null);
    setError("");
    setCurrent(null);
    let finished = false;
    try {
      await streamQuestion(clean, ctrl.signal, (event, data) => {
        if (event === "queued") setQueued(data.position);
        else if (event === "step") setSteps((s) => ({ ...s, [data.step]: data }));
        else if (event === "token") setText((t) => t + data.text);
        else if (event === "done") {
          finished = true;
          setText(data.answer);
          setResult(data);
          setPhase("done");
          setHistory((h) => {
            const next = [{ q: clean, at: Date.now(), data }, ...h.filter((e) => e.q !== clean)].slice(0, HISTORY_MAX);
            saveHistory(next);
            return next;
          });
        } else if (event === "error") {
          finished = true;
          setError(data.message);
          setPhase("error");
        }
      });
      if (!finished && !ctrl.signal.aborted) {
        setError("The connection to Plexus closed before the answer finished. Ask again.");
        setPhase("error");
      }
    } catch (e) {
      if (ctrl.signal.aborted) return;
      setError(e.message === "Failed to fetch" || e.name === "TypeError"
        ? "Plexus is not reachable. Start it with make serve, then ask again."
        : e.message);
      setPhase("error");
    }
  }

  function reopen(entry) {
    abort.current?.abort();
    setQuestion(entry.q);
    setAsked(entry.q);
    setText(entry.data.answer);
    setResult(entry.data);
    setSteps(stepsOf(entry.data));
    setQueued(0);
    setError("");
    setCurrent(null);
    setPhase("done");
  }

  function clearHistory() {
    setHistory([]);
    saveHistory([]);
  }

  function onCite(n) {
    setCurrent(n);
    const el = document.getElementById(`source-${n}`);
    if (el) {
      const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      el.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
      el.focus({ preventScroll: true });
    }
  }

  const refused = steps.guard?.refused;
  const noEvidence = phase === "done" && !refused && text.trim().startsWith(NOT_ENOUGH);
  const citedNums = new Set([...(result?.answer || "").matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])));
  const sources = result?.sources || [];
  const cited = sources.filter((s) => citedNums.has(s.n));
  const others = sources.filter((s) => !citedNums.has(s.n));
  const statusText = {
    asking: steps.answer ? "Checking each sentence…" : text ? "Writing the answer…" : "Searching the filings…",
    done: "Answer ready",
    error: "Stopped",
  }[phase];

  return (
    <>
      {phase === "start" && (
        <section className="intro" aria-labelledby="intro-title">
          <h1 id="intro-title">Ask what chip companies told the SEC</h1>
          <p>Short answers from the 2025 filings of 39 semiconductor companies. Every sentence cites the passage it came from.</p>
        </section>
      )}

      <form className="ask" onSubmit={(e) => { e.preventDefault(); ask(question); }} noValidate>
        <label htmlFor="question">Your question</label>
        <div className="ask-row">
          <input id="question" ref={inputRef} className="ask-input" type="text" enterKeyHint="search"
            autoComplete="off" maxLength={1000} value={question}
            onChange={(e) => setQuestion(e.target.value)}
            aria-describedby={fieldError ? "question-error question-help" : "question-help"}
            aria-invalid={fieldError ? "true" : undefined}
            placeholder="Which companies rely on TSMC?" />
          <button className="ask-button" type="submit" disabled={phase === "asking"}>
            {phase === "asking" ? "Asking…" : "Ask"}
          </button>
        </div>
        {fieldError ? (
          <p id="question-error" className="field-error" role="alert"><WarnIcon />{fieldError}</p>
        ) : null}
        <p id="question-help" className="helper">
          Name one or more companies, or ask about the whole sector. Plexus does not give investment advice.
        </p>
      </form>

      {history.length > 0 && (
        <details className="history" open={phase === "start"}>
          <summary>Your questions this session ({history.length})</summary>
          <ul className="history-list">
            {history.map((e) => (
              <li key={e.at}>
                <button type="button" className="history-item" onClick={() => reopen(e)}
                  aria-current={phase !== "start" && e.q === asked ? "true" : undefined}>
                  {e.q}
                </button>
              </li>
            ))}
          </ul>
          <p className="history-note">
            Kept only in this browser tab; closing the tab clears it.{" "}
            <button type="button" className="text-button" onClick={clearHistory}>Clear these questions</button>
          </p>
        </details>
      )}

      <p className="visually-hidden" aria-live="polite">{phase !== "start" ? statusText : ""}</p>

      {phase === "start" && (
        <div className="start">
          <section aria-labelledby="examples-title">
            <h2 id="examples-title">Try one of these</h2>
            <ul className="examples">
              {EXAMPLES.map((e) => (
                <li key={e.q}>
                  <button type="button" className="example" onClick={() => { setQuestion(e.q); ask(e.q); }}>
                    <span className="example-kind">{e.kind}</span>
                    {e.q}
                  </button>
                </li>
              ))}
            </ul>
          </section>
          <section className="coverage" aria-labelledby="coverage-title">
            <h2 id="coverage-title">What Plexus covers</h2>
            {meta ? (
              <p>
                <strong>{meta.filings} filings</strong> (annual 10-K and quarterly 10-Q reports filed in {meta.years}) from{" "}
                <strong>{meta.companies.length} companies</strong>: {meta.companies.join(", ")}.
              </p>
            ) : metaError ? (
              <p className="notice">Plexus is not running, so the list of companies cannot load. Start it with <code>make serve</code>, then reload this page.</p>
            ) : (
              <p>Loading the list of companies…</p>
            )}
          </section>
        </div>
      )}

      {phase !== "start" && (
        <div className="result">
          <section aria-labelledby="answer-title">
            <div className="answer-head">
              <h2 id="answer-title">{asked}</h2>
              <span className="status">{statusText}</span>
            </div>

            {phase === "error" ? (
              <div className="notice" role="alert">
                <p>{error}</p>
                <button type="button" className="retry" onClick={() => ask(asked)}>Ask again</button>
              </div>
            ) : refused ? (
              <div className="notice">
                <p>{text}</p>
                <p>Try one of these instead:</p>
                <ul className="examples">
                  {EXAMPLES.slice(0, 2).map((e) => (
                    <li key={e.q}><button type="button" className="example" onClick={() => { setQuestion(e.q); ask(e.q); }}>{e.q}</button></li>
                  ))}
                </ul>
              </div>
            ) : noEvidence ? (
              <div className="notice">
                <p>The filings Plexus found do not answer this question.</p>
                <p>Name the company, or ask about something filings cover, such as risks, suppliers, customers or results. The passages it searched are listed below.</p>
              </div>
            ) : (
              <Answer text={text} busy={phase === "asking"} result={result} current={current} onCite={onCite} />
            )}

            {phase === "done" && !refused && sources.length > 0 && (
              <section className="sources" aria-labelledby="sources-title">
                <h2 id="sources-title">{cited.length ? "Sources" : "Passages searched"}</h2>
                {(cited.length ? cited : others).map((s) => <Source key={s.n} s={s} cited={cited.length > 0} current={current} />)}
                {cited.length > 0 && others.length > 0 && (
                  <details className="also">
                    <summary>Also searched, not cited ({others.length})</summary>
                    {others.map((s) => <Source key={s.n} s={s} cited={false} current={current} />)}
                  </details>
                )}
              </section>
            )}
          </section>

          <Trace steps={steps} phase={phase} queued={queued} result={result} />
        </div>
      )}
    </>
  );
}
