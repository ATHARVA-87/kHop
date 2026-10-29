"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_BASE || "";

export default function About() {
  const [figures, setFigures] = useState(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    fetch(`${API}/api/about`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((d) => setFigures(d.figures))
      .catch(() => setFailed(true));
  }, []);

  return (
    <article className="about">
      <h1>How Plexus answers</h1>
      <p>
        Plexus reads the 2025 SEC filings of 39 semiconductor companies and answers questions
        about them in a few sentences. Each sentence points to the passage it came from, so you
        can check it in the filing.
      </p>

      <h2>What happens when you ask</h2>
      <ol>
        <li><strong>Check the question.</strong> Requests for investment advice, questions about other topics and attempts to change Plexus's rules are declined before anything is searched.</li>
        <li><strong>Choose how to search.</strong> A question about one company searches only that company's filings. A question naming several companies follows the links between them, such as a shared supplier. A sector-wide question takes one passage per company first, so no single company dominates.</li>
        <li><strong>Search the filings.</strong> Passages are found by both the words they use and their meaning, then the best ones are kept.</li>
        <li><strong>Write the answer.</strong> A language model running on this computer writes the answer from those passages only, citing one for every sentence. Text inside the filings is marked as data, so instructions hidden in a filing are ignored.</li>
        <li><strong>Check each sentence.</strong> A separate fact-checking model compares every sentence with the passage it cites. Sentences it cannot match are flagged in the answer.</li>
      </ol>

      <h2>How well it works</h2>
      <p>
        These figures come from tests run on the project's own question sets, many checked by
        hand. They are read from the saved results each time this page loads.
      </p>
      {figures ? (
        <table className="figures">
          <caption>Measured results</caption>
          <thead>
            <tr><th scope="col">What was measured</th><th scope="col">Result</th></tr>
          </thead>
          <tbody>
            {figures.map((f) => (
              <tr key={f.label}><td>{f.label}</td><td>{f.value}</td></tr>
            ))}
          </tbody>
        </table>
      ) : failed ? (
        <p className="notice">Plexus is not running, so the results cannot load. Start it with <code>make serve</code>, then reload this page.</p>
      ) : (
        <p>Loading the results…</p>
      )}

      <h2>What it does not do</h2>
      <p>
        Plexus does not give investment advice and does not know anything outside these filings,
        including news or stock prices. It can still be wrong: read the cited passage before
        relying on an answer.
      </p>
    </article>
  );
}
