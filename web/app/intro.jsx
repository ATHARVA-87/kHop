"use client";

import { forwardRef } from "react";

// What kHop is and how to ask well. Shown once on the first visit (remembered in this
// browser), reopened from "How to ask" in the header. A native <dialog>: keyboard focus stays
// inside, Esc closes it, screen readers announce it.
const Intro = forwardRef(function Intro({ onClose }, ref) {
  return (
    <dialog ref={ref} className="intro-dialog" aria-labelledby="intro-dialog-title" onClose={onClose}>
      <h2 id="intro-dialog-title">How to ask kHop</h2>
      <p>
        kHop answers questions from the 2025 SEC filings (annual 10-K and quarterly 10-Q
        reports) of 39 semiconductor companies. Every sentence of an answer cites the passage it
        came from, so you can check it.
      </p>
      <h3>Questions that work well</h3>
      <ul>
        <li><strong>One company:</strong> "What does Micron say about its exposure to China?"</li>
        <li><strong>Two companies:</strong> "How do Lam Research and KLA each describe export controls?"</li>
        <li><strong>The whole sector:</strong> "How do chip companies describe customer concentration?"</li>
      </ul>
      <h3>What kHop will not do</h3>
      <ul>
        <li>Give investment advice, such as whether to buy or sell a stock.</li>
        <li>Answer about news, prices or companies outside its 2025 filings.</li>
      </ul>
      <p>
        Your questions this session are kept only in this browser tab, so you can reopen an
        answer. The text of your questions is not stored on the server.
      </p>
      <form method="dialog">
        <button type="submit" className="ask-button" autoFocus>Start asking</button>
      </form>
    </dialog>
  );
});

export default Intro;
