"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import Intro from "./intro";

function MoonIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
    </svg>
  );
}

function SunIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </svg>
  );
}

export default function Masthead() {
  const path = usePathname();
  const [dark, setDark] = useState(false);

  const intro = useRef(null);

  useEffect(() => {
    setDark(document.documentElement.dataset.theme === "dark");
    let seen = true;
    try {
      seen = localStorage.getItem("plexus-intro-seen") === "1";
    } catch {}
    if (!seen) intro.current?.showModal(); // first visit only
  }, []);

  function introClosed() {
    try {
      localStorage.setItem("plexus-intro-seen", "1");
    } catch {}
  }

  function toggle() {
    const next = !dark;
    setDark(next);
    if (next) document.documentElement.dataset.theme = "dark";
    else delete document.documentElement.dataset.theme;
    try {
      localStorage.setItem("plexus-theme", next ? "dark" : "light");
    } catch {
      // storage blocked (private window): the switch still works for this visit
    }
  }

  const here = (href) => (path === href || path === href + "/" ? "page" : undefined);

  return (
    <header className="masthead">
      <Link href="/" className="wordmark" aria-label="Plexus, home">
        Plexus<span className="wordmark-mark" aria-hidden="true">.</span>
      </Link>
      <nav className="nav" aria-label="Main">
        <Link href="/" aria-current={here("/")}>
          Ask
        </Link>
        <Link href="/about/" aria-current={here("/about")}>
          About
        </Link>
        <button type="button" className="nav-button" onClick={() => intro.current?.showModal()}>
          How to ask
        </button>
        <button type="button" className="theme-toggle" onClick={toggle} aria-pressed={dark}>
          {dark ? <SunIcon /> : <MoonIcon />}
          <span className="theme-label">Dark mode</span>
          <span className="visually-hidden">{dark ? "(on)" : "(off)"}</span>
        </button>
      </nav>
      <Intro ref={intro} onClose={introClosed} />
    </header>
  );
}
