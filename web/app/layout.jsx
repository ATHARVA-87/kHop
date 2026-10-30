import { Literata, Public_Sans } from "next/font/google";
import "./globals.css";
import Masthead from "./masthead";

// Fonts are downloaded at build time and served by kHop itself: no request to Google
// from the reader's browser.
const read = Literata({ subsets: ["latin"], variable: "--font-read", display: "swap" });
const ui = Public_Sans({ subsets: ["latin"], variable: "--font-ui", display: "swap" });

export const metadata = {
  title: "kHop: answers from chip companies' SEC filings",
  description:
    "Ask a question about the 2025 SEC filings of 39 semiconductor companies and get a short answer with its sources.",
};

export const viewport = { width: "device-width", initialScale: 1 };

// Light is the default. Dark mode only when the reader chose it before (stored in this
// browser). Runs before the page paints, so there is no flash of the wrong theme.
const themeScript = `try{if(localStorage.getItem("plexus-theme")==="dark")document.documentElement.dataset.theme="dark"}catch(e){}`;

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`${read.variable} ${ui.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>
        <a className="skip-link" href="#main">
          Skip to the question box
        </a>
        <div className="frame">
          <Masthead />
          <main id="main">{children}</main>
          <footer className="footer">
            <p>
              kHop answers only from public SEC filings (10-K and 10-Q, filed in 2025). It does
              not give investment advice. Check important facts in the filing itself: every source
              links to it on SEC.gov.
            </p>
          </footer>
        </div>
      </body>
    </html>
  );
}
