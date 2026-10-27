/* TextLens website — tabs, copy buttons, navigation, reveal, live badges, docs search. */
document.documentElement.classList.add("js");

/* Tabs: [data-tabs] containing [data-tab] buttons and [data-panel] panes */
document.querySelectorAll("[data-tabs]").forEach((group) => {
  const buttons = group.querySelectorAll("[data-tab]");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.setAttribute("aria-selected", String(b === btn)));
      group.querySelectorAll("[data-panel]").forEach((panel) => {
        panel.hidden = panel.dataset.panel !== btn.dataset.tab;
      });
    });
  });
});

/* Copy buttons: data-copy="#selector", or data-copy-active (visible panel's [data-code]) */
function copyText(text, btn) {
  const done = () => {
    const label = btn.querySelector("span");
    const old = label ? label.textContent : "";
    btn.classList.add("copied");
    if (label) label.textContent = "Copied";
    setTimeout(() => {
      btn.classList.remove("copied");
      if (label) label.textContent = old;
    }, 1600);
  };
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(done).catch(() => {});
  } else {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); done(); } catch (e) { /* ignore */ }
    ta.remove();
  }
}

function cleanCode(el) {
  // Drop prompt markers ("$ ") so the copied text runs as-is.
  const clone = el.cloneNode(true);
  clone.querySelectorAll(".p").forEach((p) => p.remove());
  return clone.textContent.replace(/\s+$/, "") + "\n";
}

document.querySelectorAll("[data-copy]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = document.querySelector(btn.dataset.copy);
    if (target) copyText(target.textContent.trim(), btn);
  });
});

document.querySelectorAll("[data-copy-active]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const group = btn.closest("[data-tabs]");
    const panel = [...group.querySelectorAll("[data-panel]")].find((p) => !p.hidden);
    const code = panel && (panel.querySelector("[data-code]") || panel.querySelector("pre"));
    if (code) copyText(cleanCode(code), btn);
  });
});

/* Docs code blocks: a copy button per .code block */
document.querySelectorAll(".doc .code").forEach((block) => {
  const btn = block.querySelector(".copy-btn");
  const pre = block.querySelector("pre");
  if (btn && pre) btn.addEventListener("click", () => copyText(cleanCode(pre), btn));
});

/* Mobile menu */
(() => {
  const toggle = document.querySelector(".menu-toggle");
  const nav = document.getElementById("nav");
  if (!toggle || !nav) return;
  toggle.addEventListener("click", () => {
    const open = nav.classList.toggle("open");
    toggle.setAttribute("aria-expanded", String(open));
  });
  nav.querySelectorAll("a").forEach((a) =>
    a.addEventListener("click", () => {
      nav.classList.remove("open");
      toggle.setAttribute("aria-expanded", "false");
    })
  );
})();

/* Reveal on scroll */
(() => {
  const items = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window)) {
    items.forEach((el) => el.classList.add("in"));
    return;
  }
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add("in");
          io.unobserve(e.target);
        }
      });
    },
    { threshold: 0.08, rootMargin: "0px 0px -40px 0px" }
  );
  items.forEach((el) => io.observe(el));
})();

/* Highlight the nav/sidebar link for the section in view */
(() => {
  const links = [...document.querySelectorAll('.nav a[href^="#"], .sidebar a[href^="#"]')];
  if (!links.length || !("IntersectionObserver" in window)) return;
  const byId = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        links.forEach((a) => a.classList.remove("active"));
        const link = byId.get(e.target.id);
        if (link) link.classList.add("active");
      });
    },
    { rootMargin: "-30% 0px -60% 0px" }
  );
  byId.forEach((_, id) => {
    const el = document.getElementById(id);
    if (el) io.observe(el);
  });
})();

/* Live badges: latest PyPI version and GitHub stars (silently skipped offline) */
(() => {
  const versionEls = document.querySelectorAll("[data-pypi-version]");
  if (versionEls.length) {
    fetch("https://pypi.org/pypi/textlens-ocr/json")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((d) => versionEls.forEach((el) => (el.textContent = "v" + d.info.version)))
      .catch(() => {});
  }
  const starEls = document.querySelectorAll("[data-gh-stars]");
  if (starEls.length) {
    fetch("https://api.github.com/repos/Srevarshan05/textlens")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((d) => {
        const n = d.stargazers_count;
        if (typeof n !== "number") return;
        const text = n >= 1000 ? (n / 1000).toFixed(1).replace(/\.0$/, "") + "k" : String(n);
        starEls.forEach((el) => (el.textContent = "★ " + text));
      })
      .catch(() => {});
  }
})();

/* Docs search: filter sections by text */
(() => {
  const input = document.querySelector(".search");
  if (!input) return;
  const sections = [...document.querySelectorAll(".doc section")];
  const empty = document.querySelector(".no-results");
  input.addEventListener("input", () => {
    const q = input.value.trim().toLowerCase();
    let shown = 0;
    sections.forEach((s) => {
      const hit = !q || s.textContent.toLowerCase().includes(q);
      s.hidden = !hit;
      if (hit) shown++;
    });
    document.querySelectorAll(".sidebar a[href^='#']").forEach((a) => {
      const s = document.getElementById(a.getAttribute("href").slice(1));
      a.style.display = s && s.hidden ? "none" : "";
    });
    if (empty) empty.hidden = shown > 0;
  });
})();
