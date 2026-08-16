/* ── TextLens — app.js (Advanced Scroll Animations & Active Nav Observer) ── */

/* Mark <html> so CSS reveal animations activate */
document.documentElement.classList.add('js');

/* ── Smooth Scroll & Reveal (IntersectionObserver with Scale & Blur) ── */
(function () {
  const targets = document.querySelectorAll('.reveal, .reveal-left, .reveal-right, .reveal-scale, .reveal-blur, .stagger');
  if (!targets.length) return;

  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add('in-view');
          io.unobserve(e.target); // fire once when scrolled into view
        }
      });
    },
    { threshold: 0.12, rootMargin: '0px 0px -40px 0px' }
  );

  targets.forEach((el) => io.observe(el));
})();

/* ── Active Header Nav Link Observer on Scroll ─────────────────── */
(function () {
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-links a[href^="#"]');
  if (!sections.length || !navLinks.length) return;

  const navObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const id = entry.target.getAttribute('id');
          navLinks.forEach((link) => {
            const href = link.getAttribute('href').substring(1);
            if (href === id) {
              link.style.color = 'var(--lime)';
              link.style.fontWeight = '700';
            } else {
              link.style.color = '';
              link.style.fontWeight = '';
            }
          });
        }
      });
    },
    { threshold: 0.3 }
  );

  sections.forEach((sec) => navObserver.observe(sec));
})();

/* ── Mobile Navigation Toggle ───────────────────────────────────── */
(function () {
  const btn = document.querySelector('.menu-button');
  const nav = document.querySelector('.nav-links');
  if (!btn || !nav) return;

  btn.addEventListener('click', () => {
    const open = nav.classList.toggle('is-open');
    btn.setAttribute('aria-expanded', String(open));
  });

  /* Close on link click */
  nav.querySelectorAll('a').forEach((a) =>
    a.addEventListener('click', () => {
      nav.classList.remove('is-open');
      btn.setAttribute('aria-expanded', 'false');
    })
  );
})();

/* ── Copy-to-clipboard buttons ──────────────────────────────────── */
(function () {
  document.querySelectorAll('.copy-button[data-copy]').forEach((btn) => {
    btn.addEventListener('click', () => {
      navigator.clipboard
        .writeText(btn.dataset.copy)
        .then(() => {
          const original = btn.textContent;
          btn.textContent = 'Copied!';
          setTimeout(() => (btn.textContent = original), 1600);
        })
        .catch(() => {});
    });
  });
})();

/* ── CLI showcase interactive tab switcher ──────────────────────── */
(function () {
  const cmds      = document.querySelectorAll('.cli-cmd');
  const tabs      = document.querySelectorAll('.cli-tab');
  const barLabel  = document.getElementById('cli-bar-label');
  if (!cmds.length) return;

  const labels = {
    models:   'textlens models',
    doctor:   'textlens doctor',
    discover: 'textlens discover',
    read:     'textlens read invoice.png',
    batch:    'textlens batch ./documents',
  };

  function activate(tabId) {
    /* Buttons */
    cmds.forEach((c) => c.classList.remove('active'));
    const activeCmd = document.querySelector(`.cli-cmd[data-tab="${tabId}"]`);
    if (activeCmd) activeCmd.classList.add('active');

    /* Output panels */
    tabs.forEach((t) => t.classList.remove('active'));
    const activeTab = document.getElementById(`tab-${tabId}`);
    if (activeTab) activeTab.classList.add('active');

    /* Bar label */
    if (barLabel) barLabel.textContent = labels[tabId] || tabId;
  }

  cmds.forEach((cmd) => {
    cmd.addEventListener('click', () => activate(cmd.dataset.tab));
  });
})();
