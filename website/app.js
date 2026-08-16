/* ── TextLens — app.js ──────────────────────────────────────────── */

/* Mark <html> so CSS reveal animations can activate */
document.documentElement.classList.add('js');

/* ── Scroll reveal (IntersectionObserver) ──────────────────────── */
(function () {
  const targets = document.querySelectorAll('.reveal, .reveal-left, .reveal-right, .stagger');
  if (!targets.length) return;

  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add('in-view');
          io.unobserve(e.target); // fire once
        }
      });
    },
    { threshold: 0.12, rootMargin: '0px 0px -48px 0px' }
  );

  targets.forEach((el) => io.observe(el));
})();

/* ── Mobile navigation toggle ───────────────────────────────────── */
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

  /* Auto-cycle every 4 seconds when section is visible */
  const tabOrder = ['models', 'doctor', 'discover', 'read', 'batch'];
  let idx = 0;
  let timer = null;

  function startCycle() {
    if (timer) return;
    timer = setInterval(() => {
      idx = (idx + 1) % tabOrder.length;
      activate(tabOrder[idx]);
    }, 4000);
  }

  function stopCycle() {
    clearInterval(timer);
    timer = null;
  }

  const cliSection = document.getElementById('cli-section');
  if (cliSection) {
    const sectionObs = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) startCycle();
        else stopCycle();
      },
      { threshold: 0.3 }
    );
    sectionObs.observe(cliSection);
  }

  /* Stop auto-cycle on manual click */
  cmds.forEach((cmd) => {
    cmd.addEventListener('click', () => {
      stopCycle();
      idx = tabOrder.indexOf(cmd.dataset.tab);
    });
  });
})();

/* ── FAQ smooth toggle ──────────────────────────────────────────── */
(function () {
  document.querySelectorAll('details').forEach((d) => {
    d.addEventListener('toggle', () => {
      if (d.open) {
        d.querySelectorAll('p').forEach((p) => {
          p.style.animation = 'fadeSlideIn .3s ease both';
        });
      }
    });
  });
})();
