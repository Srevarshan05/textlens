/* ═══════════════════════════════════════════════════════════════════
   TextLens — app.js   (Interactive CLI Engine & UI Controls)
   ═══════════════════════════════════════════════════════════════════ */

/* Enable JS animations class */
document.documentElement.classList.add('js');

/* ── Scroll Reveal via IntersectionObserver ──────────────────────── */
(function () {
  const targets = document.querySelectorAll('.reveal, .reveal-left, .reveal-right, .stagger');
  if (!targets.length) return;

  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add('in-view');
          io.unobserve(e.target);
        }
      });
    },
    { threshold: 0.12, rootMargin: '0px 0px -40px 0px' }
  );

  targets.forEach((el) => io.observe(el));
})();

/* ── Mobile Navigation Toggle ────────────────────────────────────── */
(function () {
  const btn = document.querySelector('.menu-button');
  const nav = document.querySelector('.nav-links');
  if (!btn || !nav) return;

  btn.addEventListener('click', () => {
    const open = nav.classList.toggle('is-open');
    btn.setAttribute('aria-expanded', String(open));
  });

  nav.querySelectorAll('a').forEach((a) =>
    a.addEventListener('click', () => {
      nav.classList.remove('is-open');
      btn.setAttribute('aria-expanded', 'false');
    })
  );
})();

/* ── General Copy Buttons ────────────────────────────────────────── */
(function () {
  document.querySelectorAll('.copy-button[data-copy]').forEach((btn) => {
    btn.addEventListener('click', () => {
      navigator.clipboard
        .writeText(btn.dataset.copy)
        .then(() => {
          const original = btn.textContent;
          btn.textContent = 'Copied!';
          btn.style.background = '#70df7f';
          btn.style.color = '#060a06';
          setTimeout(() => {
            btn.textContent = original;
            btn.style.background = '';
            btn.style.color = '';
          }, 1600);
        })
        .catch(() => {});
    });
  });
})();

/* ── CLI Showcase Interactive Tab Engine ─────────────────────────── */
(function () {
  const cmds       = document.querySelectorAll('.cli-cmd');
  const tabs       = document.querySelectorAll('.cli-tab');
  const barLabel   = document.getElementById('cli-bar-label');
  const copyBtn    = document.getElementById('cli-copy-btn');
  const cliSection = document.getElementById('cli-section');
  if (!cmds.length || !tabs.length) return;

  const tabOrder = ['models', 'doctor', 'discover', 'read', 'batch'];
  let activeIndex = 0;
  let cycleTimer = null;
  let isUserInteracted = false;

  const commandMap = {
    models:   'textlens models',
    doctor:   'textlens doctor',
    discover: 'textlens discover GLM-OCR --compatible-only',
    read:     'textlens read invoice.png --model glm-ocr',
    batch:    'textlens batch ./documents --model glm-ocr --workers 1',
  };

  function activateTab(tabId, manual = false) {
    if (manual) isUserInteracted = true;

    // Update command buttons
    cmds.forEach((c) => {
      c.classList.remove('active');
      const prog = c.querySelector('.cli-cmd-progress');
      if (prog) {
        prog.style.transition = 'none';
        prog.style.width = '0%';
      }
    });

    const activeCmd = document.querySelector(`.cli-cmd[data-tab="${tabId}"]`);
    if (activeCmd) {
      activeCmd.classList.add('active');
      // Trigger smooth CSS progress bar animation
      const prog = activeCmd.querySelector('.cli-cmd-progress');
      if (prog && !isUserInteracted) {
        requestAnimationFrame(() => {
          prog.style.transition = 'width 4.5s linear';
          prog.style.width = '100%';
        });
      }
    }

    // Update output panels
    tabs.forEach((t) => t.classList.remove('active'));
    const activeTab = document.getElementById(`tab-${tabId}`);
    if (activeTab) activeTab.classList.add('active');

    // Update terminal top bar label
    const cmdStr = commandMap[tabId] || tabId;
    if (barLabel) barLabel.textContent = cmdStr;
    activeIndex = tabOrder.indexOf(tabId);
  }

  // Click listeners for each card
  cmds.forEach((cmd) => {
    cmd.addEventListener('click', () => {
      stopAutoCycle();
      activateTab(cmd.dataset.tab, true);
    });
  });

  // Terminal header copy button
  if (copyBtn) {
    copyBtn.addEventListener('click', () => {
      const currentTab = tabOrder[activeIndex];
      const cmdToCopy = commandMap[currentTab] || 'textlens';
      navigator.clipboard.writeText(cmdToCopy).then(() => {
        const textSpan = copyBtn.querySelector('span');
        if (textSpan) {
          const orig = textSpan.textContent;
          textSpan.textContent = 'Copied!';
          copyBtn.style.color = '#70df7f';
          setTimeout(() => {
            textSpan.textContent = orig;
            copyBtn.style.color = '';
          }, 1500);
        }
      });
    });
  }

  // Auto-cycle logic
  function nextTab() {
    if (isUserInteracted) return;
    activeIndex = (activeIndex + 1) % tabOrder.length;
    activateTab(tabOrder[activeIndex]);
  }

  function startAutoCycle() {
    if (cycleTimer || isUserInteracted) return;
    activateTab(tabOrder[activeIndex]);
    cycleTimer = setInterval(nextTab, 4800);
  }

  function stopAutoCycle() {
    if (cycleTimer) {
      clearInterval(cycleTimer);
      cycleTimer = null;
    }
  }

  if (cliSection) {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          startAutoCycle();
        } else {
          stopAutoCycle();
        }
      },
      { threshold: 0.3 }
    );
    observer.observe(cliSection);
  }
})();

/* ── FAQ Details Toggle Animation ────────────────────────────────── */
(function () {
  document.querySelectorAll('details').forEach((d) => {
    d.addEventListener('toggle', () => {
      if (d.open) {
        d.querySelectorAll('p').forEach((p) => {
          p.style.animation = 'smoothLineReveal .3s ease both';
        });
      }
    });
  });
})();
