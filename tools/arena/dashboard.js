const tabs = Array.from(document.querySelectorAll('[role="tab"]'));

function showStage(nextTab) {
  for (const tab of tabs) {
    const selected = tab === nextTab;
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
    document.getElementById(tab.getAttribute('aria-controls')).hidden = !selected;
  }
}

for (const tab of tabs) {
  tab.addEventListener('click', () => showStage(tab));
  tab.addEventListener('keydown', (event) => {
    const index = tabs.indexOf(tab);
    let next;
    switch (event.key) {
      case 'ArrowLeft':
        next = tabs[(index + tabs.length - 1) % tabs.length];
        break;
      case 'ArrowRight':
        next = tabs[(index + 1) % tabs.length];
        break;
      case 'Home':
        next = tabs[0];
        break;
      case 'End':
        next = tabs[tabs.length - 1];
        break;
      default:
        return;
    }
    event.preventDefault();
    showStage(next);
    next.focus();
  });
}

for (const card of document.querySelectorAll('[data-stage-tab]')) {
  const tab = document.getElementById(card.dataset.stageTab);
  card.addEventListener('click', () => {
    showStage(tab);
    tab.focus();
  });
}

document.getElementById('print-report').addEventListener('click', () => window.print());
