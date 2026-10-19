/* Filtering is local; no account, API or analytics is needed to browse. */
function initLibrary() {
  const library = document.querySelector('[data-library]');
  if (!library || library.dataset.ready) return;
  library.dataset.ready = 'true';
  const input = library.querySelector('[data-library-search]');
  const rows = [...library.querySelectorAll('[data-library-item]')];
  const count = library.querySelector('[data-library-count]');
  const empty = library.querySelector('[data-library-empty]');
  const buttons = [...library.querySelectorAll('[data-category]')];
  const params = new URLSearchParams(location.search);
  let category = params.get('category') || 'all';
  if (!buttons.some(button => button.dataset.category === category)) category = 'all';
  input.value = params.get('q') || '';
  function filter() {
    const terms = input.value.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
    let visible = 0;
    rows.forEach(row => {
      row.hidden = !(category === 'all' || row.dataset.section === category) || !terms.every(term => row.dataset.search.includes(term));
      if (!row.hidden) visible++;
    });
    buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.category === category)));
    count.textContent = `找到 ${visible} / ${rows.length} 篇文档`;
    empty.hidden = visible !== 0;
    const url = new URL(location.href);
    input.value.trim() ? url.searchParams.set('q', input.value.trim()) : url.searchParams.delete('q');
    category === 'all' ? url.searchParams.delete('category') : url.searchParams.set('category', category);
    history.replaceState(history.state, '', url);
  }
  input.addEventListener('input', filter);
  buttons.forEach(button => button.addEventListener('click', () => { category = button.dataset.category; filter(); }));
  filter();
}
if (typeof document$ !== 'undefined') document$.subscribe(initLibrary);
else document.addEventListener('DOMContentLoaded', initLibrary);
