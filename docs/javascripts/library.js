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
  const subrows = [...library.querySelectorAll('[data-subfilters]')];
  const params = new URLSearchParams(location.search);
  let category = params.get('category') || 'all';
  if (!buttons.some(button => button.dataset.category === category)) category = 'all';
  let group = params.get('group') || 'all';
  const groupButtons = () => {
    const row = subrows.find(row => row.dataset.subfilters === category);
    return row ? [...row.querySelectorAll('[data-group]')] : [];
  };
  if (!groupButtons().some(button => button.dataset.group === group)) group = 'all';
  input.value = params.get('q') || '';
  function filter() {
    const terms = input.value.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
    let visible = 0;
    rows.forEach(row => {
      row.hidden = !(category === 'all' || row.dataset.section === category)
        || !(group === 'all' || row.dataset.group === group)
        || !terms.every(term => row.dataset.search.includes(term));
      if (!row.hidden) visible++;
    });
    buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.category === category)));
    subrows.forEach(row => { row.hidden = row.dataset.subfilters !== category; });
    groupButtons().forEach(button => button.setAttribute('aria-pressed', String(button.dataset.group === group)));
    count.textContent = `找到 ${visible} / ${rows.length} 篇文档`;
    empty.hidden = visible !== 0;
    const url = new URL(location.href);
    input.value.trim() ? url.searchParams.set('q', input.value.trim()) : url.searchParams.delete('q');
    category === 'all' ? url.searchParams.delete('category') : url.searchParams.set('category', category);
    group === 'all' ? url.searchParams.delete('group') : url.searchParams.set('group', group);
    history.replaceState(history.state, '', url);
  }
  input.addEventListener('input', filter);
  buttons.forEach(button => button.addEventListener('click', () => {
    if (button.dataset.category !== category) group = 'all';
    category = button.dataset.category;
    filter();
  }));
  subrows.forEach(row => row.querySelectorAll('[data-group]').forEach(button => button.addEventListener('click', () => {
    group = button.dataset.group;
    filter();
  })));
  filter();
}
if (typeof document$ !== 'undefined') document$.subscribe(initLibrary);
else document.addEventListener('DOMContentLoaded', initLibrary);
