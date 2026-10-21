/* Scroll-triggered reveals. Content stays visible without JS. */
function initCinema() {
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const targets = [...document.querySelectorAll('[data-reveal]:not(.is-visible)')];
  if (still || !('IntersectionObserver' in window)) {
    targets.forEach(el => el.classList.add('is-visible'));
  } else {
    const observer = new IntersectionObserver(entries => entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add('is-visible');
      observer.unobserve(entry.target);
    }), { rootMargin: '0px 0px -8% 0px', threshold: 0.08 });
    targets.forEach(el => observer.observe(el));
  }
}
if (typeof document$ !== 'undefined') document$.subscribe(initCinema);
else document.addEventListener('DOMContentLoaded', initCinema);
