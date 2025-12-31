window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon
window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function() {
// Move any non-.item children out of carousels to stop accidental absorption.
document.querySelectorAll('.carousel').forEach(function(el) {
try {
if (el.dataset.sanitized) return;
const children = Array.from(el.children);
const isItem = n => n.classList && n.classList.contains('item');
const itemIdxs = children.map((c, i) => isItem(c) ? i : -1).filter(i => i >= 0);
if (!itemIdxs.length) return; // nothing to sanitize
  const firstItem = Math.min(...itemIdxs);
  const lastItem = Math.max(...itemIdxs);

  const leading = children.slice(0, firstItem).filter(n => !isItem(n));
  const middle = children.slice(firstItem, lastItem + 1).filter(n => !isItem(n));
  const trailing = children.slice(lastItem + 1).filter(n => !isItem(n));

  // Move leading nodes back before the carousel (preserve order)
  leading.forEach(n => el.parentNode.insertBefore(n, el));

  // Move middle+trailing nodes after the carousel (preserve order)
  const fragAfter = document.createDocumentFragment();
  middle.forEach(n => fragAfter.appendChild(n));
  trailing.forEach(n => fragAfter.appendChild(n));
  el.parentNode.insertBefore(fragAfter, el.nextSibling);

  el.dataset.sanitized = 'true';
} catch (e) {
  console.warn('Carousel sanitize failed', e);
}
});

var options = {
slidesToScroll: 1,
slidesToShow: 1,
loop: true,
infinite: true,
autoplay: true,
autoplaySpeed: 5000
};

bulmaCarousel.attach('.carousel', options);
bulmaSlider.attach();
});

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 5000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();

})
