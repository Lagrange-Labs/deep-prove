// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded "><a href="layers.html"><strong aria-hidden="true">1.</strong> Layers</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="maxpool.html"><strong aria-hidden="true">1.1.</strong> Maxpool</a></li></ol></li><li class="chapter-item expanded "><a href="lookups.html"><strong aria-hidden="true">2.</strong> Lookup Arguments</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="end_to_end_lu.html"><strong aria-hidden="true">2.1.</strong> End-to-End Lookup Protocol</a></li><li class="chapter-item expanded "><a href="relu.html"><strong aria-hidden="true">2.2.</strong> Relu</a></li><li class="chapter-item expanded "><a href="range_check.html"><strong aria-hidden="true">2.3.</strong> Range Checks</a></li></ol></li><li class="chapter-item expanded "><a href="commitments.html"><strong aria-hidden="true">3.</strong> Commitments</a></li><li class="chapter-item expanded "><a href="LLMs.html"><strong aria-hidden="true">4.</strong> LLMs</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="llms-layers/qkv.html"><strong aria-hidden="true">4.1.</strong> QKV Layer</a></li><li class="chapter-item expanded "><a href="llms-layers/positionals.html"><strong aria-hidden="true">4.2.</strong> Positional Encoding</a></li><li class="chapter-item expanded "><a href="llms-layers/embeddings.html"><strong aria-hidden="true">4.3.</strong> Embeddings</a></li><li class="chapter-item expanded "><a href="llms-layers/mha.html"><strong aria-hidden="true">4.4.</strong> MHA Layer</a></li><li class="chapter-item expanded "><a href="softmax.html"><strong aria-hidden="true">4.5.</strong> softmax</a></li><li class="chapter-item expanded "><a href="llms-layers/layernorm.html"><strong aria-hidden="true">4.6.</strong> LayerNorm</a></li></ol></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split("#")[0].split("?")[0];
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
