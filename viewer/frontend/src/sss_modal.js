import OpenSeadragon from 'openseadragon';
/**
 * SSS Modal: full-screen OpenSeadragon viewer for detailed UCH inspection.
 *
 * Exposes window.SSSModal with:
 *   open(filename)  - open the modal for a given .jsf filename, default LF
 *   close()         - close the modal
 *   setFreq(freq)   - switch between 'HF' and 'LF' inside the modal
 *
 * Depends on:
 *   - waterfallIndex (loaded by app.js, available as window.waterfallIndex)
 *   - OpenSeadragon CDN (loaded in index.html before this script)
 */
(function () {
    let viewer = null;        // OpenSeadragon instance
    let currentFile = null;   // current .jsf filename
    let currentFreq = "LF";   // default LF (overview-friendly)

    function ensureModalElements() {
        // The modal DOM is defined in index.html; we just verify it exists.
        return document.getElementById("sss-modal");
    }

    function imageUrl(filename, freq) {
        const idx = window.waterfallIndex;
        if (!idx || !idx.sss) return null;
        const entry = idx.sss[`${filename}_${freq}`];
        return entry ? `/waterfalls/${entry.image}` : null;
    }

    function entryFor(filename, freq) {
        const idx = window.waterfallIndex;
        return idx?.sss?.[`${filename}_${freq}`] || null;
    }

    function updateStatusBar(filename, freq) {
        const entry = entryFor(filename, freq);
        const statusEl = document.getElementById("sss-modal-status");
        if (!statusEl) return;
        if (!entry) {
            statusEl.textContent = `${filename} — ${freq} (no data)`;
            return;
        }
        const parts = [
            `${filename}`,
            `${freq}`,
            `${entry.pings} pings`,
        ];
        if (entry.altitude_median_m != null) {
            parts.push(`alt: ${entry.altitude_min_m.toFixed(1)}–${entry.altitude_max_m.toFixed(1)} m`);
        }
        statusEl.textContent = parts.join("  |  ");
    }

    function destroyViewer() {
        if (viewer) {
            viewer.destroy();
            viewer = null;
        }
    }

    function loadImage(filename, freq) {
        const url = imageUrl(filename, freq);
        if (!url) {
            console.warn(`No SSS image for ${filename}_${freq}`);
            return;
        }

        destroyViewer();

        viewer = OpenSeadragon({
            id: "sss-modal-viewer",
            prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/",
            tileSources: {
                type: "image",
                url: url,
                buildPyramid: false,
            },
            showNavigator: true,
            navigatorPosition: "BOTTOM_RIGHT",
            navigatorHeight: "12%",
            navigatorWidth: "12%",
            showRotationControl: false,
            showFullPageControl: false,
            showHomeControl: true,
            showZoomControl: true,
            zoomInButton: "sss-modal-zoom-in",
            zoomOutButton: "sss-modal-zoom-out",
            homeButton: "sss-modal-home",
            visibilityRatio: 1.0,
            constrainDuringPan: true,
            minZoomLevel: 0.5,
            maxZoomPixelRatio: 4,
            animationTime: 0.3,
            blendTime: 0,
            gestureSettingsMouse: { clickToZoom: false },
        });

        updateStatusBar(filename, freq);
    }

    function setActiveFreqButton(freq) {
        ["hf", "lf"].forEach((f) => {
            const btn = document.getElementById(`sss-modal-freq-${f}`);
            if (!btn) return;
            if (f === freq.toLowerCase()) {
                btn.classList.add("bg-blue-600", "text-white");
                btn.classList.remove("bg-slate-200", "text-slate-700");
            } else {
                btn.classList.remove("bg-blue-600", "text-white");
                btn.classList.add("bg-slate-200", "text-slate-700");
            }
        });
    }

    function open(filename) {
        const modal = ensureModalElements();
        if (!modal) return;
        currentFile = filename;
        currentFreq = "LF";

        modal.classList.remove("hidden");
        modal.classList.add("flex");

        setActiveFreqButton(currentFreq);
        loadImage(currentFile, currentFreq);
    }

    function close() {
        const modal = ensureModalElements();
        if (!modal) return;
        destroyViewer();
        modal.classList.add("hidden");
        modal.classList.remove("flex");
        currentFile = null;
    }

    function setFreq(freq) {
        if (!currentFile) return;
        if (freq !== "HF" && freq !== "LF") return;
        currentFreq = freq;
        setActiveFreqButton(freq);
        loadImage(currentFile, freq);
    }

    // Wire up close + freq buttons + Esc key after DOM is ready
    function bindEvents() {
        document.getElementById("sss-modal-close")?.addEventListener("click", close);
        document.getElementById("sss-modal-freq-hf")?.addEventListener("click", () => setFreq("HF"));
        document.getElementById("sss-modal-freq-lf")?.addEventListener("click", () => setFreq("LF"));

        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && currentFile) close();
        });

        // Click outside viewer to close
        document.getElementById("sss-modal")?.addEventListener("click", (e) => {
            if (e.target.id === "sss-modal") close();
        });
    }

    // Bind on DOMContentLoaded if not already loaded, else immediately
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", bindEvents);
    } else {
        bindEvents();
    }

    window.SSSModal = { open, close, setFreq };
})();