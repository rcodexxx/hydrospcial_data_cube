import L from 'leaflet';
import { state } from '../state.js';
import {
    API,
    INITIAL_CENTER,
    INITIAL_ZOOM,
    SEDIMENT_COLORS,
    SED_LABELS,
} from '../constants.js';

// Sediment legend control (lazily added when sediment layer is active)
const sedLegend = L.control({ position: 'bottomright' });
sedLegend.onAdd = function () {
    const div = L.DomUtil.create('div');
    div.style.cssText = 'background:white;padding:8px 12px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);font-size:11px;line-height:1.8;';
    div.innerHTML = '<div style="font-weight:bold;margin-bottom:4px;color:#334155;">Sediment Class</div>'
        + SEDIMENT_COLORS.map((c, i) =>
            `<div style="display:flex;align-items:center;gap:6px;">`
            + `<span style="display:inline-block;width:14px;height:14px;border-radius:3px;background:${c};border:1px solid rgba(0,0,0,0.1);"></span>`
            + `<span style="color:#475569;">${SED_LABELS[i]}</span>`
            + `</div>`
        ).join('');
    return div;
};
let sedLegendAdded = false;


function createGraticule(map, interval = 0.05) {
    const lines = [];
    const bounds = map.getBounds().pad(0.1);

    for (let lon = Math.floor(bounds.getWest() / interval) * interval;
         lon <= bounds.getEast(); lon += interval) {
        lines.push(L.polyline(
            [[bounds.getSouth(), lon], [bounds.getNorth(), lon]],
            { color: '#ffffff', weight: 0.5, opacity: 0.5, interactive: false }
        ));
    }
    for (let lat = Math.floor(bounds.getSouth() / interval) * interval;
         lat <= bounds.getNorth(); lat += interval) {
        lines.push(L.polyline(
            [[lat, bounds.getWest()], [lat, bounds.getEast()]],
            { color: '#ffffff', weight: 0.5, opacity: 0.5, interactive: false }
        ));
    }
    return L.layerGroup(lines);
}


function getContourStyle(layerId) {
    const baseStyles = {
        bathymetry:     { color: '#1e293b', opacity: 0.6 },
        imagery_hf:     { color: '#ffffff', opacity: 0.7 },
        sediment_class: { color: '#1e293b', opacity: 0.6 },
        mag_anomaly:    { color: '#1e293b', opacity: 0.5 },
    };
    const base = baseStyles[layerId] || baseStyles.bathymetry;

    return (feature) => {
        const isMajor = feature.properties?.level === 'major';
        return {
            color: base.color,
            weight: isMajor ? 1.5 : 0.6,
            opacity: isMajor ? base.opacity : base.opacity * 0.5,
            dashArray: '',
        };
    };
}


function loadStaticContours(map) {
    fetch(`${API}/api/contours`)
        .then(r => {
            if (!r.ok) throw new Error("Contour file not found");
            return r.json();
        })
        .then(geojson => {
            state.contourLayer = L.geoJSON(geojson, {
                style: getContourStyle(state.currentBaseLayerId),
                interactive: false
            });

            if (document.getElementById('chk-contours')?.checked) {
                state.contourLayer.addTo(map);
            }
        })
        .catch(err => console.log("等待執行 build_contours.py 產出靜態等高線...", err));
}


export function initMap() {
    const map = L.map('map', {
        center: INITIAL_CENTER,
        zoom: INITIAL_ZOOM,
        maxZoom: 20,
        minZoom: 15,
        renderer: L.canvas({ tolerance: 15 }),
        zoomControl: false,
    });

    L.control.zoom({ position: 'bottomright' }).addTo(map);
    L.control.scale({ position: 'bottomright', metric: true, imperial: false }).addTo(map);

    L.tileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        { attribution: 'Esri', maxZoom: 18, maxNativeZoom: 18 }
    ).addTo(map);

    // Graticule (refreshed on pan/zoom)
    let graticule = createGraticule(map, 0.05);
    graticule.addTo(map);
    map.on('moveend zoomend', () => {
        map.removeLayer(graticule);
        graticule = createGraticule(map, 0.05);
        graticule.addTo(map);
    });

    state.map = map;
    return map;
}


export function loadTileLayers(map) {
    fetch(`${API}/api/layers`).then(r => r.json()).then(data => {
        state.HAS_ISOPACH = data.features?.has_isopach ?? false;
        console.log('Feature flags:', data.features);

        if (data.bounds) {
            map.setMaxBounds(L.latLngBounds(data.bounds).pad(0.3));
            map.fitBounds(L.latLngBounds(data.bounds).pad(0.05));
        }
        for (const [key, cfg] of Object.entries(data.layers)) {
            state.tileLayers[key] = L.tileLayer(cfg.url, {
                opacity: 0.75, maxZoom: 20, maxNativeZoom: 18,
            });
        }
        if (state.tileLayers['bathymetry']) {
            state.tileLayers['bathymetry'].addTo(map);
            state.currentOverlay = state.tileLayers['bathymetry'];
        }
    });

    loadStaticContours(map);
}


export function bindMapUI(map) {
    // Contours toggle
    document.getElementById('chk-contours')?.addEventListener('change', (e) => {
        if (e.target.checked && state.contourLayer) state.contourLayer.addTo(map);
        else if (state.contourLayer) map.removeLayer(state.contourLayer);
    });

    // Base layer buttons
    document.querySelectorAll('.layer-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const targetBtn = e.currentTarget;
            const layerId = targetBtn.dataset.layer;
            state.currentBaseLayerId = layerId;

            document.querySelectorAll('.layer-btn').forEach(b => {
                b.classList.remove('active', 'bg-slate-800', 'text-white', 'font-bold', 'shadow-md');
                b.classList.add('text-slate-600', 'hover:bg-slate-100', 'font-medium');
                b.querySelector('.check-icon')?.classList.add('hidden');
            });
            targetBtn.classList.remove('text-slate-600', 'hover:bg-slate-100', 'font-medium');
            targetBtn.classList.add('active', 'bg-slate-800', 'text-white', 'font-bold', 'shadow-md');
            targetBtn.querySelector('.check-icon')?.classList.remove('hidden');

            // Imagery sub-control
            const subControl = document.getElementById('imagery-subcontrol');
            if (subControl) {
                if (targetBtn.dataset.layerGroup === 'imagery') {
                    subControl.classList.remove('hidden');
                    subControl.classList.add('flex');
                } else {
                    subControl.classList.add('hidden');
                    subControl.classList.remove('flex');
                }
            }

            // Switch tile layer
            if (!state.tileLayers[layerId]) return;
            if (state.currentOverlay) map.removeLayer(state.currentOverlay);
            state.tileLayers[layerId].setOpacity(document.getElementById('opacity-slider').value / 100);
            state.tileLayers[layerId].addTo(map);
            state.currentOverlay = state.tileLayers[layerId];

            // Contours: bathymetry default ON, others OFF
            const chkContours = document.getElementById('chk-contours');
            if (chkContours) {
                chkContours.checked = (layerId === 'bathymetry');
                if (state.contourLayer) {
                    state.contourLayer.setStyle(getContourStyle(state.currentBaseLayerId));
                    if (chkContours.checked) {
                        if (!map.hasLayer(state.contourLayer)) state.contourLayer.addTo(map);
                        state.contourLayer.bringToFront();
                    } else {
                        if (map.hasLayer(state.contourLayer)) map.removeLayer(state.contourLayer);
                    }
                }
            }

            // Sediment legend
            if (layerId === 'sediment_class') {
                if (!sedLegendAdded) { sedLegend.addTo(map); sedLegendAdded = true; }
            } else {
                if (sedLegendAdded) { sedLegend.remove(); sedLegendAdded = false; }
            }
        });
    });

    // SSS HF/LF sub-control
    document.querySelectorAll('.imagery-freq-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const freq = e.currentTarget.dataset.freq;
            const newLayerId = freq === 'hf' ? 'imagery_hf' : 'imagery_lf';

            document.querySelectorAll('.imagery-freq-btn').forEach(b => {
                if (b.dataset.freq === freq) {
                    b.classList.add('bg-blue-600', 'text-white');
                    b.classList.remove('bg-slate-200', 'text-slate-700');
                } else {
                    b.classList.remove('bg-blue-600', 'text-white');
                    b.classList.add('bg-slate-200', 'text-slate-700');
                }
            });

            if (state.currentOverlay) map.removeLayer(state.currentOverlay);
            if (state.tileLayers[newLayerId]) {
                state.tileLayers[newLayerId].setOpacity(document.getElementById('opacity-slider').value / 100);
                state.tileLayers[newLayerId].addTo(map);
                state.currentOverlay = state.tileLayers[newLayerId];
                state.currentBaseLayerId = newLayerId;
            }
        });
    });

    // Opacity slider
    document.getElementById('opacity-slider')?.addEventListener('input', (e) => {
        if (state.currentOverlay) state.currentOverlay.setOpacity(e.target.value / 100);
    });

    // On page load: if SSS Imagery is active, show sub-control
    document.addEventListener('DOMContentLoaded', () => {
        const activeImagery = document.querySelector('.layer-btn[data-layer-group="imagery"].active');
        if (activeImagery) {
            document.getElementById('imagery-subcontrol')?.classList.remove('hidden');
            document.getElementById('imagery-subcontrol')?.classList.add('flex');
        }
    });
}