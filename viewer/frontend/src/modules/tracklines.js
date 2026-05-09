import { state } from '../state.js';
import { API } from '../constants.js';

let magTargetsLayer = null;


function selectTrackline(feature, layer, parentLayer) {
    if (state.selectedTrackline && state.selectedParentLayer) {
        state.selectedParentLayer.resetStyle(state.selectedTrackline);
    }
    state.selectedTrackline = layer;
    state.selectedParentLayer = parentLayer;

    layer.setStyle({
        weight: 6, opacity: 1, color: '#F57D15', dashArray: '',
        filter: 'drop-shadow(0px 0px 4px rgba(245,125,21,0.8))',
    });
    layer.bringToFront();

    if (typeof window.showWaterfallSidebar === 'function') {
        window.showWaterfallSidebar(feature);
    }

    state.map.panTo(layer.getBounds().getCenter(), { animate: true });
}


export function loadTracklines() {
    const map = state.map;

    fetch(`${API}/api/tracklines`).then(r => r.json()).then(geojson => {
        const commonOptions = {
            style: (feature) => (feature.properties.instrument === 'SSS')
                ? { color: '#ffffff', weight: 3, opacity: 0.8, dashArray: '', lineCap: 'round' }
                : { color: '#ec4899', weight: 3, opacity: 0.8, dashArray: '' },

            onEachFeature: (feature, layer) => {
                const props = feature.properties;
                const tooltipHtml = `
                    <div class="px-3 py-2 font-sans">
                        <div class="font-bold text-orange-400 border-b border-slate-600 pb-1 mb-1 text-[11px] uppercase tracking-wider">
                            ${props.instrument === 'SSS' ? '🌊 Side Scan Sonar' : '🕳️ Sub-bottom Profiler'}
                        </div>
                        <div class="text-slate-300 text-xs">File: <span class="text-white font-mono">${props.file || 'Unknown'}</span></div>
                    </div>
                `;
                layer.bindTooltip(tooltipHtml, { sticky: true, className: 'custom-track-tooltip' });

                layer.on('click', (e) => {
                    L.DomEvent.stopPropagation(e);
                    const parent = (feature.properties.instrument === 'SSS') ? state.sssLayer : state.sbpLayer;
                    selectTrackline(feature, layer, parent);
                });
                layer.on('mouseover', () => {
                    if (layer !== state.selectedTrackline) {
                        layer.setStyle({ weight: 5, opacity: 1, color: '#38bdf8' });
                        layer.bringToFront();
                    }
                });
                layer.on('mouseout', () => {
                    if (layer !== state.selectedTrackline) {
                        const parent = (feature.properties.instrument === 'SSS') ? state.sssLayer : state.sbpLayer;
                        parent.resetStyle(layer);
                    }
                });
            },
        };

        state.sssLayer = L.geoJSON(geojson, { ...commonOptions, filter: (f) => f.properties.instrument === 'SSS' });
        state.sbpLayer = L.geoJSON(geojson, { ...commonOptions, filter: (f) => f.properties.instrument === 'SBP' });
    });
}


export function loadMagTargets() {
    fetch(`${API}/api/mag-targets`).then(r => r.json()).then(geojson => {
        magTargetsLayer = L.geoJSON(geojson, {
            pointToLayer: (feature, latlng) => {
                const a = feature.properties.anomaly_nT;
                const radius = Math.min(14, Math.max(5, Math.log10(Math.abs(a)) * 4));
                const fill = a > 0 ? '#ef4444' : '#3b82f6';
                return L.circleMarker(latlng, {
                    radius,
                    fillColor: fill,
                    color: '#ffffff',
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.85,
                });
            },
            onEachFeature: (feature, layer) => {
                const p = feature.properties;
                const popupHtml = `
                    <div class="px-2 py-1 font-sans">
                        <div class="font-bold text-orange-500 border-b border-slate-300 pb-1 mb-1 text-[11px] uppercase tracking-wider">
                            🧲 MAG Candidate
                        </div>
                        <div class="text-xs space-y-0.5">
                            <div><span class="text-slate-500">ID:</span> <span class="font-mono">${p.target_id}</span></div>
                            <div><span class="text-slate-500">Anomaly:</span> <span class="font-mono">${p.anomaly_nT > 0 ? '+' : ''}${p.anomaly_nT} nT</span></div>
                            <div><span class="text-slate-500">Polarity:</span> ${p.polarity}</div>
                        </div>
                    </div>
                `;
                layer.bindPopup(popupHtml);
            },
        });
    });
}


export function bindTracklinesUI() {
    const map = state.map;

    document.getElementById('chk-trackline-sss')?.addEventListener('change', (e) => {
        if (e.target.checked && state.sssLayer) state.sssLayer.addTo(map);
        else if (state.sssLayer) map.removeLayer(state.sssLayer);
    });

    document.getElementById('chk-trackline-sbp')?.addEventListener('change', (e) => {
        if (e.target.checked && state.sbpLayer) state.sbpLayer.addTo(map);
        else if (state.sbpLayer) map.removeLayer(state.sbpLayer);
    });

    document.getElementById('chk-mag-targets')?.addEventListener('change', (e) => {
        if (!magTargetsLayer) return;
        if (e.target.checked) magTargetsLayer.addTo(map);
        else map.removeLayer(magTargetsLayer);
    });
}