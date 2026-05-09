import L from 'leaflet';
import { state } from '../state.js';
import { API, INITIAL_CENTER, INITIAL_ZOOM } from '../constants.js';
import { interpolatePolyline } from '../utils.js';
import { openPanels, closePanels } from './layout.js';
import { doPointQuery } from './popup.js';
import { doRegionSelect } from './region.js';
import { renderProfileChart } from './waterfall.js';
import { closeBorehole } from './borehole.js';
import { close3D } from './block3d.js';


function bindToolButtons() {
    const map = state.map;

    document.querySelectorAll('.tool-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const targetBtn = e.currentTarget;
            const toolId = targetBtn.dataset.tool;

            document.querySelectorAll('.tool-btn').forEach(b => {
                b.classList.remove('active', 'bg-blue-600', 'text-white', 'shadow-md');
                b.classList.add('text-slate-500', 'hover:text-slate-800', 'hover:bg-slate-100');
            });
            targetBtn.classList.remove('text-slate-500', 'hover:text-slate-800', 'hover:bg-slate-100');
            targetBtn.classList.add('active', 'bg-blue-600', 'text-white', 'shadow-md');

            map.closePopup();
            closePanels();
            close3D();
            closeBorehole();

            if (state.linePreview) { map.removeLayer(state.linePreview); state.linePreview = null; }
            state.lineStart = null;
            if (state.selectRect) { map.removeLayer(state.selectRect); state.selectRect = null; }
            state.selectStart = null;
            if (state.drawnLine) { map.removeLayer(state.drawnLine); state.drawnLine = null; }
            if (state.clickMarker) { map.removeLayer(state.clickMarker); state.clickMarker = null; }

            state.currentTool = toolId;
            const mapDiv = document.getElementById('map');
            mapDiv.classList.remove('cursor-pan', 'cursor-query', 'cursor-line', 'cursor-select');
            mapDiv.classList.add(`cursor-${state.currentTool}`);

            if (state.currentTool === 'pan') {
                map.dragging.enable();
                map.touchZoom.enable();
                map.doubleClickZoom.enable();
                map.scrollWheelZoom.enable();
                map.boxZoom.enable();
            } else {
                map.dragging.disable();
                map.touchZoom.disable();
                map.doubleClickZoom.disable();
                map.scrollWheelZoom.disable();
                map.boxZoom.disable();
            }
        });
    });
}


function bindMapMouseEvents() {
    const map = state.map;

    map.on('click', (e) => {
        if (state.currentTool === 'query') doPointQuery(e.latlng.lat, e.latlng.lng);
    });

    map.on('mousedown', (e) => {
        if (state.currentTool === 'select') state.selectStart = e.latlng;
        else if (state.currentTool === 'line') state.lineStart = e.latlng;
    });

    map.on('mousemove', (e) => {
        const cd = document.getElementById('coord-display');
        if (cd) cd.textContent = `${e.latlng.lat.toFixed(6)}°N, ${e.latlng.lng.toFixed(6)}°E`;

        if (state.currentTool === 'select' && state.selectStart) {
            if (state.selectRect) map.removeLayer(state.selectRect);
            state.selectRect = L.rectangle([state.selectStart, e.latlng], {
                color: '#2563eb', weight: 2, fillOpacity: 0.15, dashArray: '5,5',
            }).addTo(map);
        }
        if (state.currentTool === 'line' && state.lineStart) {
            if (state.linePreview) map.removeLayer(state.linePreview);
            state.linePreview = L.polyline([state.lineStart, e.latlng], {
                color: '#F57D15', weight: 3, dashArray: '8,4',
            }).addTo(map);
        }
    });

    map.on('mouseup', (e) => {
        if (state.currentTool === 'select' && state.selectStart) {
            const bounds = L.latLngBounds(state.selectStart, e.latlng);
            state.selectStart = null;
            if (!bounds.getNorthEast().equals(bounds.getSouthWest())) doRegionSelect(bounds);
        }
        if (state.currentTool === 'line' && state.lineStart) {
            const endPoint = e.latlng;
            if (state.linePreview) map.removeLayer(state.linePreview);
            state.linePreview = null;

            if (map.distance(state.lineStart, endPoint) > 5) {
                if (state.drawnLine) map.removeLayer(state.drawnLine);
                state.drawnLine = L.polyline([state.lineStart, endPoint], {
                    color: '#F57D15', weight: 3,
                }).addTo(map);

                openPanels('drawn-line');
                const bpTitle = document.getElementById('bp-title');
                if (bpTitle) bpTitle.textContent = '✏️ Hand-Drawn Profile';

                state.currentTrackCoords = interpolatePolyline(
                    map,
                    [[state.lineStart.lng, state.lineStart.lat], [endPoint.lng, endPoint.lat]],
                    100
                );
                state.currentWfPings = 100;

                const d = map.distance(state.lineStart, endPoint);
                const infoText = document.getElementById('bp-info-text');
                if (infoText) infoText.textContent = `Length: ${d.toFixed(0)}m`;

                const coordStr = state.currentTrackCoords.map(c => `${c[0]},${c[1]}`).join(';');
                fetch(`${API}/api/profile?coords=${encodeURIComponent(coordStr)}`)
                    .then(r => r.json())
                    .then(data => {
                        renderProfileChart('bp-echarts-container', data.depth, data.isopach, data.sediment);
                    });
            }
            state.lineStart = null;
        }
    });
}


function resetMapState() {
    const map = state.map;
    closePanels();
    map.closePopup();
    close3D();
    closeBorehole();

    if (state.clickMarker) { map.removeLayer(state.clickMarker); state.clickMarker = null; }
    if (state.selectRect) { map.removeLayer(state.selectRect); state.selectRect = null; }
    if (state.linePreview) { map.removeLayer(state.linePreview); state.linePreview = null; }
    if (state.drawnLine) { map.removeLayer(state.drawnLine); state.drawnLine = null; }

    map.setView(INITIAL_CENTER, INITIAL_ZOOM);
    document.getElementById('btn-tool-pan')?.click();
}


export function bindToolbar() {
    bindToolButtons();
    bindMapMouseEvents();
    document.getElementById('btn-reset')?.addEventListener('click', resetMapState);

    // For HTML inline onclick="resetMapState()" if any
    window.resetMapState = resetMapState;
}