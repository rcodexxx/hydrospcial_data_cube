import { state } from '../state.js';
import { API, SEDIMENT_COLORS } from '../constants.js';
import { interpolatePolyline } from '../utils.js';
import { openPanels } from './layout.js';


export function loadWaterfallIndex() {
    fetch(`${API}/api/waterfall-index`).then(r => r.json()).then(data => {
        state.waterfallIndex = data;
        // SSSModal still reads from window.waterfallIndex; keep it for now
        window.waterfallIndex = data;
    });
}


export function showWaterfallSidebar(feature) {
    if (!state.waterfallIndex) return;

    const props = feature.properties;
    const filename = props.file;

    state.currentTrackCoords = interpolatePolyline(state.map, feature.geometry.coordinates, 100);
    state.currentWfPings = props.pings || 100;

    if (props.instrument === 'SSS') {
        openPanels('sss');
        document.getElementById('rp-title').textContent = `SSS Viewer - ${filename || 'Unknown'}`;
        document.getElementById('rp-echarts-cursor')?.classList.remove('hidden');

        const lfEntry = state.waterfallIndex.sss[`${filename}_LF`];
        if (lfEntry) document.getElementById('img-preview').src = `/waterfalls/${lfEntry.image}`;

        document.getElementById('rp-expand-btn').onclick = () => {
            window.SSSModal?.open(filename);
        };

        loadProfileData('rp', state.currentTrackCoords);
    } else if (props.instrument === 'SBP') {
        openPanels('sbp');
        document.getElementById('bp-title').textContent = `SBP Viewer - ${filename || 'Unknown'}`;
        document.getElementById('bp-echarts-cursor')?.classList.remove('hidden');

        if (state.waterfallIndex.sbp[filename]) {
            document.getElementById('bp-sbp-image').src = `/waterfalls/${state.waterfallIndex.sbp[filename].image}`;
            document.getElementById('bp-sbp-cursor')?.classList.remove('hidden');
        }
        loadProfileData('bp', state.currentTrackCoords);
    }
}


function loadProfileData(panelPrefix, coords) {
    const coordStr = coords.map(c => `${c[0]},${c[1]}`).join(';');
    fetch(`${API}/api/profile?coords=${encodeURIComponent(coordStr)}`).then(r => r.json()).then(data => {
        const container = document.getElementById(`${panelPrefix}-echarts-container`);
        if (container) {
            container._depthData = data.depth;
            renderProfileChart(`${panelPrefix}-echarts-container`, data.depth, data.isopach, data.sediment);
        }
        document.getElementById(`${panelPrefix}-slider`)?.dispatchEvent(new Event('input'));
    });
}


export function renderProfileChart(containerId, depth, isopach, sediment) {
    const container = document.getElementById(containerId);
    if (!container) return;
    if (container._chart) container._chart.dispose();
    const chart = echarts.init(container);
    container._chart = chart;

    const showIsopach = state.HAS_ISOPACH && isopach && isopach.some(v => v !== null && !isNaN(v));

    const seafloor = [];
    const subbottom = [];
    const validDepths = depth.filter(v => v !== null && !isNaN(v));
    if (validDepths.length === 0) return;

    for (let i = 0; i < depth.length; i++) {
        const pct = (i / (depth.length - 1)) * 100;
        if (depth[i] !== null && !isNaN(depth[i])) {
            seafloor.push([pct, depth[i]]);
            if (showIsopach) {
                const thick = (isopach[i] !== null && !isNaN(isopach[i])) ? isopach[i] : 0;
                subbottom.push([pct, depth[i] + thick]);
            }
        }
    }
    if (seafloor.length === 0) return;

    // Sediment polygon series — only if isopach gives band thickness
    const sedimentSeries = [];
    if (showIsopach && sediment && sediment.length > 0) {
        const customData = [];
        for (let i = 0; i < depth.length; i++) {
            if (depth[i] === null || isNaN(depth[i])) continue;
            const thick = (isopach[i] !== null && !isNaN(isopach[i])) ? isopach[i] : 0;
            if (thick <= 0) continue;
            const cls = (sediment[i] !== null && !isNaN(sediment[i]) && sediment[i] >= 0)
                ? Math.round(sediment[i]) : -1;
            const pct = (i / (depth.length - 1)) * 100;
            customData.push({
                pct,
                depthTop: depth[i],
                depthBot: depth[i] + thick,
                classId: cls,
            });
        }

        sedimentSeries.push({
            name: 'Sediment',
            type: 'custom',
            data: customData,
            renderItem: (params, api) => {
                const idx = params.dataIndex;
                const curr = customData[idx];
                if (idx >= customData.length - 1) return null;
                const next = customData[idx + 1];
                const color = (curr.classId >= 0 && curr.classId < SEDIMENT_COLORS.length)
                    ? SEDIMENT_COLORS[curr.classId] : '#d1d5db';
                return {
                    type: 'polygon',
                    shape: { points: [
                        api.coord([curr.pct, curr.depthTop]),
                        api.coord([next.pct, next.depthTop]),
                        api.coord([next.pct, next.depthBot]),
                        api.coord([curr.pct, curr.depthBot]),
                    ]},
                    style: { fill: color, opacity: 1 },
                };
            },
        });
    }

    const series = [
        // Water area: always show
        {
            name: 'Water', type: 'line', data: seafloor,
            symbol: 'none', lineStyle: { width: 0 },
            areaStyle: { color: 'rgba(59,130,246,0.15)', origin: 'start' },
            z: 0, silent: true,
        },
    ];

    if (showIsopach) {
        series.push({
            name: 'Bedrock Fill', type: 'line', data: subbottom,
            symbol: 'none', lineStyle: { width: 0 },
            areaStyle: { color: '#64748b', opacity: 0.15, origin: 'end' },
            z: 1, silent: true,
        });
        series.push(...sedimentSeries);
        series.push({
            name: 'Isopach Base', type: 'line', data: subbottom,
            symbol: 'none',
            lineStyle: { color: '#64748b', width: 1.5, type: 'dashed' },
            z: 11,
        });
    }

    series.push({
        name: 'Seafloor', type: 'line', data: seafloor,
        symbol: 'none',
        lineStyle: { color: '#0f172a', width: 1.5 },
        z: 10,
    });

    chart.setOption({
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'axis', showContent: false,
            axisPointer: {
                type: 'cross',
                lineStyle: { color: '#FF5722', width: 1, type: 'dashed' },
                crossStyle: { color: '#FF5722', width: 1, type: 'dashed' },
            },
        },
        grid: { top: 30, bottom: 20, left: 40, right: 30 },
        xAxis: { type: 'value', min: 0, max: 100, splitLine: { show: false }, axisLabel: { show: false } },
        yAxis: { type: 'value', inverse: true, scale: true, axisLabel: { fontSize: 10 }, splitLine: { lineStyle: { type: 'dashed', color: '#e2e8f0' } } },
        series,
    });

    window.addEventListener('resize', () => chart.resize());
    setTimeout(() => chart.resize(), 350);
}


function handleSlider(e, panelPrefix) {
    const map = state.map;
    const pct = e.target.value / 100;
    const eContainer = document.getElementById(`${panelPrefix}-echarts-container`);

    if (eContainer && eContainer._chart) {
        const chart = eContainer._chart;
        const xPixel = chart.convertToPixel({ xAxisIndex: 0 }, pct * 100);
        if (xPixel != null && !isNaN(xPixel)) {
            chart.dispatchAction({
                type: 'showTip',
                x: xPixel,
                y: chart.getHeight() / 2,
            });
        }
    }

    if (panelPrefix === 'rp') {
        document.querySelectorAll('.wf-redline').forEach(el => {
            const img = el.parentElement.querySelector('img');
            if (img) el.style.top = (pct * img.clientHeight) + 'px';
        });
    }
    if (panelPrefix === 'bp') {
        const sbpCursor = document.getElementById('bp-sbp-cursor');
        const sbpImg = document.getElementById('bp-sbp-image');
        if (sbpCursor && sbpImg) sbpCursor.style.left = `${pct * sbpImg.clientWidth}px`;
    }

    if (state.currentTrackCoords.length > 0) {
        const dataIndex = Math.min(Math.floor(pct * state.currentTrackCoords.length), state.currentTrackCoords.length - 1);
        const coord = state.currentTrackCoords[dataIndex];

        if (state.mapTrackMarker) {
            state.mapTrackMarker.setLatLng([coord[1], coord[0]]);
        } else {
            const sonarIcon = L.divIcon({ className: 'sonar-ping-marker', iconSize: [12, 12], iconAnchor: [6, 6] });
            state.mapTrackMarker = L.marker([coord[1], coord[0]], { icon: sonarIcon }).addTo(map);
        }
    }
}


export function bindSliders() {
    document.getElementById('bp-slider')?.addEventListener('input', (e) => handleSlider(e, 'bp'));
    document.getElementById('rp-slider')?.addEventListener('input', (e) => handleSlider(e, 'rp'));
}