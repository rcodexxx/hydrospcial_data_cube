const API = '';

const SED_NATURAL_COLORS = [
    '#c2a64d',  // 0: Coarse sand - 深黃棕
    '#d4b96a',  // 1: Fine sand - 淺黃棕  
    '#dcc88a',  // 2: Very fine sand - 米黃
    '#b8a88a',  // 3: Silty sand - 灰棕
    '#a09880',  // 4: Sandy silt - 暗灰棕
    '#8a8578',  // 5: Silt - 灰褐
    '#7a7a70',  // 6: Sandy-silt-clay - 暗灰
    '#6b6e6a',  // 7: Silty clay - 深灰
    '#5c6260',  // 8: Clayey silt - 暗灰綠
    '#4a5550',  // 9: Framework-supported mud - 深灰綠
    '#3a4a55',  // 10: Fluid mud - 深藍灰
];

const SED_LABELS = [
    'Coarse sand', 'Fine sand', 'Very fine sand', 'Silty sand',
    'Sandy silt', 'Silt', 'Sandy-silt-clay', 'Silty clay',
    'Clayey silt', 'Framework mud', 'Fluid mud'
];

const TARGETS = [
    { id: 'TG01', lat: 22.136756, lon: 120.787840, depth: 27.2, size: '3.13×0.68×0.2', result: '樹幹', sss: true, mbes: true, sbp: false, mag: false, dive: true },
    { id: 'TG02', lat: 22.138375, lon: 120.789651, depth: 17.7, size: '4.13×0.45×0.4', result: '石塊', sss: true, mbes: true, sbp: false, mag: false, dive: true },
    { id: 'TG03', lat: 22.135644, lon: 120.787286, depth: 30.0, size: '7.69×6.13×1.1', result: '樹幹與樹枝', sss: true, mbes: true, sbp: false, mag: false, dive: true },
    { id: 'TG04', lat: 22.138672, lon: 120.789286, depth: 17.6, size: '5.77×1.37×0.5', result: '樹幹', sss: true, mbes: true, sbp: true, mag: false, dive: true },
    { id: 'TG05', lat: 22.138724, lon: 120.788889, depth: 18.1, size: '3.03×0.40×0.3', result: '樹幹', sss: true, mbes: true, sbp: false, mag: false, dive: true },
    { id: 'TG06', lat: 22.133582, lon: 120.780874, depth: 34.3, size: '2.10×0.95×0.1', result: '疑似石塊', sss: true, mbes: true, sbp: false, mag: false, dive: false },
    { id: 'TG07', lat: 22.135546, lon: 120.780076, depth: 21.7, size: '7.50×2.49×0.6', result: '廢棄鋁架', sss: true, mbes: true, sbp: false, mag: false, dive: true },
    { id: 'TG08', lat: 22.136010, lon: 120.779564, depth: 23.6, size: '0.61×0.51×0.6', result: '廢棄塑膠水桶', sss: true, mbes: true, sbp: false, mag: false, dive: true },
    { id: 'TG09', lat: 22.136240, lon: 120.781183, depth: 18.0, size: '0.97×0.88×0.7', result: '石塊', sss: true, mbes: true, sbp: false, mag: false, dive: true },
];

let currentOverlay = null, tileLayers = {};
let clickMarker = null, selectRect = null, selectStart = null;
let targetMarkers = [], targetLabels = [];
let sssLayer = null, sbpLayer = null;
let selectedTrackline = null, selectedParentLayer = null;
let currentTool = 'pan', lineStart = null, linePreview = null, drawnLine = null;
let waterfallIndex = null, currentWfPings = 0, mapTrackMarker = null, currentTrackCoords = [];

const INITIAL_CENTER = [22.137, 120.785], INITIAL_ZOOM = 15;

const map = L.map('map', { center: INITIAL_CENTER, zoom: INITIAL_ZOOM, maxZoom: 22, minZoom: 14, renderer: L.canvas({ tolerance: 15 }), zoomControl: false });
L.control.zoom({ position: 'bottomright' }).addTo(map);
L.control.scale({ position: 'bottomright', metric: true, imperial: false }).addTo(map);
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', { attribution: 'Esri', maxZoom: 18, maxNativeZoom: 18 }).addTo(map);

fetch(API + '/api/layers').then(r => r.json()).then(data => {
    if (data.bounds) { map.setMaxBounds(L.latLngBounds(data.bounds).pad(0.3)); map.fitBounds(L.latLngBounds(data.bounds).pad(0.05)); }
    for (const [key, cfg] of Object.entries(data.layers)) { tileLayers[key] = L.tileLayer(cfg.url, { opacity: 0.75, maxZoom: 22, maxNativeZoom: 22 }); }
    if (tileLayers['bathymetry']) { tileLayers['bathymetry'].addTo(map); currentOverlay = tileLayers['bathymetry']; }
});

// ── Sediment Legend Control ──────────────────────────────────
const sedLegend = L.control({ position: 'bottomright' });
sedLegend.onAdd = function() {
    const div = L.DomUtil.create('div');
    div.style.cssText = 'background:white;padding:8px 12px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);font-size:11px;line-height:1.8;';
    div.innerHTML = '<div style="font-weight:bold;margin-bottom:4px;color:#334155;">Sediment Class</div>'
        + SED_NATURAL_COLORS.map((c, i) =>
            `<div style="display:flex;align-items:center;gap:6px;">`
            + `<span style="display:inline-block;width:14px;height:14px;border-radius:3px;background:${c};border:1px solid rgba(0,0,0,0.1);"></span>`
            + `<span style="color:#475569;">${SED_LABELS[i]}</span>`
            + `</div>`
        ).join('');
    return div;
};
let sedLegendAdded = false;

// ── 💡 核心演算法：沿線均勻插值 (解決轉彎距離失真與掉幀問題) ──────
function interpolatePolyline(coords, numPoints) {
    if (coords.length < 2) return Array(numPoints).fill(coords[0] || [0,0]);
    const cumDist = [0];
    for (let i = 1; i < coords.length; i++) {
        cumDist.push(cumDist[i-1] + map.distance(L.latLng(coords[i-1][1], coords[i-1][0]), L.latLng(coords[i][1], coords[i][0])));
    }
    const totalDist = cumDist[cumDist.length - 1];
    if (totalDist === 0) return Array(numPoints).fill(coords[0]);
    
    const step = totalDist / (numPoints - 1);
    const result = [];
    for (let i = 0; i < numPoints; i++) {
        const targetDist = i * step;
        if (i === 0) { result.push(coords[0]); continue; }
        if (i === numPoints - 1) { result.push(coords[coords.length - 1]); continue; }
        
        let segIdx = cumDist.findIndex(d => d >= targetDist) - 1;
        if (segIdx < 0) segIdx = 0;
        
        const ratio = (targetDist - cumDist[segIdx]) / (cumDist[segIdx + 1] - cumDist[segIdx]);
        const lon = coords[segIdx][0] + ratio * (coords[segIdx+1][0] - coords[segIdx][0]);
        const lat = coords[segIdx][1] + ratio * (coords[segIdx+1][1] - coords[segIdx][1]);
        result.push([lon, lat]);
    }
    return result;
}

// ── 💡 佈局引擎 (Layout Manager) ──────────────────────────
const mapWrapper = document.getElementById('map-wrapper');
const sidebar = document.getElementById('sidebar');
const bottomPanel = document.getElementById('bottom-panel');
const rightPanel = document.getElementById('right-panel');

let isSidebarOpen = true;
let currentRightWidth = 0;   
let currentBottomHeight = 0; 

function applyLayout() {
    const leftOffset = isSidebarOpen ? 320 : 0;
    sidebar.style.transform = isSidebarOpen ? 'translateX(0)' : 'translateX(-100%)';
    
    rightPanel.style.transform = currentRightWidth > 0 ? 'translateX(0)' : 'translateX(100%)';
    rightPanel.style.width = currentRightWidth > 0 ? `${currentRightWidth}px` : '0px';

    bottomPanel.style.transform = currentBottomHeight > 0 ? 'translateY(0)' : 'translateY(100%)';
    bottomPanel.style.height = currentBottomHeight > 0 ? `${currentBottomHeight}px` : '0px';
    bottomPanel.style.left = `${leftOffset}px`;
    bottomPanel.style.right = currentRightWidth > 0 ? `${currentRightWidth}px` : '0px'; // L型避讓地圖

    mapWrapper.style.left = `${leftOffset}px`;
    mapWrapper.style.right = currentRightWidth > 0 ? `${currentRightWidth}px` : '0px';
    mapWrapper.style.bottom = currentBottomHeight > 0 ? `${currentBottomHeight}px` : '0px';

    setTimeout(() => {
        map.invalidateSize({animate: true});
        document.getElementById('bp-echarts-container')?._chart?.resize();
        document.getElementById('rp-echarts-container')?._chart?.resize();
    }, 300);
}

document.getElementById('btn-toggle-sidebar')?.addEventListener('click', () => {
    isSidebarOpen = !isSidebarOpen;
    applyLayout();
});

window.closePanels = function() {
    currentRightWidth = 0; currentBottomHeight = 0; applyLayout();
    if (mapTrackMarker) { map.removeLayer(mapTrackMarker); mapTrackMarker = null; }
    if (selectedTrackline && selectedParentLayer) { selectedParentLayer.resetStyle(selectedTrackline); selectedTrackline = null; }
}

window.resetMapState = function() {
    closePanels(); 
    map.closePopup();
    
    if (clickMarker) { map.removeLayer(clickMarker); clickMarker = null; }
    if (selectRect) { map.removeLayer(selectRect); selectRect = null; }
    if (linePreview) { map.removeLayer(linePreview); linePreview = null; }
    if (drawnLine) { map.removeLayer(drawnLine); drawnLine = null; }
    
    map.setView(INITIAL_CENTER, INITIAL_ZOOM);
    
    document.querySelectorAll('.tool-btn').forEach(b => { 
        if (b.id !== 'btn-reset' && b.id !== 'btn-toggle-sidebar') { 
            b.classList.remove('active', 'bg-blue-50', 'text-blue-600', 'shadow-sm'); 
            b.classList.add('text-slate-500'); 
        } 
    });
    document.querySelector('[data-tool="pan"]')?.classList.add('active', 'bg-blue-50', 'text-blue-600', 'shadow-sm');
    currentTool = 'pan'; 

    // 💡 安全切換游標
    const mapEl = document.getElementById('map');
    if (mapEl) {
        mapEl.classList.remove('cursor-query', 'cursor-line', 'cursor-select');
        mapEl.classList.add('cursor-pan');
    }
    map.dragging.enable();
}
document.getElementById('btn-reset')?.addEventListener('click', resetMapState);

function openPanels(mode) {
    document.getElementById('bp-sbp-section')?.classList.add('hidden');
    document.getElementById('bp-echarts-cursor')?.classList.add('hidden');
    document.getElementById('rp-echarts-cursor')?.classList.add('hidden');
    
    // 💡 安全防呆：如果找不到 Slider 也不會報錯當機
    document.getElementById('bp-slider')?.classList.add('hidden');
    document.getElementById('rp-slider')?.classList.add('hidden');
    
    if (mode === 'drawn-line') {
        currentRightWidth = 0; currentBottomHeight = window.innerHeight * 0.35;
    } else if (mode === 'sss') {
        currentBottomHeight = 0; currentRightWidth = 450; 
        document.getElementById('rp-slider')?.classList.remove('hidden');
    } else if (mode === 'sbp') {
        currentRightWidth = 0; currentBottomHeight = window.innerHeight * 0.55;
        document.getElementById('bp-sbp-section')?.classList.remove('hidden');
        document.getElementById('bp-slider')?.classList.remove('hidden');
    }
    applyLayout();
}

// ── 💡 Targets 與 Popup 邏輯 ───────────────────────────
TARGETS.forEach(t => {
    // 1. 建立橘色三角形地標
    const mk = L.marker([t.lat, t.lon], { 
        icon: L.divIcon({ 
            html: `<div style="width:0;height:0;border-left:8px solid transparent;border-right:8px solid transparent;border-bottom:14px solid #F57D15;filter:drop-shadow(0 2px 2px rgba(0,0,0,0.5));cursor:pointer;"></div>`, 
            className: '', 
            iconSize: [16, 14], 
            iconAnchor: [8, 14] 
        }) 
    }).addTo(map);

    // 2. 建立黑色小標籤 (如 TG01)
    const lb = L.marker([t.lat, t.lon], { 
        icon: L.divIcon({ 
            html: `<div style="color:#FFF;font-size:10px;font-weight:bold;background:rgba(0,0,0,0.6);padding:2px 4px;border-radius:4px;white-space:nowrap;box-shadow:0 1px 3px rgba(0,0,0,0.3);">${t.id}</div>`, 
            className: '', 
            iconSize: [50, 16], 
            iconAnchor: [-5, 20] 
        }), 
        interactive: false // 標籤不阻擋點擊
    }).addTo(map);

    // 3. 點擊事件：在地圖上彈出精緻的 Popup
    mk.on('click', (e) => {
        L.DomEvent.stopPropagation(e); // 防止觸發地圖底層的點擊事件

        // 整理 Sensors 陣列
        let sensors = [];
        if (t.sss) sensors.push('SSS'); 
        if (t.mbes) sensors.push('MBES'); 
        if (t.sbp) sensors.push('SBP'); 
        if (t.mag) sensors.push('MAG'); 
        if (t.dive) sensors.push('Dive');

        // Popup 初始 HTML (基本資訊 + 讀取中狀態)
        let html = `
            <div class="w-64">
                <div class="font-bold text-orange-600 mb-1 text-base">🔺 ${t.id} — ${t.result}</div>
                <div class="text-[10px] text-slate-500 mb-2 font-mono">📍 ${t.lat.toFixed(6)}°N, ${t.lon.toFixed(6)}°E</div>
                <div class="text-xs flex justify-between border-b border-slate-100 py-1">
                    <span class="font-semibold text-slate-600">Depth:</span> 
                    <span class="text-slate-900 font-medium">${t.depth} m</span>
                </div>
                <div class="text-xs flex justify-between border-b border-slate-100 py-1">
                    <span class="font-semibold text-slate-600">Size:</span> 
                    <span class="text-slate-900 font-medium">${t.size} m</span>
                </div>
                <div class="text-xs flex justify-between border-b border-slate-100 py-1">
                    <span class="font-semibold text-slate-600">Sensors:</span> 
                    <span class="text-slate-900 font-medium">${sensors.join(', ')}</span>
                </div>
                <div id="env-data-${t.id}" class="mt-2 text-xs text-slate-400 italic animate-pulse">
                    Fetching environment data...
                </div>
            </div>
        `;

        // 開啟 Popup
        const popup = L.popup({ maxWidth: 300, minWidth: 250 })
            .setLatLng([t.lat, t.lon])
            .setContent(html)
            .openOn(map);

        // 打 API 拿水下環境與底層資料 (如沉積物、磁力異常)
        fetch(`${API}/api/query?lat=${t.lat}&lon=${t.lon}`)
            .then(r => r.json())
            .then(data => {
                let envHtml = `<div class="border-t border-slate-200 mt-1 pt-1">`;
                for (const [key, info] of Object.entries(data)) {
                    // 過濾掉不必要的座標欄位或空值
                    if (['lat', 'lon', 'x_3826', 'y_3826'].includes(key) || !info || !info.name || info.value === null) continue;
                    
                    if (key === 'sediment_class') {
                        const color = SED_COLORS[info.class_id] || '#888';
                        envHtml += `
                            <div class="text-xs flex justify-between py-1 items-center">
                                <span class="font-semibold text-slate-600">${info.name}:</span> 
                                <span class="text-[10px] text-white px-1.5 py-0.5 rounded shadow-sm" style="background:${color}">${info.value}</span>
                            </div>`;
                    } else {
                        envHtml += `
                            <div class="text-xs flex justify-between py-1">
                                <span class="font-semibold text-slate-600">${info.name}:</span> 
                                <span class="text-slate-900 font-medium">${info.value} <span class="text-[10px] text-slate-500">${info.units}</span></span>
                            </div>`;
                    }
                }
                envHtml += `</div>`;
                
                // 更新 Popup 內容
                const envDiv = document.getElementById(`env-data-${t.id}`);
                if (envDiv) { 
                    envDiv.innerHTML = envHtml; 
                    popup.update(); // 讓 Leaflet 重新計算 Popup 高度
                }
            })
            .catch(err => {
                const envDiv = document.getElementById(`env-data-${t.id}`);
                if (envDiv) envDiv.innerHTML = `<span class="text-red-500 text-xs">Failed to load data.</span>`;
            });
    });

    targetMarkers.push(mk); 
    targetLabels.push(lb);
});

// 4. 左側選單的 Checkbox 開關綁定
document.getElementById('chk-targets')?.addEventListener('change', (e) => {
    targetMarkers.forEach(m => e.target.checked ? m.addTo(map) : map.removeLayer(m));
    targetLabels.forEach(l => e.target.checked ? l.addTo(map) : map.removeLayer(l));
});
// ──────────────────────────────────────────────────────────────────

// ── 💡 拖曳調整引擎 (Resizers) ──────────────────────────────────
let resizeTarget = null; 

document.getElementById('rp-resizer')?.addEventListener('mousedown', () => { resizeTarget = 'right'; document.body.style.cursor = 'col-resize'; mapWrapper.style.transition = 'none'; });
document.getElementById('bp-resizer')?.addEventListener('mousedown', () => { resizeTarget = 'bottom'; document.body.style.cursor = 'row-resize'; mapWrapper.style.transition = 'none'; });
document.getElementById('rp-internal-resizer')?.addEventListener('mousedown', () => { resizeTarget = 'rp-internal'; document.body.style.cursor = 'row-resize'; document.body.classList.add('no-select'); });
document.getElementById('bp-internal-resizer')?.addEventListener('mousedown', () => { resizeTarget = 'bp-internal'; document.body.style.cursor = 'row-resize'; document.body.classList.add('no-select'); });

window.addEventListener('mousemove', (e) => {
    if (!resizeTarget) return;
    if (resizeTarget === 'right') {
        let newW = window.innerWidth - e.clientX;
        if (newW > 300 && newW < window.innerWidth - 350) { currentRightWidth = newW; applyLayout(); }
    } else if (resizeTarget === 'bottom') {
        let newH = window.innerHeight - e.clientY;
        if (newH > 150 && newH < window.innerHeight - 100) { currentBottomHeight = newH; applyLayout(); }
    } else if (resizeTarget === 'rp-internal') {
        let topSec = document.getElementById('rp-top-section');
        let newH = e.clientY - topSec.getBoundingClientRect().top;
        if (newH > 100 && newH < window.innerHeight - 200) { topSec.style.height = `${newH}px`; topSec.querySelector('#rp-echarts-container')?._chart?.resize(); }
    } else if (resizeTarget === 'bp-internal') {
        let topSec = document.getElementById('bp-top-section');
        let newH = e.clientY - topSec.getBoundingClientRect().top;
        if (newH > 100 && newH < currentBottomHeight - 100) { topSec.style.height = `${newH}px`; topSec.querySelector('#bp-echarts-container')?._chart?.resize(); }
    }
});

window.addEventListener('mouseup', () => {
    if (resizeTarget) {
        resizeTarget = null; document.body.style.cursor = ''; document.body.classList.remove('no-select');
        mapWrapper.style.transition = 'left 0.3s, right 0.3s, bottom 0.3s ease-in-out';
        map.invalidateSize({animate: false});
        document.getElementById('bp-slider')?.dispatchEvent(new Event('input'));
        document.getElementById('rp-slider')?.dispatchEvent(new Event('input'));
    }
});

// ── 💡 統一的 ECharts 繪製引擎 (使用百分比 X 軸完美對齊影像) ─────
function renderProfileChart(containerId, depth, isopach, sediment) {
    const container = document.getElementById(containerId);
    if (!container) return;
    if (container._chart) container._chart.dispose();
    const chart = echarts.init(container);
    container._chart = chart;

    const seafloor = [], subbottom = [];
    for (let i = 0; i < depth.length; i++) {
        const pct = (i / (depth.length - 1)) * 100;
        if (depth[i] !== null && !isNaN(depth[i])) {
            seafloor.push([pct, depth[i]]);
            const thick = (isopach && isopach[i] !== null && !isNaN(isopach[i])) ? isopach[i] : 0.5;
            subbottom.push([pct, depth[i] + thick]);
        }
    }

    if (seafloor.length === 0) return;

    // sediment fill using custom series
    const sedimentSeries = [];
    if (sediment && sediment.length > 0) {
        const customData = [];
        for (let i = 0; i < depth.length; i++) {
            if (depth[i] === null || isNaN(depth[i])) continue;
            const thick = (isopach && isopach[i] !== null && !isNaN(isopach[i])) ? isopach[i] : 0.5;
            const cls = (sediment[i] !== null && !isNaN(sediment[i]) && sediment[i] >= 0) ? sediment[i] : -1;
            customData.push({
                pct: (i / (depth.length - 1)) * 100,
                depthTop: depth[i],
                depthBot: depth[i] + thick,
                classId: cls,
                index: i,
            });
        }

        sedimentSeries.push({
            type: 'custom',
            name: 'Sediment',
            data: customData,
            z: 1,
            silent: true,
            renderItem: function(params, api) {
                if (params.dataIndex >= customData.length - 1) return;
                const curr = customData[params.dataIndex];
                const next = customData[params.dataIndex + 1];
                if (curr.classId < 0) return;

                const color = SED_NATURAL_COLORS[curr.classId] || '#888';
                const x0 = api.coord([curr.pct, 0])[0];
                const x1 = api.coord([next.pct, 0])[0];
                const y0_top = api.coord([0, curr.depthTop])[1];
                const y0_bot = api.coord([0, curr.depthBot])[1];
                const y1_top = api.coord([0, next.depthTop])[1];
                const y1_bot = api.coord([0, next.depthBot])[1];

                return {
                    type: 'polygon',
                    shape: {
                        points: [[x0, y0_top], [x1, y1_top], [x1, y1_bot], [x0, y0_bot]]
                    },
                    style: { fill: color, opacity: 0.8 }
                };
            },
        });
    }

    // find used sediment classes for legend
    const usedClasses = new Set();
    if (sediment) sediment.forEach(s => { if (s !== null && !isNaN(s) && s >= 0) usedClasses.add(s); });

    chart.setOption({
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'cross' },
            formatter: function(params) {
                if (!params || params.length === 0) return '';
                const pctVal = params[0].data[0];
                let html = '';
                params.forEach(p => {
                    if (p.seriesName === 'Sediment') return;
                    html += `<div style="font-size:11px;"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${p.color};margin-right:4px;"></span>${p.seriesName}: ${p.data[1].toFixed(2)} m</div>`;
                });
                if (sediment) {
                    const idx = Math.round(pctVal / 100 * (depth.length - 1));
                    if (idx >= 0 && idx < sediment.length && sediment[idx] !== null && sediment[idx] >= 0) {
                        const cls = sediment[idx];
                        const color = SED_NATURAL_COLORS[cls] || '#888';
                        const label = SED_LABELS[cls] || `Class ${cls}`;
                        html += `<div style="font-size:11px;margin-top:2px;"><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${color};margin-right:4px;"></span>Sediment: ${label}</div>`;
                    }
                }
                return html;
            }
        },
        grid: { top: 30, bottom: 20, left: 40, right: 30 },
        xAxis: { type: 'value', min: 0, max: 100, splitLine: { show: false }, axisLabel: { show: false } },
        yAxis: { type: 'value', inverse: true, scale: true, axisLabel: { fontSize: 10 }, splitLine: { lineStyle: { type: 'dashed', color: '#e2e8f0' } } },
        series: [
            { name: 'Water', type: 'line', data: seafloor, symbol: 'none', lineStyle: { width: 0 }, areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(59,130,246,0.3)' }, { offset: 1, color: 'rgba(59,130,246,0.05)' }]) }, z: 0, silent: true },
            ...sedimentSeries,
            { name: 'Seafloor', type: 'line', data: seafloor, symbol: 'none', lineStyle: { color: '#2563eb', width: 2.5 }, z: 10 },
            { name: 'Isopach Base', type: 'line', data: subbottom, symbol: 'none', lineStyle: { color: '#dc2626', width: 1.5, type: 'dashed' }, z: 10 },
        ]
    });

    // sediment legend using graphic elements
    if (usedClasses.size > 0) {
        const items = [];
        let xPos = 45;
        usedClasses.forEach(cls => {
            const color = SED_NATURAL_COLORS[cls] || '#888';
            const label = SED_LABELS[cls] || `Class ${cls}`;
            items.push({ type: 'rect', left: xPos, top: 6, shape: { width: 10, height: 10, r: 2 }, style: { fill: color } });
            items.push({ type: 'text', left: xPos + 14, top: 6, style: { text: label, fill: '#64748b', fontSize: 9 } });
            xPos += 14 + label.length * 6.5 + 10;
        });
        chart.setOption({ graphic: { elements: items } });
    }

    window.addEventListener('resize', () => chart.resize());
    setTimeout(() => chart.resize(), 300);
}

// ── 💡 統一滑桿連動邏輯 ─────────────────────────────────────
function handleSlider(e, panelPrefix) {
    const pct = e.target.value / 100;
    
    // ECharts 游標連動 (使用 CSS Calc 完美對齊網格)
    const eCursor = document.getElementById(`${panelPrefix}-echarts-cursor`);
    if (eCursor && !eCursor.classList.contains('hidden')) {
        eCursor.style.left = `calc(40px + ${pct} * (100% - 70px))`;
    }

    // SSS 游標連動 (Right Panel)
    if (panelPrefix === 'rp') {
        document.querySelectorAll('.wf-redline').forEach(el => { const img = el.parentElement.querySelector('img'); if (img) el.style.top = (pct * img.clientHeight) + 'px'; });
    }

    // SBP 游標連動 (Bottom Panel)
    if (panelPrefix === 'bp') {
        const sbpCursor = document.getElementById('bp-sbp-cursor');
        const sbpImg = document.getElementById('bp-sbp-image');
        if (sbpCursor && sbpImg) sbpCursor.style.left = `${pct * sbpImg.clientWidth}px`;
    }

    // 地圖 Marker 與 資訊文字
    if (currentTrackCoords.length > 0) {
        const dataIndex = Math.min(Math.floor(pct * currentTrackCoords.length), currentTrackCoords.length - 1);
        const coord = currentTrackCoords[dataIndex];
        
        if (mapTrackMarker) mapTrackMarker.setLatLng([coord[1], coord[0]]);
        else mapTrackMarker = L.circleMarker([coord[1], coord[0]], { radius: 6, color: '#2563eb', fillColor: '#F57D15', fillOpacity: 1, weight: 2 }).addTo(map);
        
        let text = `📍 Ping: ${Math.floor(pct * currentWfPings)} / ${currentWfPings} | ${coord[1].toFixed(5)}°N, ${coord[0].toFixed(5)}°E`;
        const eContainer = document.getElementById(`${panelPrefix}-echarts-container`);
        if (eContainer && eContainer._depthData) {
            const depth = eContainer._depthData[dataIndex];
            if (depth !== null && !isNaN(depth)) text += ` | Depth: ${depth.toFixed(1)}m`;
        }
        const infoEl = document.getElementById(`${panelPrefix}-info-text`);
        if (infoEl) infoEl.textContent = text;
    }
}

document.getElementById('bp-slider')?.addEventListener('input', (e) => handleSlider(e, 'bp'));
document.getElementById('rp-slider')?.addEventListener('input', (e) => handleSlider(e, 'rp'));

// ── SSS 模式切換器 ──────────────────────────────────────────
window.setSSSMode = function(mode) {
    document.querySelectorAll('.sss-tab').forEach(b => { b.classList.remove('bg-white', 'shadow-sm', 'text-blue-600'); b.classList.add('text-slate-500'); });
    document.getElementById(`tab-${mode}`)?.classList.add('bg-white', 'shadow-sm', 'text-blue-600');
    document.getElementById(`tab-${mode}`)?.classList.remove('text-slate-500');

    const hf = document.getElementById('sss-hf-container');
    const lf = document.getElementById('sss-lf-container');
    const imagesDiv = document.getElementById('rp-images');

    if (mode === 'hf') { 
        if(hf) hf.style.display = 'block'; if(lf) lf.style.display = 'none'; 
        imagesDiv?.classList.add('flex-col'); imagesDiv?.classList.remove('flex-row'); 
        currentRightWidth = 450; 
    } else if (mode === 'lf') { 
        if(hf) hf.style.display = 'none'; if(lf) lf.style.display = 'block'; 
        imagesDiv?.classList.add('flex-col'); imagesDiv?.classList.remove('flex-row'); 
        currentRightWidth = 450; 
    } else { 
        if(hf) hf.style.display = 'block'; if(lf) lf.style.display = 'block'; 
        imagesDiv?.classList.remove('flex-col'); imagesDiv?.classList.add('flex-row'); 
        currentRightWidth = 800;
    }
    
    applyLayout();
    setTimeout(() => document.getElementById('rp-slider')?.dispatchEvent(new Event('input')), 350);
}

// ── 💡 瀑布圖與測線資料讀取 ────────────────────────────────
fetch(API + '/api/waterfall-index').then(r => r.json()).then(data => { waterfallIndex = data; }).catch(err => console.error(err));

function showWaterfallSidebar(feature) {
    if (!waterfallIndex) return;
    const props = feature.properties, filename = props.file;
    const isSSS = props.instrument === 'SSS';
    const isSBP = props.instrument === 'SBP';
    
    // 強制將原始稀疏座標，插值為 100 個均勻點，解決圖表尾巴消失問題！
    currentTrackCoords = interpolatePolyline(feature.geometry.coordinates, 100); 
    currentWfPings = props.pings || 100;
    
    if (isSSS) {
        openPanels('sss');
        const rpTitle = document.getElementById('rp-title');
        if(rpTitle) rpTitle.textContent = `SSS Viewer - ${filename}`;
        
        const rpCursor = document.getElementById('rp-echarts-cursor');
        if(rpCursor) rpCursor.classList.remove('hidden');

        const hfInfo = waterfallIndex.sss[`${filename}_HF`], lfInfo = waterfallIndex.sss[`${filename}_LF`];
        if (hfInfo) document.getElementById('img-hf').src = `/waterfalls/${hfInfo.image}`;
        if (lfInfo) document.getElementById('img-lf').src = `/waterfalls/${lfInfo.image}`;
        setSSSMode('hf'); 
        
        loadProfileData('rp', currentTrackCoords);
    } 
    else if (isSBP) {
        openPanels('sbp');
        const bpTitle = document.getElementById('bp-title');
        if(bpTitle) bpTitle.textContent = `SBP Viewer - ${filename}`;
        
        const bpCursor = document.getElementById('bp-echarts-cursor');
        if(bpCursor) bpCursor.classList.remove('hidden');

        const sbpInfo = waterfallIndex.sbp[filename];
        if (sbpInfo) {
            document.getElementById('bp-sbp-image').src = `/waterfalls/${sbpInfo.image}`;
            document.getElementById('bp-sbp-cursor')?.classList.remove('hidden');
        }
        
        loadProfileData('bp', currentTrackCoords);
    }
}

function loadProfileData(panelPrefix, coords) {
    const coordStr = coords.map(c => `${c[0]},${c[1]}`).join(';');
    const infoText = document.getElementById(`${panelPrefix}-info-text`);
    if (infoText) infoText.textContent = 'Loading profile data...';

    fetch(`${API}/api/profile?coords=${encodeURIComponent(coordStr)}`)
        .then(r => r.json())
        .then(data => {
            const container = document.getElementById(`${panelPrefix}-echarts-container`);
            if (container) {
                container._depthData = data.depth;
                renderProfileChart(`${panelPrefix}-echarts-container`, data.depth, data.isopach, data.sediment);
            }
            if (infoText) infoText.textContent = 'Drag slider to inspect.';
            document.getElementById(`${panelPrefix}-slider`)?.dispatchEvent(new Event('input'));
        });
}

// ── 💡 其他工具與互動邏輯 ──────────────────────────────────
document.querySelectorAll('.tool-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        if (btn.id === 'btn-toggle-sidebar' || btn.id === 'btn-reset') return;
        
        document.querySelectorAll('.tool-btn').forEach(b => { 
            if (b.id !== 'btn-reset' && b.id !== 'btn-toggle-sidebar') { 
                b.classList.remove('active', 'bg-blue-50', 'text-blue-600', 'shadow-sm'); 
                b.classList.add('text-slate-500'); 
            } 
        });
        btn.classList.add('active', 'bg-blue-50', 'text-blue-600', 'shadow-sm'); 
        btn.classList.remove('text-slate-500');
        currentTool = btn.dataset.tool;
        
        const mapEl = document.getElementById('map');
        if (mapEl) { 
            mapEl.classList.remove('cursor-pan', 'cursor-query', 'cursor-line', 'cursor-select'); 
            mapEl.classList.add(`cursor-${currentTool}`); 
        }
        
        currentTool === 'pan' ? map.dragging.enable() : map.dragging.disable();
        
        if (linePreview) { map.removeLayer(linePreview); linePreview = null; } lineStart = null;
        if (selectRect) { map.removeLayer(selectRect); selectRect = null; } selectStart = null;
    });
});

document.querySelectorAll('.layer-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const layer = btn.dataset.layer; if (!tileLayers[layer]) return;
        document.querySelectorAll('.layer-btn').forEach(b => { b.classList.remove('active', 'bg-blue-50', 'border-blue-600', 'text-blue-700', 'font-bold'); b.classList.add('text-slate-600', 'border-transparent', 'font-medium', 'hover:bg-slate-100'); });
        btn.classList.add('active', 'bg-blue-50', 'border-blue-600', 'text-blue-700', 'font-bold'); btn.classList.remove('text-slate-600', 'border-transparent', 'font-medium', 'hover:bg-slate-100');
        if (currentOverlay) map.removeLayer(currentOverlay);
        tileLayers[layer].setOpacity(document.getElementById('opacity-slider').value / 100);
        tileLayers[layer].addTo(map); currentOverlay = tileLayers[layer];

        if (layer === 'sediment_class') {
            if (!sedLegendAdded) { sedLegend.addTo(map); sedLegendAdded = true; }
        } else {
            if (sedLegendAdded) { sedLegend.remove(); sedLegendAdded = false; }
        }
    });
});
document.getElementById('opacity-slider')?.addEventListener('input', (e) => { if (currentOverlay) currentOverlay.setOpacity(e.target.value / 100); });

map.on('click', (e) => { if (currentTool === 'query') doPointQuery(e.latlng.lat, e.latlng.lng); });
map.on('mousedown', (e) => { if (currentTool === 'select') selectStart = e.latlng; else if (currentTool === 'line') lineStart = e.latlng; });
map.on('mousemove', (e) => {
    const cd = document.getElementById('coord-display'); if(cd) cd.textContent = `${e.latlng.lat.toFixed(6)}°N, ${e.latlng.lng.toFixed(6)}°E`;
    if (currentTool === 'select' && selectStart) {
        if (selectRect) map.removeLayer(selectRect);
        selectRect = L.rectangle([selectStart, e.latlng], { color: '#2563eb', weight: 2, fillOpacity: 0.15, dashArray: '5,5' }).addTo(map);
    }
    if (currentTool === 'line' && lineStart) {
        if (linePreview) map.removeLayer(linePreview);
        linePreview = L.polyline([lineStart, e.latlng], { color: '#F57D15', weight: 3, dashArray: '8,4' }).addTo(map);
    }
});
map.on('mouseup', (e) => {
    if (currentTool === 'select' && selectStart) {
        const bounds = L.latLngBounds(selectStart, e.latlng); selectStart = null;
        if (!bounds.getNorthEast().equals(bounds.getSouthWest())) doRegionSelect(bounds);
    }
    if (currentTool === 'line' && lineStart) {
        const endPoint = e.latlng; if (linePreview) map.removeLayer(linePreview); linePreview = null;
        if (map.distance(lineStart, endPoint) > 5) {
            if (drawnLine) map.removeLayer(drawnLine);
            drawnLine = L.polyline([lineStart, endPoint], { color: '#F57D15', weight: 3 }).addTo(map);
            
            openPanels('drawn-line');
            const bpTitle = document.getElementById('bp-title');
            if(bpTitle) bpTitle.textContent = '✏️ Hand-Drawn Profile';
            
            currentTrackCoords = interpolatePolyline([[lineStart.lng, lineStart.lat], [endPoint.lng, endPoint.lat]], 100);
            currentWfPings = 100;
            const d = map.distance(lineStart, endPoint);
            const infoText = document.getElementById('bp-info-text');
            if(infoText) infoText.textContent = `Length: ${d.toFixed(0)}m`;
            
            const coordStr2 = currentTrackCoords.map(c => `${c[0]},${c[1]}`).join(';');
            fetch(`${API}/api/profile?coords=${encodeURIComponent(coordStr2)}`)
                .then(r => r.json())
                .then(data => {
                    renderProfileChart('bp-echarts-container', data.depth, data.isopach, data.sediment);
                });
                    }
                    lineStart = null;
                }
});

function doPointQuery(lat, lon) {
    console.log(`=== 🔍 執行單點查詢: ${lat.toFixed(5)}, ${lon.toFixed(5)} ===`);
    
    if (clickMarker) {
        clickMarker.setLatLng([lat, lon]);
    } else {
        clickMarker = L.circleMarker([lat, lon], { 
            radius: 6, color: '#F57D15', fillColor: '#F57D15', fillOpacity: 0.8, weight: 2 
        }).addTo(map);
    }

    const loadingHtml = `
        <div class="w-56 p-1">
            <div class="font-mono text-[11px] text-blue-600 font-bold border-b border-slate-100 pb-1 mb-2">📍 ${lat.toFixed(5)}, ${lon.toFixed(5)}</div>
            <div class="text-xs text-slate-400 animate-pulse mt-2 flex items-center gap-1">
                <span>⏳ 擷取地層數據中...</span>
            </div>
        </div>
    `;
    
    // 將 popup 設為全域變數以便後續觸發 update()
    const popup = L.popup({ maxWidth: 300, autoClose: true, closeOnClick: true, autoPanPadding: [20, 20] })
        .setLatLng([lat, lon])
        .setContent(loadingHtml)
        .openOn(map);

    fetch(`${API}/api/query?lat=${lat}&lon=${lon}`)
        .then(r => r.json())
        .then(data => {
            console.log("📦 成功收到後端資料:", data);
            
            let html = `<div class="w-56 p-1">`;
            html += `<div class="font-mono text-[11px] text-blue-600 font-bold border-b border-slate-100 pb-1 mb-2">📍 ${lat.toFixed(5)}, ${lon.toFixed(5)}</div>`;
            
            if (data.error) {
                html += `<div class="text-red-500 text-xs font-bold py-2 text-center">⚠️ ${data.error}</div></div>`;
                popup.setContent(html);
                popup.update();
                return;
            }

            // --- 1. 核心指標提取 (水深、厚度、底質) ---
            const coreKeys = ['bathymetry', 'isopach', 'sediment_class'];
            let hasCoreData = false;
            let primaryHtml = '';

            coreKeys.forEach(key => {
                const info = data[key];
                if (info && info.value !== null) {
                    hasCoreData = true;
                    if (key === 'sediment_class') {
                        const classId = info.class_id !== undefined ? info.class_id : -1;
                        const color = (typeof SED_NATURAL_COLORS !== 'undefined' && SED_NATURAL_COLORS[classId]) ? SED_NATURAL_COLORS[classId] : '#888';
                        primaryHtml += `
                            <div class="text-[11px] flex justify-between py-1.5 items-center border-b border-slate-50">
                                <span class="font-bold text-slate-700">${info.name}:</span> 
                                <span class="text-[10px] text-white px-1.5 py-0.5 rounded shadow-sm" style="background:${color}">${info.value}</span>
                            </div>`;
                    } else {
                        let val = (typeof info.value === 'number' && info.value % 1 !== 0) ? info.value.toFixed(2) : info.value;
                        const unitHtml = info.units ? `<span class="text-[10px] text-slate-500 ml-1">${info.units}</span>` : '';
                        primaryHtml += `
                            <div class="text-[11px] flex justify-between py-1.5 border-b border-slate-50">
                                <span class="font-bold text-slate-700">${info.name}:</span> 
                                <span class="text-blue-600 font-bold">${val}${unitHtml}</span>
                            </div>`;
                    }
                }
            });

            // --- 2. 進階參數提取 (其他所有資料) ---
            let secondaryHtml = '';
            let hasSecondaryData = false;

            for (const [key, info] of Object.entries(data)) {
                if (['lat', 'lon', 'x_3826', 'y_3826'].includes(key) || coreKeys.includes(key) || !info || !info.name || info.value === null) continue;
                
                hasSecondaryData = true;
                let val = (typeof info.value === 'number' && info.value % 1 !== 0) ? info.value.toFixed(2) : info.value;
                const unitHtml = info.units ? `<span class="text-[10px] text-slate-500 ml-1">${info.units}</span>` : '';
                secondaryHtml += `
                    <div class="text-[10px] flex justify-between py-1 border-b border-slate-100 last:border-0">
                        <span class="text-slate-500">${info.name}:</span> 
                        <span class="text-slate-800 font-mono">${val}${unitHtml}</span>
                    </div>`;
            }

            // --- 3. 組裝 HTML 畫面 ---
            if (!hasCoreData && !hasSecondaryData) {
                html += `<div class="text-slate-400 text-[11px] py-4 text-center font-bold">此座標無地層資料</div>`;
            } else {
                html += primaryHtml;
                
                // 如果有進階參數，建立一個可摺疊的 <details> 區塊
                if (hasSecondaryData) {
                    // 💡 關鍵：加入 ontoggle 事件。當使用者展開時，延遲 10ms 呼叫 popup.update()，讓 Leaflet 重新計算高度並自動平移，防止破版！
                    html += `
                        <details class="mt-1 group" ontoggle="setTimeout(() => { if(window.map && window.map._popup) window.map._popup.update(); }, 50)">
                            <summary class="text-[10px] text-slate-400 cursor-pointer py-1.5 hover:text-blue-500 select-none outline-none font-bold flex items-center gap-1 transition-colors">
                                <span class="group-open:rotate-90 transition-transform text-[8px]">▶</span> 顯示進階探測參數
                            </summary>
                            <div class="pt-1 pb-2 bg-slate-50/50 rounded px-1.5 mt-1 border border-slate-100">
                                ${secondaryHtml}
                            </div>
                        </details>
                    `;
                }
            }
            
            html += `</div>`;
            
            popup.setContent(html);
            popup.update(); 
        })
        .catch(err => {
            console.error("❌ 解析資料失敗:", err);
            popup.setContent(`<div class="w-56 p-3 text-red-500 text-xs font-bold text-center">⚠️ 介面渲染失敗</div>`);
            popup.update();
        });
}

function doRegionSelect(bounds) {
    console.log("=== 🔍 執行框選，直接開啟 3D ===");
    
    // 1. 將 2D 地圖縮放至框選範圍
    // map.fitBounds(bounds, { padding: [50, 50] });

    const sw = bounds.getSouthWest(), ne = bounds.getNorthEast();

    // 2. 只做一件事：打 API 把 lat/lon 轉成 EPSG:3826
    Promise.all([
        fetch(`${API}/api/query?lat=${sw.lat}&lon=${sw.lng}`).then(r => r.json()),
        fetch(`${API}/api/query?lat=${ne.lat}&lon=${ne.lng}`).then(r => r.json())
    ])
    .then(([sw_d, ne_d]) => {
        if (sw_d.error || ne_d.error) {
            alert("座標轉換失敗，無法開啟 3D！");
            return;
        }

        const x0 = sw_d.x_3826, y0 = sw_d.y_3826;
        const x1 = ne_d.x_3826, y1 = ne_d.y_3826;

        console.log(`轉換完成，準備呼叫 3D 引擎: x0=${x0}, y0=${y0}, x1=${x1}, y1=${y1}`);

        // 3. 略過所有統計資料，直接彈出 3D 視窗！
        if (typeof window.build3DScene === 'function') {
            window.build3DScene(x0, y0, x1, y1);
        } else {
            alert("找不到 3D 渲染函數，請確認程式碼是否完整！");
        }
    })
    .catch(err => {
        console.error("❌ 捕捉到錯誤:", err);
        alert(`無法連線到伺服器: ${err.message}`);
    });
}

let currentRenderer = null;

// 綁定關閉按鈕 (確保 HTML 已經載入)
document.addEventListener('DOMContentLoaded', () => {
    const closeBtn = document.getElementById('btn-close-3d');
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            document.getElementById('modal-3d').classList.add('hidden');
        });
    }
});

// 全域函數，讓 HTML 內的 onclick 可以呼叫
window.build3DScene = function(x0, y0, x1, y1) {
    console.log(`3. 啟動 3D 引擎，請求範圍: ${x0}, ${y0} to ${x1}, ${y1}`);
    
    // 檢查是否有引入 Three.js
    if (typeof THREE === 'undefined') {
        alert("找不到 Three.js 套件！請確認 HTML 的 <head> 裡有加入 script 標籤。");
        return;
    }

    const modal = document.getElementById('modal-3d');
    const container = document.getElementById('canvas-container-3d');
    const loading = document.getElementById('loading-3d');
    
    if (!modal || !container || !loading) {
        console.error("找不到 HTML 中的 3D 容器 (modal-3d 等 id)");
        return;
    }

    modal.classList.remove('hidden');
    loading.classList.remove('hidden');

    if (currentRenderer) {
        container.innerHTML = '';
        currentRenderer.dispose();
    }

    fetch(`${API}/api/3d-scene?x0=${x0}&y0=${y0}&x1=${x1}&y1=${y1}`)
        .then(r => r.json())
        .then(data => {
            if (data.error) { 
                alert("後端回傳錯誤: " + data.error); 
                loading.classList.add('hidden');
                modal.classList.add('hidden');
                return; 
            }
            console.log("4. 成功取得 3D 資料，開始渲染...", data);
            initThreeJS(container, data);
            loading.classList.add('hidden');
        })
        .catch(err => {
            console.error("API /api/3d-scene 錯誤:", err);
            alert("載入 3D 失敗，請檢查後端 API 是否正常啟動");
            loading.classList.add('hidden');
            modal.classList.add('hidden');
        });
}


window.currentRenderer = null;
window.currentAnimationId = null; 
window.currentControls = null;

function initThreeJS(container, data) {
    const { width, height, step_m, bathymetry, bedrock, sss_texture, sediment_val } = data;

    container.style.position = 'relative';
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f172a); 

    const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 2000);
    camera.position.set(0, Math.max(width, height) * step_m * 0.8, Math.max(width, height) * step_m * 1.2);

    window.currentRenderer = new THREE.WebGLRenderer({ antialias: true });
    window.currentRenderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(window.currentRenderer.domElement);

    window.currentControls = new THREE.OrbitControls(camera, window.currentRenderer.domElement);
    window.currentControls.enableDamping = true;
    window.currentControls.autoRotate = false;

    // --- 1. 光源設定 ---
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(200, 300, 100);
    scene.add(dirLight);

    // --- 2. 核心參數與顏色映射 ---
    const Z_EXAGGERATION = 3.0; 
    let validBathymetry = bathymetry.filter(v => !isNaN(v));
    const minD = Math.min(...validBathymetry);
    const maxD = Math.max(...validBathymetry);
    const meanDepth = validBathymetry.reduce((a, b) => a + b, 0) / validBathymetry.length;

    // 🎨 精確 Turbo_r 演算法 (紅淺 -> 藍深)
    const getTurboR = (depth) => {
        let t = (depth - minD) / (maxD - minD || 1);
        t = 1 - Math.max(0, Math.min(1, t)); // 反轉
        const r = 0.1357 + 4.5974*t - 42.327*t*t + 130.588*Math.pow(t,3) - 150.566*Math.pow(t,4) + 58.137*Math.pow(t,5);
        const g = 0.0914 + 2.1941*t - 4.843*t*t + 14.185*Math.pow(t,3) - 32.171*Math.pow(t,4) + 28.533*Math.pow(t,5);
        const b = 0.1066 + 12.641*t - 60.582*t*t + 110.362*Math.pow(t,3) - 89.903*Math.pow(t,4) + 27.348*Math.pow(t,5);
        return new THREE.Color(Math.max(0,Math.min(1,r)), Math.max(0,Math.min(1,g)), Math.max(0,Math.min(1,b)));
    };

    // 🎨 動態抓取沉積層顏色 (完美對齊外部圖例)
    let sedHex = '#8a8578'; // 預設灰褐
    let sedLabel = 'Unknown';
    if (sediment_val !== undefined && sediment_val !== null) {
        let id = -1;
        if (typeof sediment_val === 'number') id = Math.round(sediment_val);
        else if (typeof sediment_val === 'string') id = SED_LABELS.findIndex(l => l === sediment_val);
        
        if (id >= 0 && id < SED_NATURAL_COLORS.length) {
            sedHex = SED_NATURAL_COLORS[id];
            sedLabel = SED_LABELS[id];
        }
    }
    const bedrockHex = '#1e293b'; // 深岩盤色

    // --- 3. 建立 3D 紋理 ---
    let sssTexture = null;
    if (sss_texture) {
        const texData = new Uint8Array(width * height * 4);
        for (let i = 0; i < sss_texture.length; i++) {
            const v = sss_texture[i];
            const contrast = Math.pow(v / 255, 1.2) * 255; 
            texData[i*4] = contrast; texData[i*4+1] = contrast*0.75; texData[i*4+2] = contrast*0.2; texData[i*4+3] = 255;
        }
        sssTexture = new THREE.DataTexture(texData, width, height, THREE.RGBAFormat);
        sssTexture.needsUpdate = true;
    }

    const bathyTexData = new Uint8Array(width * height * 4);
    for (let i = 0; i < bathymetry.length; i++) {
        const c = getTurboR(bathymetry[i]);
        bathyTexData[i*4] = c.r*255; bathyTexData[i*4+1] = c.g*255; bathyTexData[i*4+2] = c.b*255; bathyTexData[i*4+3] = 255;
    }
    const bathyTexture = new THREE.DataTexture(bathyTexData, width, height, THREE.RGBAFormat);
    bathyTexture.needsUpdate = true;

    // --- 4. 計算空間座標 ---
    const posS = new Float32Array(bathymetry.length * 3);
    const posB = new Float32Array(bathymetry.length * 3);
    const posBase = new Float32Array(bathymetry.length * 3);

    let minBedrockZ = Infinity;
    for (let i = 0; i < bathymetry.length; i++) {
        const x = (i % width) * step_m;
        const y = Math.floor(i / width) * step_m;
        const zS = -(bathymetry[i] - meanDepth) * Z_EXAGGERATION;
        const zB = bedrock ? -(bedrock[i] - meanDepth) * Z_EXAGGERATION : zS - 2;

        if (zB < minBedrockZ) minBedrockZ = zB;
        posS[i*3] = x; posS[i*3+1] = zS; posS[i*3+2] = y;
        posB[i*3] = x; posB[i*3+1] = zB; posB[i*3+2] = y;
    }

    const baseZ = minBedrockZ - 10;
    for (let i = 0; i < bathymetry.length; i++) {
        posBase[i*3] = posS[i*3]; posBase[i*3+1] = baseZ; posBase[i*3+2] = posS[i*3+2];
    }

    // --- 5. 建模與填滿側邊 ---
    const matSurface = new THREE.MeshStandardMaterial({ map: bathyTexture, flatShading: true, side: THREE.DoubleSide });
    const matSediment = new THREE.MeshStandardMaterial({ color: new THREE.Color(sedHex), flatShading: true, side: THREE.DoubleSide });
    const matBedrock = new THREE.MeshStandardMaterial({ color: new THREE.Color(bedrockHex), flatShading: true, side: THREE.DoubleSide });

    const createPlane = (posArray) => {
        const geom = new THREE.PlaneGeometry(width * step_m, height * step_m, width - 1, height - 1);
        geom.rotateX(-Math.PI / 2);
        geom.translate((width * step_m)/2, 0, (height * step_m)/2);
        geom.attributes.position.array.set(posArray);
        geom.computeVertexNormals();
        return geom;
    };

    const group = new THREE.Group();
    group.position.set(-(width * step_m)/2, 0, -(height * step_m)/2);
    scene.add(group);
    group.add(new THREE.Mesh(createPlane(posS), matSurface));
    group.add(new THREE.Mesh(createPlane(posB), matBedrock));

    const createWall = (tArr, bArr, count, indices, mat) => {
        const wallGeom = new THREE.PlaneGeometry(1, 1, count - 1, 1);
        const wallPos = wallGeom.attributes.position.array;
        for (let i = 0; i < count; i++) {
            const idx = indices[i];
            wallPos[i*3] = tArr[idx*3]; wallPos[i*3+1] = tArr[idx*3+1]; wallPos[i*3+2] = tArr[idx*3+2];
            wallPos[(i+count)*3] = bArr[idx*3]; wallPos[(i+count)*3+1] = bArr[idx*3+1]; wallPos[(i+count)*3+2] = bArr[idx*3+2];
        }
        wallGeom.computeVertexNormals();
        return new THREE.Mesh(wallGeom, mat);
    };

    const idxSouth = Array.from({length: width}, (_, i) => i);
    const idxNorth = Array.from({length: width}, (_, i) => (height-1)*width + i);
    const idxWest = Array.from({length: height}, (_, i) => i*width);
    const idxEast = Array.from({length: height}, (_, i) => i*width + width - 1);

    group.add(createWall(posS, posB, width, idxSouth, matSediment));
    group.add(createWall(posS, posB, width, idxNorth, matSediment));
    group.add(createWall(posS, posB, height, idxWest, matSediment));
    group.add(createWall(posS, posB, height, idxEast, matSediment));

    group.add(createWall(posB, posBase, width, idxSouth, matBedrock));
    group.add(createWall(posB, posBase, width, idxNorth, matBedrock));
    group.add(createWall(posB, posBase, height, idxWest, matBedrock));
    group.add(createWall(posB, posBase, height, idxEast, matBedrock));

    // --- 6. 加上 Bounding Box (3D 高度尺與邊界輔助線) ---
    const boxHelper = new THREE.BoxHelper(group, 0x475569);
    scene.add(boxHelper);

    // ==========================================
    // 🛠️ 7. 建立 3D 左上角資料面板 (高度尺與材質)
    // ==========================================
    const infoOverlay = document.createElement('div');
    // 統一設定：top-4, w-64, h-[200px], flex flex-col
    infoOverlay.className = "absolute top-16 left-4 bg-slate-900/80 border border-slate-600 rounded-lg p-4 z-10 shadow-2xl text-slate-200 text-xs backdrop-blur-md pointer-events-none w-64 h-[220px] flex flex-col justify-between";
    infoOverlay.innerHTML = `
        <div class="font-bold text-blue-400 border-b border-slate-600 pb-2 mb-2 shrink-0">📊 Section Metrics</div>
        
        <div class="flex-1 flex flex-col justify-around">
            <div class="flex justify-between gap-2"><span class="text-slate-400">Length x Width:</span> <span class="font-mono">${(width*step_m).toFixed(1)} x ${(height*step_m).toFixed(1)} m</span></div>
            <div class="flex justify-between gap-2"><span class="text-slate-400">Depth Range:</span> <span class="font-mono">${minD.toFixed(1)} ~ ${maxD.toFixed(1)} m</span></div>
            <div class="flex justify-between gap-2"><span class="text-slate-400">Surf. Material:</span> <span class="font-mono flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-sm inline-block shadow-sm" style="background:${sedHex}"></span>${sedLabel}</span></div>
            <div class="flex justify-between gap-2"><span class="text-slate-400">Base Material:</span> <span class="font-mono flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-sm inline-block bg-slate-800 shadow-sm border border-slate-500"></span>Weathered Rock</span></div>
        </div>
    `;
    container.appendChild(infoOverlay);

    // ==========================================
    // 🛠️ 8. 右上角控制面板 (已移除透明度滑桿)
    // ==========================================
    const uiOverlay = document.createElement('div');
    uiOverlay.className = "absolute top-16 right-4 bg-slate-900/90 border border-slate-600 rounded-lg p-4 z-10 shadow-2xl w-56 text-slate-200 text-sm backdrop-blur-md";
    uiOverlay.innerHTML = `
        <h4 class="font-bold mb-3 border-b border-slate-600 pb-1 text-blue-400">🔍 圖層控制器</h4>
        <label class="flex items-center gap-2 cursor-pointer hover:text-white transition-colors">
            <input type="checkbox" id="ctrl-sss" class="w-4 h-4 rounded accent-blue-600">
            <span>SSS 聲納影像</span>
        </label>
    `;
    container.appendChild(uiOverlay);

    // 僅保留 SSS 貼圖的切換邏輯
    document.getElementById('ctrl-sss').addEventListener('change', (e) => {
        matSurface.map = e.target.checked ? (sssTexture || bathyTexture) : bathyTexture;
        matSurface.needsUpdate = true;
    });

    function animate() {
        window.currentAnimationId = requestAnimationFrame(animate);
        window.currentControls.update(); 
        window.currentRenderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', () => {
        const modal = document.getElementById('modal-3d');
        if (modal && !modal.classList.contains('hidden')) {
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            if (window.currentRenderer) window.currentRenderer.setSize(container.clientWidth, container.clientHeight);
        }
    });
}

// 定義關閉 3D 視窗的統一函數
window.close3D = function() {
    const modal = document.getElementById('modal-3d');
    if (!modal) return;
    
    // 1. 隱藏視窗與解除鍵盤綁定
    modal.classList.add('hidden');
    document.removeEventListener('keydown', onEscKeyDown);
    
    // 2. 🛑 停止「殭屍動畫迴圈」(最重要的一步！)
    if (window.currentAnimationId !== null) {
        cancelAnimationFrame(window.currentAnimationId);
        window.currentAnimationId = null;
    }
    
    // 3. 🧹 解除滑鼠控制器的事件綁定 (釋放滑鼠控制權)
    if (window.currentControls) {
        window.currentControls.dispose();
        window.currentControls = null;
    }
    
    // 4. 🔥 銷毀渲染器與拔除畫布
    if (window.currentRenderer) {
        window.currentRenderer.dispose();
        window.currentRenderer = null;
    }
    
    const container = document.getElementById('canvas-container-3d');
    if (container) {
        container.innerHTML = ''; // 徹底清空舊的 Canvas
    }
};

// 處理 Esc 鍵
function onEscKeyDown(e) {
    if (e.key === 'Escape') window.close3D();
}

window.build3DScene = function(x0, y0, x1, y1) {
    console.log(`啟動 3D 引擎，請求範圍: ${x0}, ${y0} to ${x1}, ${y1}`);
    if (typeof THREE === 'undefined') { alert("找不到 Three.js！"); return; }

    const modal = document.getElementById('modal-3d');
    const container = document.getElementById('canvas-container-3d');
    const loading = document.getElementById('loading-3d');
    if (!modal) return;

    modal.classList.remove('hidden');
    loading.classList.remove('hidden');
    modal.onclick = (e) => { if (e.target === modal) window.close3D(); };
    document.addEventListener('keydown', onEscKeyDown);

    if (window.currentRenderer) {
        container.innerHTML = '';
        window.currentRenderer.dispose();
    }

    Promise.all([
        fetch(`${API}/api/3d-scene?x0=${x0}&y0=${y0}&x1=${x1}&y1=${y1}`).then(r => r.json()),
        fetch(`${API}/api/stats?x0=${x0}&y0=${y0}&x1=${x1}&y1=${y1}`).then(r => r.json())
    ])
    .then(([sceneData, statsData]) => {
        if (sceneData.error) { window.close3D(); alert(sceneData.error); return; }

        // 💡 提取底質分類，傳遞給 3D 引擎以對齊圖例顏色
        if (statsData.layers && statsData.layers.sediment_class) {
            sceneData.sediment_val = statsData.layers.sediment_class.dominant || statsData.layers.sediment_class.mean;
        }

        initThreeJS(container, sceneData);
        loading.classList.add('hidden');
    })
    .catch(err => {
        console.error("API 錯誤:", err);
        alert("載入 3D 失敗，請檢查網路狀態");
        window.close3D();
    });
};

fetch(API + '/api/tracklines').then(r => r.json()).then(geojson => {
    const commonOptions = {
        style: (feature) => (feature.properties.instrument === 'SSS') ? { color: '#ffffff', weight: 3, opacity: 0.9, filter: 'drop-shadow(0 0 2px #000)' } : { color: '#be185d', weight: 3, opacity: 0.9, dashArray: '4, 4' },
        onEachFeature: (feature, layer) => {
            layer.on('click', (e) => { L.DomEvent.stopPropagation(e); selectTrackline(feature, layer, feature.properties.instrument === 'SSS' ? sssLayer : sbpLayer); });
            layer.on('mouseover', () => { if (layer !== selectedTrackline) { layer.setStyle({ weight: 6, opacity: 1 }); layer.bringToFront(); }});
            layer.on('mouseout', () => { if (layer !== selectedTrackline) { (feature.properties.instrument === 'SSS' ? sssLayer : sbpLayer).resetStyle(layer); }});
        }
    };
    sssLayer = L.geoJSON(geojson, { ...commonOptions, filter: (f) => f.properties.instrument === 'SSS' });
    sbpLayer = L.geoJSON(geojson, { ...commonOptions, filter: (f) => f.properties.instrument === 'SBP' });
});

function selectTrackline(feature, layer, parentLayer) {
    if (selectedTrackline && selectedParentLayer) selectedParentLayer.resetStyle(selectedTrackline);
    selectedTrackline = layer; selectedParentLayer = parentLayer;
    layer.setStyle({ weight: 6, opacity: 1, color: '#F57D15', dashArray: '' }); layer.bringToFront();
    showWaterfallSidebar(feature); 
    map.panTo(layer.getBounds().getCenter(), {animate: true});
}

document.getElementById('chk-trackline-sss')?.addEventListener('change', (e) => { if (e.target.checked && sssLayer) sssLayer.addTo(map); else if (sssLayer) map.removeLayer(sssLayer); });
document.getElementById('chk-trackline-sbp')?.addEventListener('change', (e) => { if (e.target.checked && sbpLayer) sbpLayer.addTo(map); else if (sbpLayer) map.removeLayer(sbpLayer); });
