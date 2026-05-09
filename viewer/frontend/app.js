import { API, SEDIMENT_COLORS, SED_LABELS, INITIAL_CENTER, INITIAL_ZOOM } from './src/constants.js';
import { state } from './src/state.js';
import { interpolatePolyline } from './src/utils.js';
import { initMap, loadTileLayers, bindMapUI } from './src/modules/map.js';
import { applyLayout, openPanels, closePanels, resizeCanvases, bindLayoutUI } from './src/modules/layout.js';
import { doPointQuery } from './src/modules/popup.js';

const map = initMap();
loadTileLayers(map);
bindMapUI(map);
bindLayoutUI();


// 🔧 工具列狀態管理 (單選互斥與自動清場)
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
        window.closePanels();
        if (typeof window.close3D === 'function') window.close3D();
        if (typeof window.closeBorehole === 'function') window.closeBorehole();
        
        if (state.linePreview) { map.removeLayer(state.linePreview); state.linePreview = null; } state.lineStart = null;
        if (state.selectRect) { map.removeLayer(state.selectRect); state.selectRect = null; } state.selectStart = null;
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

window.resetMapState = function() {
    window.closePanels(); 
    map.closePopup();
    if (typeof window.close3D === 'function') window.close3D();
    if (typeof window.closeBorehole === 'function') window.closeBorehole();
    
    if (state.clickMarker) { map.removeLayer(state.clickMarker); state.clickMarker = null; }
    if (state.selectRect) { map.removeLayer(state.selectRect); state.selectRect = null; }
    if (state.linePreview) { map.removeLayer(state.linePreview); state.linePreview = null; }
    if (state.drawnLine) { map.removeLayer(state.drawnLine); state.drawnLine = null; }
    
    map.setView(INITIAL_CENTER, INITIAL_ZOOM);
    document.getElementById('btn-tool-pan')?.click();
}
document.getElementById('btn-reset')?.addEventListener('click', resetMapState);

// ── 5. 地圖滑鼠事件 (Map Interactions) ─────────────────────
map.on('click', (e) => { if (state.currentTool === 'query') doPointQuery(e.latlng.lat, e.latlng.lng); });
map.on('mousedown', (e) => { if (state.currentTool === 'select') state.selectStart = e.latlng; else if (state.currentTool === 'line') state.lineStart = e.latlng; });
map.on('mousemove', (e) => {
    const cd = document.getElementById('coord-display'); if(cd) cd.textContent = `${e.latlng.lat.toFixed(6)}°N, ${e.latlng.lng.toFixed(6)}°E`;
    if (state.currentTool === 'select' && state.selectStart) {
        if (state.selectRect) map.removeLayer(state.selectRect);
        state.selectRect = L.rectangle([state.selectStart, e.latlng], { color: '#2563eb', weight: 2, fillOpacity: 0.15, dashArray: '5,5' }).addTo(map);
    }
    if (state.currentTool === 'line' && state.lineStart) {
        if (state.linePreview) map.removeLayer(state.linePreview);
        state.linePreview = L.polyline([state.lineStart, e.latlng], { color: '#F57D15', weight: 3, dashArray: '8,4' }).addTo(map);
    }
});
map.on('mouseup', (e) => {
    if (state.currentTool === 'select' && state.selectStart) {
        const bounds = L.latLngBounds(state.selectStart, e.latlng); state.selectStart = null;
        if (!bounds.getNorthEast().equals(bounds.getSouthWest())) doRegionSelect(bounds);
    }
    if (state.currentTool === 'line' && state.lineStart) {
        const endPoint = e.latlng; if (state.linePreview) map.removeLayer(state.linePreview); state.linePreview = null;
        if (map.distance(state.lineStart, endPoint) > 5) {
            if (state.drawnLine) map.removeLayer(state.drawnLine);
            state.drawnLine = L.polyline([state.lineStart, endPoint], { color: '#F57D15', weight: 3 }).addTo(map);
            
            openPanels('drawn-line');
            const bpTitle = document.getElementById('bp-title');
            if(bpTitle) bpTitle.textContent = '✏️ Hand-Drawn Profile';
            
            state.currentTrackCoords = interpolatePolyline(map, [[state.lineStart.lng, state.lineStart.lat], [endPoint.lng, endPoint.lat]], 100);
            state.currentWfPings = 100;
            const d = map.distance(state.lineStart, endPoint);
            const infoText = document.getElementById('bp-info-text');
            if(infoText) infoText.textContent = `Length: ${d.toFixed(0)}m`;
            
            const coordStr2 = state.currentTrackCoords.map(c => `${c[0]},${c[1]}`).join(';');
            fetch(`${API}/api/profile?coords=${encodeURIComponent(coordStr2)}`)
                .then(r => r.json())
                .then(data => { renderProfileChart('bp-echarts-container', data.depth, data.isopach, data.sediment); });
        }
        state.lineStart = null;
    }
});


function doRegionSelect(bounds) {
    const sw = bounds.getSouthWest(), ne = bounds.getNorthEast();
    Promise.all([
        fetch(`${API}/api/query?lat=${sw.lat}&lon=${sw.lng}`).then(r => r.json()),
        fetch(`${API}/api/query?lat=${ne.lat}&lon=${ne.lng}`).then(r => r.json())
    ]).then(([sw_d, ne_d]) => {
        if (sw_d.error || ne_d.error) return alert("座標轉換失敗！");
        if (typeof window.build3DScene === 'function') window.build3DScene(sw_d.x_3826, sw_d.y_3826, ne_d.x_3826, ne_d.y_3826);
    }).catch(err => alert(`API 錯誤: ${err.message}`));
}

fetch(API + '/api/waterfall-index').then(r => r.json()).then(data => { 
    state.waterfallIndex = data;
    window.waterfallIndex = data;
});

function showWaterfallSidebar(feature) {
    if (!state.waterfallIndex) return;
    const props = feature.properties, filename = props.file;
    state.currentTrackCoords = interpolatePolyline(map, feature.geometry.coordinates, 100);
    state.currentWfPings = props.pings || 100;
    
    if (props.instrument === 'SSS') {
        openPanels('sss');
        document.getElementById('rp-title').textContent = `SSS Viewer - ${filename || 'Unknown'}`;
        document.getElementById('rp-echarts-cursor')?.classList.remove('hidden');
        // Sidebar shows LF preview (overview-friendly); HF detail is in modal
        const lfEntry = state.waterfallIndex.sss[`${filename}_LF`];
        if (lfEntry) document.getElementById('img-preview').src = `/waterfalls/${lfEntry.image}`;
        // Wire up the "放大" button to open the modal
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
        if (container) { container._depthData = data.depth; renderProfileChart(`${panelPrefix}-echarts-container`, data.depth, data.isopach, data.sediment); }
        document.getElementById(`${panelPrefix}-slider`)?.dispatchEvent(new Event('input'));
    });
}

function handleSlider(e, panelPrefix) {
    const pct = e.target.value / 100;
    const eContainer = document.getElementById(`${panelPrefix}-echarts-container`);

    if (eContainer && eContainer._chart) {
        const chart = eContainer._chart;
        
        // 算出目前滑桿對應的「真實螢幕 X 像素」
        const xPixel = chart.convertToPixel({ xAxisIndex: 0 }, pct * 100);
        
        if (xPixel != null && !isNaN(xPixel)) {
            // 按下遙控器：通知圖表「在此像素位置顯示 Tooltip 與十字線」
            chart.dispatchAction({
                type: 'showTip',
                x: xPixel,                     // X 軸就是我們剛算好的像素
                y: chart.getHeight() / 2       // Y 軸隨便設在圖表正中間即可，它會自動對齊資料點
            });
        }
    }

    // ==========================================
    // 2. 處理右側 / 下方面板的影像同步線 (維持不變)
    // ==========================================
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

    // ==========================================
    // 3. 處理地圖上的 2D 軌跡雷達動畫與文字標籤 (維持不變)
    // ==========================================
    if (state.currentTrackCoords.length > 0) {
        const dataIndex = Math.min(Math.floor(pct * state.currentTrackCoords.length), state.currentTrackCoords.length - 1);
        const coord = state.currentTrackCoords[dataIndex];
        
        // 更新地圖上的橘色雷達波紋
        if (state.mapTrackMarker) {
            state.mapTrackMarker.setLatLng([coord[1], coord[0]]);
        } else {
            const sonarIcon = L.divIcon({ className: 'sonar-ping-marker', iconSize: [12, 12], iconAnchor: [6, 6] });
            state.mapTrackMarker = L.marker([coord[1], coord[0]], { icon: sonarIcon }).addTo(map);
        }
        
        // 更新文字面板資訊
        // let text = `📍 ${coord[1].toFixed(5)}°N, ${coord[0].toFixed(5)}°E`;
        // if (eContainer && eContainer._depthData && eContainer._depthData[dataIndex] !== null && !isNaN(eContainer._depthData[dataIndex])) {
        //     text += ` | Depth: ${eContainer._depthData[dataIndex].toFixed(1)}m`;
        // }
        // const infoTextNode = document.getElementById(`${panelPrefix}-info-text`);
        // if(infoTextNode) infoTextNode.textContent = text;
    }
}
document.getElementById('bp-slider')?.addEventListener('input', (e) => handleSlider(e, 'bp'));
document.getElementById('rp-slider')?.addEventListener('input', (e) => handleSlider(e, 'rp'));

// ── 8. ECharts 繪製引擎 ────────────────────────────────────
function renderProfileChart(containerId, depth, isopach, sediment) {
    const container = document.getElementById(containerId);
    if (!container) return;
    if (container._chart) container._chart.dispose();
    const chart = echarts.init(container);
    container._chart = chart;

    // Detect if isopach is actually available (might be null even if state.HAS_ISOPACH)
    const showIsopach = state.HAS_ISOPACH && isopach && isopach.some(v => v !== null && !isNaN(v));

    const seafloor = [], subbottom = [];
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

    // Sediment polygon series — only if isopach gives the band thickness
    const sedimentSeries = [];
    if (showIsopach && sediment && sediment.length > 0) {
        const customData = [];
        for (let i = 0; i < depth.length; i++) {
            if (depth[i] === null || isNaN(depth[i])) continue;
            const thick = (isopach[i] !== null && !isNaN(isopach[i])) ? isopach[i] : 0;
            if (thick <= 0) continue;
            const cls = (sediment[i] !== null && !isNaN(sediment[i]) && sediment[i] >= 0) ? sediment[i] : -1;
            customData.push({
                pct: (i / (depth.length - 1)) * 100,
                depthTop: depth[i],
                depthBot: depth[i] + thick,
                classId: cls
            });
        }
        if (customData.length > 1) {
            sedimentSeries.push({
                type: 'custom', name: 'Sediment', data: customData, z: 2, silent: true,
                renderItem: function(params, api) {
                    if (params.dataIndex >= customData.length - 1) return;
                    const curr = customData[params.dataIndex], next = customData[params.dataIndex + 1];
                    const color = (curr.classId >= 0 && SEDIMENT_COLORS[curr.classId])
                                  ? SEDIMENT_COLORS[curr.classId] : '#d1d5db';
                    return {
                        type: 'polygon',
                        shape: { points: [
                            api.coord([curr.pct, curr.depthTop]),
                            api.coord([next.pct, next.depthTop]),
                            api.coord([next.pct, next.depthBot]),
                            api.coord([curr.pct, curr.depthBot])
                        ]},
                        style: { fill: color, opacity: 1 }
                    };
                }
            });
        }
    }

    // Build series array conditionally
    const series = [
        // Water area: always show
        {
            name: 'Water', type: 'line', data: seafloor,
            symbol: 'none', lineStyle: { width: 0 },
            areaStyle: { color: 'rgba(59,130,246,0.15)', origin: 'start' },
            z: 0, silent: true
        },
    ];

    // Bedrock + sediment + isopach base — only with isopach
    if (showIsopach) {
        series.push({
            name: 'Bedrock Fill', type: 'line', data: subbottom,
            symbol: 'none', lineStyle: { width: 0 },
            areaStyle: { color: '#64748b', opacity: 0.15, origin: 'end' },
            z: 1, silent: true
        });
        series.push(...sedimentSeries);
        series.push({
            name: 'Isopach Base', type: 'line', data: subbottom,
            symbol: 'none',
            lineStyle: { color: '#64748b', width: 1.5, type: 'dashed' },
            z: 11
        });
    }

    // Seafloor line: always last so it draws on top
    series.push({
        name: 'Seafloor', type: 'line', data: seafloor,
        symbol: 'none',
        lineStyle: { color: '#0f172a', width: 1.5 },
        z: 10
    });

    chart.setOption({
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'axis', showContent: false,
            axisPointer: {
                type: 'cross',
                lineStyle: { color: '#FF5722', width: 1, type: 'dashed' },
                crossStyle: { color: '#FF5722', width: 1, type: 'dashed' }
            }
        },
        grid: { top: 30, bottom: 20, left: 40, right: 30 },
        xAxis: { type: 'value', min: 0, max: 100, splitLine: { show: false }, axisLabel: { show: false } },
        yAxis: { type: 'value', inverse: true, scale: true, axisLabel: { fontSize: 10 }, splitLine: { lineStyle: { type: 'dashed', color: '#e2e8f0' } } },
        series: series
    });

    window.addEventListener('resize', () => chart.resize());
    setTimeout(() => chart.resize(), 350);
}

// ── 9. GeoJSON 軌跡線載入 ─────────────────────────────────
fetch(API + '/api/tracklines').then(r => r.json()).then(geojson => {
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
                selectTrackline(feature, layer, feature.properties.instrument === 'SSS' ? state.sssLayer : state.sbpLayer); 
            });
            layer.on('mouseover', () => { 
                if (layer !== state.selectedTrackline) { layer.setStyle({ weight: 5, opacity: 1, color: '#38bdf8' }); layer.bringToFront(); }
            });
            layer.on('mouseout', () => { 
                if (layer !== state.selectedTrackline) { (feature.properties.instrument === 'SSS' ? state.sssLayer : state.sbpLayer).resetStyle(layer); }
            });
        }
    };
    state.sssLayer = L.geoJSON(geojson, { ...commonOptions, filter: (f) => f.properties.instrument === 'SSS' });
    state.sbpLayer = L.geoJSON(geojson, { ...commonOptions, filter: (f) => f.properties.instrument === 'SBP' });
});

function selectTrackline(feature, layer, parentLayer) {
    if (state.selectedTrackline && state.selectedParentLayer) state.selectedParentLayer.resetStyle(state.selectedTrackline);
    state.selectedTrackline = layer; state.selectedParentLayer = parentLayer;
    layer.setStyle({ weight: 6, opacity: 1, color: '#F57D15', dashArray: '', filter: 'drop-shadow(0px 0px 4px rgba(245,125,21,0.8))' }); 
    layer.bringToFront();
    showWaterfallSidebar(feature); 
    map.panTo(layer.getBounds().getCenter(), {animate: true});
}

document.getElementById('chk-trackline-sss')?.addEventListener('change', (e) => { if (e.target.checked && state.sssLayer) state.sssLayer.addTo(map); else if (state.sssLayer) map.removeLayer(state.sssLayer); });
document.getElementById('chk-trackline-sbp')?.addEventListener('change', (e) => { if (e.target.checked && state.sbpLayer) state.sbpLayer.addTo(map); else if (state.sbpLayer) map.removeLayer(state.sbpLayer); });


// ── MAG candidates layer ──
// Insert this block in app.js right after the existing
// fetch(API + '/api/tracklines')...then(...) block that creates
// state.sssLayer and state.sbpLayer.

let magTargetsLayer = null;

fetch(API + '/api/mag-targets').then(r => r.json()).then(geojson => {
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

document.getElementById('chk-mag-targets')?.addEventListener('change', (e) => {
    if (!magTargetsLayer) return;
    if (e.target.checked) magTargetsLayer.addTo(map);
    else map.removeLayer(magTargetsLayer);
});

// ── 11. Three.js 虛擬岩心 (Virtual Borehole) ────────────────
window.closeBorehole = function() {
    const modal = document.getElementById('modal-borehole');
    if (modal) modal.classList.add('hidden');
    if (window.currentBoreholeAnimationId) { cancelAnimationFrame(window.currentBoreholeAnimationId); window.currentBoreholeAnimationId = null; }
    if (window.currentBoreholeControls) { window.currentBoreholeControls.dispose(); window.currentBoreholeControls = null; }
    if (window.currentBoreholeRenderer) { window.currentBoreholeRenderer.dispose(); window.currentBoreholeRenderer = null; }
    const container = document.getElementById('canvas-container-borehole');
    if (container) container.innerHTML = '';
};

window.buildBoreholeScene = function(lat, lon, title = "虛擬岩心探測") {
    if (typeof THREE === 'undefined') return alert("找不到 Three.js！");
    const modal = document.getElementById('modal-borehole');
    const container = document.getElementById('canvas-container-borehole');
    const loading = document.getElementById('loading-borehole');
    if (!modal) return;

    document.getElementById('borehole-title').innerHTML = `🕳️ ${title}`;
    modal.classList.remove('hidden'); loading.classList.remove('hidden');
    modal.onclick = (e) => { if (e.target === modal) window.closeBorehole(); };

    if (window.currentBoreholeRenderer) { container.innerHTML = ''; window.currentBoreholeRenderer.dispose(); }

    fetch(`${API}/api/query?lat=${lat}&lon=${lon}`).then(r => r.json()).then(data => {
        if (data.error) { window.closeBorehole(); return alert(data.error); }
        const depth = data.bathymetry?.value || 0;
        const isopach = state.HAS_ISOPACH ? (data.isopach?.value ?? null) : null;
        const sedClassId = data.sediment_class?.class_id ?? -1;
        const sedName = data.sediment_class?.value || 'Unknown';

        initBorehole3D(container, { depth, isopach, sedClassId, sedName });
        loading.classList.add('hidden');
    }).catch(err => { console.error(err); alert("載入岩心失敗"); window.closeBorehole(); });
};


function initBorehole3D(container, data) {
    const { depth, isopach, sedClassId, sedName } = data;
    const Z_EXAGGERATION = 5.0;
    const hasSediment = (isopach !== null && isopach > 0);

    container.style.position = 'relative';
    const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0a0f1a);
    const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(20, 10, 25);

    window.currentBoreholeRenderer = new THREE.WebGLRenderer({ antialias: true });
    window.currentBoreholeRenderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(window.currentBoreholeRenderer.domElement);

    window.currentBoreholeControls = new THREE.OrbitControls(camera, window.currentBoreholeRenderer.domElement);
    window.currentBoreholeControls.enableDamping = true;
    window.currentBoreholeControls.target.set(0, -5, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8); dirLight.position.set(10, 20, 10); scene.add(dirLight);

    const radius = 4;
    const actualIsopach = hasSediment ? isopach * Z_EXAGGERATION : 0;
    const bedrockVisualHeight = 15, waterVisualHeight = 8;
    const sedHex = (sedClassId >= 0 && sedClassId < SEDIMENT_COLORS.length) ? SEDIMENT_COLORS[sedClassId] : '#8a8578';

    const matWater = new THREE.MeshPhysicalMaterial({ color: 0x3b82f6, transparent: true, opacity: 0.15, roughness: 0.1 });
    const matBedrock = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.9, flatShading: true });

    const group = new THREE.Group(); scene.add(group);

    // Water column always
    const meshWater = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, waterVisualHeight, 64), matWater);
    meshWater.position.y = waterVisualHeight / 2; group.add(meshWater);

    // Sediment layer only if isopach available
    if (hasSediment) {
        const matSediment = new THREE.MeshStandardMaterial({ color: new THREE.Color(sedHex), roughness: 0.8, flatShading: true });
        const meshSediment = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, actualIsopach, 64), matSediment);
        meshSediment.position.y = -actualIsopach / 2; group.add(meshSediment);
    }

    // Bedrock cylinder starts right under sediment (or right under seafloor if no sediment)
    const meshBedrock = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, bedrockVisualHeight, 64), matBedrock);
    meshBedrock.position.y = -actualIsopach - (bedrockVisualHeight / 2); group.add(meshBedrock);

    const ring = new THREE.Mesh(
        new THREE.RingGeometry(radius, radius + 0.5, 64),
        new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide, transparent: true, opacity: 0.3 })
    );
    ring.rotateX(Math.PI / 2); scene.add(ring);

    // Info overlay
    const isopachText = hasSediment
        ? `<span class="font-mono text-orange-400">${isopach.toFixed(2)}m</span>`
        : `<span class="font-mono text-slate-500 italic">N/A</span>`;
    const sedRow = hasSediment
        ? `<div class="flex justify-between mt-2 pt-2 border-t border-slate-700"><span>Type:</span> <span class="font-bold" style="color:${sedHex}">${sedName}</span></div>`
        : `<div class="mt-2 pt-2 border-t border-slate-700 text-slate-500 italic text-[10px]">Sediment thickness unavailable for this site</div>`;

    const infoOverlay = document.createElement('div');
    infoOverlay.className = "absolute top-4 left-4 bg-slate-900/80 border border-slate-700 rounded p-3 z-10 text-slate-300 text-xs shadow-lg w-48";
    infoOverlay.innerHTML = `
        <div class="flex justify-between mb-1"><span>Water:</span> <span class="font-mono text-blue-300">${depth.toFixed(1)}m</span></div>
        <div class="flex justify-between mb-1"><span>Mud:</span> ${isopachText}</div>
        ${sedRow}
    `;
    container.appendChild(infoOverlay);

    function animate() {
        window.currentBoreholeAnimationId = requestAnimationFrame(animate);
        window.currentBoreholeControls.update();
        window.currentBoreholeRenderer.render(scene, camera);
    }
    animate();
}

// ── 12. Three.js 3D 地塊 (Block Model) ───────────────────────
window.close3D = function() {
    const modal = document.getElementById('modal-3d');
    if (!modal) return;
    modal.classList.add('hidden');
    if (window.currentAnimationId) { cancelAnimationFrame(window.currentAnimationId); window.currentAnimationId = null; }
    if (window.currentControls) { window.currentControls.dispose(); window.currentControls = null; }
    if (window.currentRenderer) { window.currentRenderer.dispose(); window.currentRenderer = null; }
    const container = document.getElementById('canvas-container-3d');
    if (container) container.innerHTML = '';
};

window.build3DScene = function(x0, y0, x1, y1) {
    if (typeof THREE === 'undefined') return alert("找不到 Three.js！");
    const modal = document.getElementById('modal-3d'), container = document.getElementById('canvas-container-3d'), loading = document.getElementById('loading-3d');
    if (!modal) return;

    modal.classList.remove('hidden'); loading.classList.remove('hidden');
    modal.onclick = (e) => { if (e.target === modal) window.close3D(); };
    if (window.currentRenderer) { container.innerHTML = ''; window.currentRenderer.dispose(); }

    Promise.all([
        fetch(`${API}/api/3d-scene?x0=${x0}&y0=${y0}&x1=${x1}&y1=${y1}`).then(r => r.json()),
        fetch(`${API}/api/stats?x0=${x0}&y0=${y0}&x1=${x1}&y1=${y1}`).then(r => r.json())
    ]).then(([sceneData, statsData]) => {
        if (sceneData.error) { window.close3D(); return alert(sceneData.error); }
        if (statsData.layers && statsData.layers.sediment_class) sceneData.sediment_val = statsData.layers.sediment_class.dominant || statsData.layers.sediment_class.mean;
        initThreeJS(container, sceneData);
        loading.classList.add('hidden');
    }).catch(err => { console.error(err); alert("載入 3D 失敗"); window.close3D(); });
};

function initThreeJS(container, data) {
    const { width, height, step_m, bathymetry, bedrock, sss_texture, sediment_val } = data;
    container.style.position = 'relative';
    const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0f172a); 
    const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 2000);
    camera.position.set(0, Math.max(width, height) * step_m * 0.8, Math.max(width, height) * step_m * 1.2);

    window.currentRenderer = new THREE.WebGLRenderer({ antialias: true });
    window.currentRenderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(window.currentRenderer.domElement);

    window.currentControls = new THREE.OrbitControls(camera, window.currentRenderer.domElement);
    window.currentControls.enableDamping = true; window.currentControls.autoRotate = false;

    scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8); dirLight.position.set(200, 300, 100); scene.add(dirLight);

    const Z_EXAGGERATION = 3.0; 
    let validBathymetry = bathymetry.filter(v => !isNaN(v));
    const minD = Math.min(...validBathymetry), maxD = Math.max(...validBathymetry), meanDepth = validBathymetry.reduce((a, b) => a + b, 0) / validBathymetry.length;

    const getTurboR = (depth) => {
        let t = 1 - Math.max(0, Math.min(1, (depth - minD) / (maxD - minD || 1)));
        const r = 0.1357 + 4.5974*t - 42.327*t*t + 130.588*Math.pow(t,3) - 150.566*Math.pow(t,4) + 58.137*Math.pow(t,5);
        const g = 0.0914 + 2.1941*t - 4.843*t*t + 14.185*Math.pow(t,3) - 32.171*Math.pow(t,4) + 28.533*Math.pow(t,5);
        const b = 0.1066 + 12.641*t - 60.582*t*t + 110.362*Math.pow(t,3) - 89.903*Math.pow(t,4) + 27.348*Math.pow(t,5);
        return new THREE.Color(Math.max(0,Math.min(1,r)), Math.max(0,Math.min(1,g)), Math.max(0,Math.min(1,b)));
    };

    let sedHex = '#8a8578', sedLabel = 'Unknown';
    if (sediment_val !== undefined && sediment_val !== null) {
        let id = (typeof sediment_val === 'number') ? Math.round(sediment_val) : SED_LABELS.findIndex(l => l === sediment_val);
        if (id >= 0 && id < SEDIMENT_COLORS.length) { sedHex = SEDIMENT_COLORS[id]; sedLabel = SED_LABELS[id]; }
    }
    const bedrockHex = '#1e293b'; 

    let sssTexture = null;
    if (sss_texture) {
        const texData = new Uint8Array(width * height * 4);
        for (let i = 0; i < sss_texture.length; i++) {
            const contrast = Math.pow(sss_texture[i] / 255, 1.2) * 255; 
            texData[i*4] = contrast; texData[i*4+1] = contrast*0.75; texData[i*4+2] = contrast*0.2; texData[i*4+3] = 255;
        }
        sssTexture = new THREE.DataTexture(texData, width, height, THREE.RGBAFormat); sssTexture.needsUpdate = true;
    }

    const bathyTexData = new Uint8Array(width * height * 4);
    for (let i = 0; i < bathymetry.length; i++) {
        const c = getTurboR(bathymetry[i]);
        bathyTexData[i*4] = c.r*255; bathyTexData[i*4+1] = c.g*255; bathyTexData[i*4+2] = c.b*255; bathyTexData[i*4+3] = 255;
    }
    const bathyTexture = new THREE.DataTexture(bathyTexData, width, height, THREE.RGBAFormat); bathyTexture.needsUpdate = true;

    const posS = new Float32Array(bathymetry.length * 3), posB = new Float32Array(bathymetry.length * 3), posBase = new Float32Array(bathymetry.length * 3);
    let minBedrockZ = Infinity;
    for (let i = 0; i < bathymetry.length; i++) {
        const x = (i % width) * step_m, y = Math.floor(i / width) * step_m;
        const zS = -(bathymetry[i] - meanDepth) * Z_EXAGGERATION;
        const zB = bedrock ? -(bedrock[i] - meanDepth) * Z_EXAGGERATION : zS - 2;
        if (zB < minBedrockZ) minBedrockZ = zB;
        posS[i*3] = x; posS[i*3+1] = zS; posS[i*3+2] = y; posB[i*3] = x; posB[i*3+1] = zB; posB[i*3+2] = y;
    }
    const baseZ = minBedrockZ - 10;
    for (let i = 0; i < bathymetry.length; i++) { posBase[i*3] = posS[i*3]; posBase[i*3+1] = baseZ; posBase[i*3+2] = posS[i*3+2]; }

    const matSurface = new THREE.MeshStandardMaterial({ map: bathyTexture, flatShading: true, side: THREE.DoubleSide });
    const matSediment = new THREE.MeshStandardMaterial({ color: new THREE.Color(sedHex), flatShading: true, side: THREE.DoubleSide });
    const matBedrock = new THREE.MeshStandardMaterial({ color: new THREE.Color(bedrockHex), flatShading: true, side: THREE.DoubleSide });

    const createPlane = (posArray) => {
        const geom = new THREE.PlaneGeometry(width * step_m, height * step_m, width - 1, height - 1);
        geom.rotateX(-Math.PI / 2); geom.translate((width * step_m)/2, 0, (height * step_m)/2);
        geom.attributes.position.array.set(posArray); geom.computeVertexNormals(); return geom;
    };

    const group = new THREE.Group(); group.position.set(-(width * step_m)/2, 0, -(height * step_m)/2); scene.add(group);
    group.add(new THREE.Mesh(createPlane(posS), matSurface)); group.add(new THREE.Mesh(createPlane(posB), matBedrock));

    const createWall = (tArr, bArr, count, indices, mat) => {
        const wallGeom = new THREE.PlaneGeometry(1, 1, count - 1, 1), wallPos = wallGeom.attributes.position.array;
        for (let i = 0; i < count; i++) {
            const idx = indices[i];
            wallPos[i*3] = tArr[idx*3]; wallPos[i*3+1] = tArr[idx*3+1]; wallPos[i*3+2] = tArr[idx*3+2];
            wallPos[(i+count)*3] = bArr[idx*3]; wallPos[(i+count)*3+1] = bArr[idx*3+1]; wallPos[(i+count)*3+2] = bArr[idx*3+2];
        }
        wallGeom.computeVertexNormals(); return new THREE.Mesh(wallGeom, mat);
    };

    const idxSouth = Array.from({length: width}, (_, i) => i), idxNorth = Array.from({length: width}, (_, i) => (height-1)*width + i);
    const idxWest = Array.from({length: height}, (_, i) => i*width), idxEast = Array.from({length: height}, (_, i) => i*width + width - 1);

    group.add(createWall(posS, posB, width, idxSouth, matSediment)); group.add(createWall(posS, posB, width, idxNorth, matSediment));
    group.add(createWall(posS, posB, height, idxWest, matSediment)); group.add(createWall(posS, posB, height, idxEast, matSediment));
    group.add(createWall(posB, posBase, width, idxSouth, matBedrock)); group.add(createWall(posB, posBase, width, idxNorth, matBedrock));
    group.add(createWall(posB, posBase, height, idxWest, matBedrock)); group.add(createWall(posB, posBase, height, idxEast, matBedrock));

    scene.add(new THREE.BoxHelper(group, 0x475569));

    const infoOverlay = document.createElement('div');
    infoOverlay.className = "absolute top-16 left-4 bg-slate-900/80 border border-slate-600 rounded-lg p-4 z-10 shadow-2xl text-slate-200 text-xs w-64 h-[220px] flex flex-col justify-between";
    infoOverlay.innerHTML = `
        <div class="font-bold text-blue-400 border-b border-slate-600 pb-2 mb-2">📊 Section Metrics</div>
        <div class="flex-1 flex flex-col justify-around">
            <div class="flex justify-between"><span>Area:</span> <span class="font-mono">${(width*step_m).toFixed(0)}x${(height*step_m).toFixed(0)}m</span></div>
            <div class="flex justify-between"><span>Depth:</span> <span class="font-mono">${minD.toFixed(1)}~${maxD.toFixed(1)}m</span></div>
            <div class="flex justify-between"><span>Sediment:</span> <span class="font-bold" style="color:${sedHex}">${sedLabel}</span></div>
        </div>
    `;
    container.appendChild(infoOverlay);

    const uiOverlay = document.createElement('div');
    uiOverlay.className = "absolute top-16 right-4 bg-slate-900/90 border border-slate-600 rounded-lg p-3 z-10 text-slate-200 text-sm";
    uiOverlay.innerHTML = `<label class="flex items-center gap-2 cursor-pointer hover:text-white"><input type="checkbox" id="ctrl-sss" class="accent-blue-500"> SSS Texture</label>`;
    container.appendChild(uiOverlay);

    document.getElementById('ctrl-sss').addEventListener('change', (e) => { matSurface.map = e.target.checked ? (sssTexture || bathyTexture) : bathyTexture; matSurface.needsUpdate = true; });

    function animate() { window.currentAnimationId = requestAnimationFrame(animate); window.currentControls.update(); window.currentRenderer.render(scene, camera); }
    animate();
    window.addEventListener('resize', () => {
        const modal = document.getElementById('modal-3d');
        if (modal && !modal.classList.contains('hidden')) {
            camera.aspect = container.clientWidth / container.clientHeight; camera.updateProjectionMatrix();
            if (window.currentRenderer) window.currentRenderer.setSize(container.clientWidth, container.clientHeight);
        }
    });
}
