import { state } from '../state.js';
import { API, SEDIMENT_COLORS } from '../constants.js';


export function doPointQuery(lat, lon) {
    const map = state.map;

    if (state.clickMarker) {
        state.clickMarker.setLatLng([lat, lon]);
    } else {
        state.clickMarker = L.circleMarker([lat, lon], {
            radius: 6, color: '#F57D15', fillColor: '#F57D15',
            fillOpacity: 0.8, weight: 2,
        }).addTo(map);
    }

    const popup = L.popup({
        maxWidth: 300, autoClose: true, closeOnClick: true,
        autoPanPadding: [20, 20],
    })
        .setLatLng([lat, lon])
        .setContent(`<div class="w-56 p-1 text-slate-400 animate-pulse text-xs font-bold">⏳ 擷取地層數據中...</div>`)
        .openOn(map);

    fetch(`${API}/api/query?lat=${lat}&lon=${lon}`)
        .then(r => r.json())
        .then(data => {
            let html = `<div class="w-56 p-1"><div class="font-mono text-[11px] text-blue-600 font-bold border-b border-slate-100 pb-1 mb-2">📍 ${lat.toFixed(5)}, ${lon.toFixed(5)}</div>`;

            if (data.error) {
                popup.setContent(html + `<div class="text-red-500 text-xs font-bold py-2 text-center">⚠️ ${data.error}</div></div>`);
                popup.update();
                return;
            }

            const coreKeys = ['bathymetry', 'isopach', 'sediment_class'];
            let hasCoreData = false, primaryHtml = '', secondaryHtml = '', hasSecondaryData = false;

            coreKeys.forEach(key => {
                const info = data[key];
                if (info && info.value !== null) {
                    hasCoreData = true;
                    if (key === 'sediment_class') {
                        const classId = info.class_id !== undefined ? info.class_id : -1;
                        const color = (SEDIMENT_COLORS[classId]) ? SEDIMENT_COLORS[classId] : '#888';
                        primaryHtml += `<div class="text-[11px] flex justify-between py-1.5 items-center border-b border-slate-50"><span class="font-bold text-slate-700">${info.name}:</span><span class="text-[10px] text-white px-1.5 py-0.5 rounded shadow-sm" style="background:${color}">${info.value}</span></div>`;
                    } else {
                        const val = (typeof info.value === 'number' && info.value % 1 !== 0) ? info.value.toFixed(2) : info.value;
                        const unitHtml = info.units ? `<span class="text-[10px] text-slate-500 ml-1">${info.units}</span>` : '';
                        primaryHtml += `<div class="text-[11px] flex justify-between py-1.5 border-b border-slate-50"><span class="font-bold text-slate-700">${info.name}:</span><span class="text-blue-600 font-bold">${val}${unitHtml}</span></div>`;
                    }
                }
            });

            for (const [key, info] of Object.entries(data)) {
                if (['lat', 'lon', 'x_3826', 'y_3826'].includes(key) || coreKeys.includes(key) || !info || !info.name || info.value === null) continue;
                hasSecondaryData = true;
                const val = (typeof info.value === 'number' && info.value % 1 !== 0) ? info.value.toFixed(2) : info.value;
                const unitHtml = info.units ? `<span class="text-[10px] text-slate-500 ml-1">${info.units}</span>` : '';
                secondaryHtml += `<div class="text-[10px] flex justify-between py-1 border-b border-slate-100 last:border-0"><span class="text-slate-500">${info.name}:</span><span class="text-slate-800 font-mono">${val}${unitHtml}</span></div>`;
            }

            if (!hasCoreData && !hasSecondaryData) {
                html += `<div class="text-slate-400 text-[11px] py-4 text-center font-bold">此座標無地層資料</div>`;
            } else {
                html += primaryHtml;
                if (hasSecondaryData) {
                    html += `
                        <details class="mt-1 group" ontoggle="setTimeout(() => { if(window.map && window.map._popup) window.map._popup.update(); }, 50)">
                            <summary class="text-[10px] text-slate-400 cursor-pointer py-1.5 hover:text-blue-500 select-none outline-none font-bold flex items-center gap-1 transition-colors">
                                <span class="group-open:rotate-90 transition-transform text-[8px]">▶</span> 顯示進階探測參數
                            </summary>
                            <div class="pt-1 pb-2 bg-slate-50/50 rounded px-1.5 mt-1 border border-slate-100">${secondaryHtml}</div>
                        </details>`;
                }

                html += `
                    <button onclick="if(typeof window.buildBoreholeScene === 'function') window.buildBoreholeScene(${lat}, ${lon})" class="mt-2 w-full py-1.5 bg-slate-800 hover:bg-orange-500 text-white text-xs font-bold rounded shadow-sm transition-colors flex items-center justify-center gap-1">
                        🕳️ 鑽取虛擬岩心
                    </button>
                `;
            }
            popup.setContent(html + `</div>`);
            popup.update();
        })
        .catch(err => {
            popup.setContent(`<div class="w-56 p-3 text-red-500 text-xs font-bold text-center">⚠️ 介面渲染失敗</div>`);
            popup.update();
        });
}