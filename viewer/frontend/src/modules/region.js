import { API } from '../constants.js';
import { build3DScene } from './block3d.js';


export function doRegionSelect(bounds) {
    const sw = bounds.getSouthWest();
    const ne = bounds.getNorthEast();

    Promise.all([
        fetch(`${API}/api/query?lat=${sw.lat}&lon=${sw.lng}`).then(r => r.json()),
        fetch(`${API}/api/query?lat=${ne.lat}&lon=${ne.lng}`).then(r => r.json()),
    ]).then(([sw_d, ne_d]) => {
        if (sw_d.error || ne_d.error) return alert("座標轉換失敗！");
        build3DScene(sw_d.x_3826, sw_d.y_3826, ne_d.x_3826, ne_d.y_3826);
    }).catch(err => alert(`API 錯誤: ${err.message}`));
}