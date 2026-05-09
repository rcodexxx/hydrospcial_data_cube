// Pure helper functions — no state, no DOM access.

/**
 * Interpolate a polyline (in [lon, lat] format) into N evenly-spaced points
 * by distance along the line.
 */
export function interpolatePolyline(map, coords, numPoints) {
    if (coords.length < 2) {
        return Array(numPoints).fill(coords[0] || [0, 0]);
    }

    const cumDist = [0];
    for (let i = 1; i < coords.length; i++) {
        cumDist.push(
            cumDist[i - 1] + map.distance(
                L.latLng(coords[i - 1][1], coords[i - 1][0]),
                L.latLng(coords[i][1], coords[i][0])
            )
        );
    }
    const totalDist = cumDist[cumDist.length - 1];
    if (totalDist === 0) return Array(numPoints).fill(coords[0]);

    const step = totalDist / (numPoints - 1);
    const result = [];
    for (let i = 0; i < numPoints; i++) {
        if (i === 0) { result.push(coords[0]); continue; }
        if (i === numPoints - 1) { result.push(coords[coords.length - 1]); continue; }

        const targetDist = i * step;
        let segIdx = cumDist.findIndex(d => d >= targetDist) - 1;
        if (segIdx < 0) segIdx = 0;

        const ratio = (targetDist - cumDist[segIdx]) / (cumDist[segIdx + 1] - cumDist[segIdx]);
        const lon = coords[segIdx][0] + ratio * (coords[segIdx + 1][0] - coords[segIdx][0]);
        const lat = coords[segIdx][1] + ratio * (coords[segIdx + 1][1] - coords[segIdx][1]);
        result.push([lon, lat]);
    }
    return result;
}