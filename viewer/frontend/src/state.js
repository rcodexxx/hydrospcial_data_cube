// Shared mutable state.
// All modules import the same `state` object — mutations are visible everywhere.
export const state = {
    // Map and Leaflet
    map: null,
    currentOverlay: null,
    tileLayers: {},
    currentBaseLayerId: 'bathymetry',
    contourLayer: null,
    HAS_ISOPACH: false,

    // Click and selection
    clickMarker: null,
    selectRect: null,
    selectStart: null,
    drawnLine: null,
    linePreview: null,
    lineStart: null,

    // Tracklines
    sssLayer: null,
    sbpLayer: null,
    selectedTrackline: null,
    selectedParentLayer: null,

    // Tool state
    currentTool: 'pan',

    // Waterfall and profile
    waterfallIndex: null,
    currentWfPings: 0,
    mapTrackMarker: null,
    currentTrackCoords: [],
};