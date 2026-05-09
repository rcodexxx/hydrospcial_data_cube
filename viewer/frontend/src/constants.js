// Sediment classification (Hamilton 7-class adapted for freshwater reservoir)
export const SEDIMENT_COLORS = [
    "#A0522D",  // 0: Coarse sand
    "#CD853F",  // 1: Fine sand / Silty sand
    "#DEB887",  // 2: Silt / Sandy silt
    "#BDB76B",  // 3: Sand-silt-clay
    "#8FBC8F",  // 4: Compacted mud
    "#6495ED",  // 5: Clayey silt / Silty clay
    "#2E5C8A",  // 6: Fluid mud
];

export const SED_LABELS = [
    'Coarse sand',
    'Fine sand / Silty sand',
    'Silt / Sandy silt',
    'Sand-silt-clay',
    'Compacted mud',
    'Clayey silt / Silty clay',
    'Fluid mud',
];

// Map defaults
export const INITIAL_CENTER = [22.137, 120.785];
export const INITIAL_ZOOM = 15;

// API base path (empty = same-origin)
export const API = '';