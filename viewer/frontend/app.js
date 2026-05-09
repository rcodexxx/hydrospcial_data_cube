import { API, SEDIMENT_COLORS, SED_LABELS, INITIAL_CENTER, INITIAL_ZOOM } from './src/constants.js';
import { state } from './src/state.js';
import { interpolatePolyline } from './src/utils.js';
import { initMap, loadTileLayers, bindMapUI } from './src/modules/map.js';
import { applyLayout, openPanels, closePanels, resizeCanvases, bindLayoutUI } from './src/modules/layout.js';
import { doPointQuery } from './src/modules/popup.js';
import { doRegionSelect } from './src/modules/region.js';
import { loadTracklines, loadMagTargets, bindTracklinesUI } from './src/modules/tracklines.js';
import { bindToolbar } from './src/modules/toolbar.js';
import { loadWaterfallIndex, showWaterfallSidebar, renderProfileChart, bindSliders } from './src/modules/waterfall.js';
import { exposeBorehole } from './src/modules/borehole.js';
import { exposeBlock3D } from './src/modules/block3d.js';

const map = initMap();
loadTileLayers(map);
bindMapUI(map);
bindLayoutUI();
loadTracklines();
loadMagTargets();
bindTracklinesUI();
bindToolbar();
loadWaterfallIndex();
bindSliders();
exposeBorehole();
exposeBlock3D();

