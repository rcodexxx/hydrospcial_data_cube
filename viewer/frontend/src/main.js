// Tailwind CSS — must come first
import './index-tw.css'

// Library CSS
import 'leaflet/dist/leaflet.css'

// Project styles
import '../style.css'

// Wire up the application
import { initMap, loadTileLayers, bindMapUI } from './modules/map.js'
import { bindLayoutUI } from './modules/layout.js'
import { loadTracklines, loadMagTargets, bindTracklinesUI } from './modules/tracklines.js'
import { bindToolbar } from './modules/toolbar.js'
import { loadWaterfallIndex, bindSliders } from './modules/waterfall.js'
import { bindBorehole } from './modules/borehole.js'
import { bindBlock3D } from './modules/block3d.js'

import './sss_modal.js'

const map = initMap()
loadTileLayers(map)
bindMapUI(map)
bindLayoutUI()
loadTracklines()
loadMagTargets()
bindTracklinesUI()
bindToolbar()
loadWaterfallIndex()
bindSliders()
bindBorehole()
bindBlock3D()