// Tailwind CSS — must come first so utility classes are available
import './index-tw.css'

// Library imports
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

import * as echarts from 'echarts'

import OpenSeadragon from 'openseadragon'

// Expose to window for app.js / sss_modal.js (which were written assuming globals).
// Future: refactor app.js to import these directly, then remove window aliases.
window.L = L
window.THREE = THREE
// THREE namespace import is frozen, so wrap it in a plain object that we CAN extend
window.THREE = { ...THREE, OrbitControls }
window.echarts = echarts
window.OpenSeadragon = OpenSeadragon

// Project styles
import '../style.css'

// Run app code
import '../app.js'
import '../sss_modal.js'