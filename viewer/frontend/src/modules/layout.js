import L from 'leaflet';
import { state } from '../state.js';

const mapWrapper = document.getElementById('map-wrapper');
const sidebar = document.getElementById('sidebar');
const bottomPanel = document.getElementById('bottom-panel');
const rightPanel = document.getElementById('right-panel');
const sidebarChevron = document.getElementById('sidebar-chevron');

let isSidebarOpen = true;
let currentRightWidth = 0;
let currentBottomHeight = 0;


export function applyLayout() {
    const leftOffset = isSidebarOpen ? 320 : 0;
    if (sidebar) sidebar.style.transform = isSidebarOpen ? 'translateX(0)' : 'translateX(-100%)';

    if (sidebarChevron) {
        sidebarChevron.innerHTML = isSidebarOpen
            ? '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M15 19l-7-7 7-7"></path>'
            : '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M9 5l7 7-7 7"></path>';
    }

    if (rightPanel) {
        rightPanel.style.transform = currentRightWidth > 0 ? 'translateX(0)' : 'translateX(100%)';
        rightPanel.style.width = currentRightWidth > 0 ? `${currentRightWidth}px` : '0px';
    }

    if (bottomPanel) {
        bottomPanel.style.transform = currentBottomHeight > 0 ? 'translateY(0)' : 'translateY(100%)';
        bottomPanel.style.height = currentBottomHeight > 0 ? `${currentBottomHeight}px` : '0px';
        bottomPanel.style.left = `${leftOffset}px`;
        bottomPanel.style.right = currentRightWidth > 0 ? `${currentRightWidth}px` : '0px';
    }

    if (mapWrapper) {
        mapWrapper.style.left = `${leftOffset}px`;
        mapWrapper.style.right = currentRightWidth > 0 ? `${currentRightWidth}px` : '0px';
        mapWrapper.style.bottom = currentBottomHeight > 0 ? `${currentBottomHeight}px` : '0px';
    }
}


export function openPanels(mode) {
    document.getElementById('bp-sbp-section')?.classList.add('hidden');
    document.getElementById('bp-echarts-cursor')?.classList.add('hidden');
    document.getElementById('rp-echarts-cursor')?.classList.add('hidden');
    document.getElementById('bp-slider')?.classList.add('hidden');
    document.getElementById('rp-slider')?.classList.add('hidden');

    if (mode === 'drawn-line') {
        currentRightWidth = 0;
        currentBottomHeight = window.innerHeight * 0.35;
    } else if (mode === 'sss') {
        currentBottomHeight = 0;
        currentRightWidth = 450;
        document.getElementById('rp-slider')?.classList.remove('hidden');
    } else if (mode === 'sbp') {
        currentRightWidth = 0;
        currentBottomHeight = window.innerHeight * 0.55;
        document.getElementById('bp-sbp-section')?.classList.remove('hidden');
        document.getElementById('bp-slider')?.classList.remove('hidden');
    }
    applyLayout();
}


export function closePanels() {
    currentRightWidth = 0;
    currentBottomHeight = 0;
    applyLayout();
    if (state.mapTrackMarker) {
        state.map.removeLayer(state.mapTrackMarker);
        state.mapTrackMarker = null;
    }
    if (state.selectedTrackline && state.selectedParentLayer) {
        state.selectedParentLayer.resetStyle(state.selectedTrackline);
        state.selectedTrackline = null;
    }
}


export function resizeCanvases() {
    if (state.map) state.map.invalidateSize({ animate: false });
    const bpChart = document.getElementById('bp-echarts-container');
    const rpChart = document.getElementById('rp-echarts-container');
    if (bpChart && bpChart._chart) bpChart._chart.resize();
    if (rpChart && rpChart._chart) rpChart._chart.resize();
}


function bindResizers() {
    let resizeTarget = null;

    document.getElementById('rp-resizer')?.addEventListener('mousedown', () => {
        resizeTarget = 'right';
        document.body.style.cursor = 'col-resize';
        document.body.classList.add('select-none');
        if (rightPanel) rightPanel.style.transition = 'none';
        if (mapWrapper) mapWrapper.style.transition = 'none';
    });

    document.getElementById('bp-resizer')?.addEventListener('mousedown', () => {
        resizeTarget = 'bottom';
        document.body.style.cursor = 'row-resize';
        document.body.classList.add('select-none');
        if (bottomPanel) bottomPanel.style.transition = 'none';
        if (mapWrapper) mapWrapper.style.transition = 'none';
    });

    document.getElementById('rp-internal-resizer')?.addEventListener('mousedown', () => {
        resizeTarget = 'rp-internal';
        document.body.style.cursor = 'row-resize';
        document.body.classList.add('no-select');
    });

    document.getElementById('bp-internal-resizer')?.addEventListener('mousedown', () => {
        resizeTarget = 'bp-internal';
        document.body.style.cursor = 'row-resize';
        document.body.classList.add('no-select');
    });

    window.addEventListener('mousemove', (e) => {
        if (!resizeTarget) return;
        if (resizeTarget === 'right') {
            const newW = window.innerWidth - e.clientX;
            if (newW > 300 && newW < window.innerWidth - 350) {
                currentRightWidth = newW;
                applyLayout();
            }
        } else if (resizeTarget === 'bottom') {
            const newH = window.innerHeight - e.clientY;
            if (newH > 150 && newH < window.innerHeight - 100) {
                currentBottomHeight = newH;
                applyLayout();
            }
        } else if (resizeTarget === 'rp-internal') {
            const topSec = document.getElementById('rp-top-section');
            const newH = e.clientY - topSec.getBoundingClientRect().top;
            if (newH > 100 && newH < window.innerHeight - 200) {
                topSec.style.height = `${newH}px`;
                topSec.querySelector('#rp-echarts-container')?._chart?.resize();
            }
        } else if (resizeTarget === 'bp-internal') {
            const topSec = document.getElementById('bp-top-section');
            const newH = e.clientY - topSec.getBoundingClientRect().top;
            if (newH > 100 && newH < currentBottomHeight - 100) {
                topSec.style.height = `${newH}px`;
                topSec.querySelector('#bp-echarts-container')?._chart?.resize();
            }
        }
    });

    window.addEventListener('mouseup', () => {
        if (resizeTarget) {
            resizeTarget = null;
            document.body.style.cursor = '';
            document.body.classList.remove('select-none');

            if (rightPanel) rightPanel.style.transition = 'transform 0.3s ease-in-out';
            if (bottomPanel) bottomPanel.style.transition = 'transform 0.3s ease-in-out';
            if (mapWrapper) mapWrapper.style.transition = 'all 0.3s ease-in-out';

            if (state.map) state.map.invalidateSize({ animate: false });
            document.getElementById('bp-slider')?.dispatchEvent(new Event('input'));
            document.getElementById('rp-slider')?.dispatchEvent(new Event('input'));
        }
        resizeCanvases();
    });
}


export function bindLayoutUI() {
    document.getElementById('btn-toggle-sidebar')?.addEventListener('click', () => {
        isSidebarOpen = !isSidebarOpen;
        applyLayout();
        setTimeout(resizeCanvases, 300);
    });

    bindResizers();

    // Close buttons for panels
    document.getElementById('btn-close-bp')?.addEventListener('click', closePanels);
    document.getElementById('btn-close-rp')?.addEventListener('click', closePanels);
}