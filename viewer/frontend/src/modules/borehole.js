import { state } from '../state.js';
import { API, SEDIMENT_COLORS } from '../constants.js';


function closeBorehole() {
    const modal = document.getElementById('modal-borehole');
    if (modal) modal.classList.add('hidden');

    if (window.currentBoreholeAnimationId) {
        cancelAnimationFrame(window.currentBoreholeAnimationId);
        window.currentBoreholeAnimationId = null;
    }
    if (window.currentBoreholeControls) {
        window.currentBoreholeControls.dispose();
        window.currentBoreholeControls = null;
    }
    if (window.currentBoreholeRenderer) {
        window.currentBoreholeRenderer.dispose();
        window.currentBoreholeRenderer = null;
    }
    const container = document.getElementById('canvas-container-borehole');
    if (container) container.innerHTML = '';
}


function buildBoreholeScene(lat, lon, title = "虛擬岩心探測") {
    if (typeof THREE === 'undefined') return alert("找不到 Three.js！");

    const modal = document.getElementById('modal-borehole');
    const container = document.getElementById('canvas-container-borehole');
    const loading = document.getElementById('loading-borehole');
    if (!modal) return;

    document.getElementById('borehole-title').innerHTML = `🕳️ ${title}`;
    modal.classList.remove('hidden');
    loading.classList.remove('hidden');
    modal.onclick = (e) => { if (e.target === modal) closeBorehole(); };

    if (window.currentBoreholeRenderer) {
        container.innerHTML = '';
        window.currentBoreholeRenderer.dispose();
    }

    fetch(`${API}/api/query?lat=${lat}&lon=${lon}`).then(r => r.json()).then(data => {
        if (data.error) { closeBorehole(); return alert(data.error); }
        const depth = data.bathymetry?.value || 0;
        const isopach = state.HAS_ISOPACH ? (data.isopach?.value ?? null) : null;
        const sedClassId = data.sediment_class?.class_id ?? -1;
        const sedName = data.sediment_class?.value || 'Unknown';

        initBorehole3D(container, { depth, isopach, sedClassId, sedName });
        loading.classList.add('hidden');
    }).catch(err => {
        console.error(err);
        alert("載入岩心失敗");
        closeBorehole();
    });
}


function initBorehole3D(container, data) {
    const { depth, isopach, sedClassId, sedName } = data;
    const Z_EXAGGERATION = 5.0;
    const hasSediment = (isopach !== null && isopach > 0);

    container.style.position = 'relative';
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0f1a);

    const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(20, 10, 25);

    window.currentBoreholeRenderer = new THREE.WebGLRenderer({ antialias: true });
    window.currentBoreholeRenderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(window.currentBoreholeRenderer.domElement);

    window.currentBoreholeControls = new THREE.OrbitControls(camera, window.currentBoreholeRenderer.domElement);
    window.currentBoreholeControls.enableDamping = true;
    window.currentBoreholeControls.target.set(0, -5, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(10, 20, 10);
    scene.add(dirLight);

    const radius = 4;
    const actualIsopach = hasSediment ? isopach * Z_EXAGGERATION : 0;
    const bedrockVisualHeight = 15, waterVisualHeight = 8;
    const sedHex = (sedClassId >= 0 && sedClassId < SEDIMENT_COLORS.length) ? SEDIMENT_COLORS[sedClassId] : '#8a8578';

    const matWater = new THREE.MeshPhysicalMaterial({ color: 0x3b82f6, transparent: true, opacity: 0.15, roughness: 0.1 });
    const matBedrock = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.9, flatShading: true });

    const group = new THREE.Group(); scene.add(group);

    // Water column
    const meshWater = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, waterVisualHeight, 64), matWater);
    meshWater.position.y = waterVisualHeight / 2;
    group.add(meshWater);

    // Sediment layer (only if isopach available)
    if (hasSediment) {
        const matSediment = new THREE.MeshStandardMaterial({ color: new THREE.Color(sedHex), roughness: 0.8, flatShading: true });
        const meshSediment = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, actualIsopach, 64), matSediment);
        meshSediment.position.y = -actualIsopach / 2;
        group.add(meshSediment);
    }

    // Bedrock cylinder starts right under sediment (or right under seafloor if no sediment)
    const meshBedrock = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, bedrockVisualHeight, 64), matBedrock);
    meshBedrock.position.y = -actualIsopach - (bedrockVisualHeight / 2);
    group.add(meshBedrock);

    // Surface ring at borehole position
    const ring = new THREE.Mesh(
        new THREE.RingGeometry(radius, radius + 0.5, 64),
        new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide, transparent: true, opacity: 0.3 })
    );
    ring.rotateX(Math.PI / 2);
    scene.add(ring);

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


export function exposeBorehole() {
    // Expose for popup HTML inline onclick (Step 13 will refactor that)
    // and toolbar.js close-on-tool-switch logic.
    window.buildBoreholeScene = buildBoreholeScene;
    window.closeBorehole = closeBorehole;
}