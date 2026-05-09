import { API, SEDIMENT_COLORS, SED_LABELS } from '../constants.js';


function close3D() {
    const modal = document.getElementById('modal-3d');
    if (!modal) return;
    modal.classList.add('hidden');

    if (window.currentAnimationId) {
        cancelAnimationFrame(window.currentAnimationId);
        window.currentAnimationId = null;
    }
    if (window.currentControls) {
        window.currentControls.dispose();
        window.currentControls = null;
    }
    if (window.currentRenderer) {
        window.currentRenderer.dispose();
        window.currentRenderer = null;
    }
    const container = document.getElementById('canvas-container-3d');
    if (container) container.innerHTML = '';
}


function build3DScene(x0, y0, x1, y1) {
    if (typeof THREE === 'undefined') return alert("找不到 Three.js！");

    const modal = document.getElementById('modal-3d');
    const container = document.getElementById('canvas-container-3d');
    const loading = document.getElementById('loading-3d');
    if (!modal) return;

    modal.classList.remove('hidden');
    loading.classList.remove('hidden');
    modal.onclick = (e) => { if (e.target === modal) close3D(); };

    if (window.currentRenderer) {
        container.innerHTML = '';
        window.currentRenderer.dispose();
    }

    Promise.all([
        fetch(`${API}/api/3d-scene?x0=${x0}&y0=${y0}&x1=${x1}&y1=${y1}`).then(r => r.json()),
        fetch(`${API}/api/stats?x0=${x0}&y0=${y0}&x1=${x1}&y1=${y1}`).then(r => r.json()),
    ]).then(([sceneData, statsData]) => {
        if (sceneData.error) { close3D(); return alert(sceneData.error); }
        if (statsData.layers && statsData.layers.sediment_class) {
            sceneData.sediment_val = statsData.layers.sediment_class.dominant
                || statsData.layers.sediment_class.mean;
        }
        initThreeJS(container, sceneData);
        loading.classList.add('hidden');
    }).catch(err => {
        console.error(err);
        alert("載入 3D 失敗");
        close3D();
    });
}


function initThreeJS(container, data) {
    const { width, height, step_m, bathymetry, bedrock, sss_texture, sediment_val } = data;
    container.style.position = 'relative';

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f172a);

    const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 2000);
    camera.position.set(0, Math.max(width, height) * step_m * 0.8, Math.max(width, height) * step_m * 1.2);

    window.currentRenderer = new THREE.WebGLRenderer({ antialias: true });
    window.currentRenderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(window.currentRenderer.domElement);

    window.currentControls = new THREE.OrbitControls(camera, window.currentRenderer.domElement);
    window.currentControls.enableDamping = true;
    window.currentControls.autoRotate = false;

    scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(200, 300, 100);
    scene.add(dirLight);

    const Z_EXAGGERATION = 3.0;
    const validBathymetry = bathymetry.filter(v => !isNaN(v));
    const minD = Math.min(...validBathymetry);
    const maxD = Math.max(...validBathymetry);
    const meanDepth = validBathymetry.reduce((a, b) => a + b, 0) / validBathymetry.length;

    const getTurboR = (depth) => {
        const t = 1 - Math.max(0, Math.min(1, (depth - minD) / (maxD - minD || 1)));
        const r = 0.1357 + 4.5974*t - 42.327*t*t + 130.588*Math.pow(t,3) - 150.566*Math.pow(t,4) + 58.137*Math.pow(t,5);
        const g = 0.0914 + 2.1941*t - 4.843*t*t + 14.185*Math.pow(t,3) - 32.171*Math.pow(t,4) + 28.533*Math.pow(t,5);
        const b = 0.1066 + 12.641*t - 60.582*t*t + 110.362*Math.pow(t,3) - 89.903*Math.pow(t,4) + 27.348*Math.pow(t,5);
        return new THREE.Color(Math.max(0, Math.min(1, r)), Math.max(0, Math.min(1, g)), Math.max(0, Math.min(1, b)));
    };

    let sedHex = '#8a8578', sedLabel = 'Unknown';
    if (sediment_val !== undefined && sediment_val !== null) {
        const id = (typeof sediment_val === 'number')
            ? Math.round(sediment_val)
            : SED_LABELS.findIndex(l => l === sediment_val);
        if (id >= 0 && id < SEDIMENT_COLORS.length) {
            sedHex = SEDIMENT_COLORS[id];
            sedLabel = SED_LABELS[id];
        }
    }
    const bedrockHex = '#1e293b';

    let sssTexture = null;
    if (sss_texture) {
        const texData = new Uint8Array(width * height * 4);
        for (let i = 0; i < sss_texture.length; i++) {
            const contrast = Math.pow(sss_texture[i] / 255, 1.2) * 255;
            texData[i*4] = contrast;
            texData[i*4+1] = contrast * 0.75;
            texData[i*4+2] = contrast * 0.2;
            texData[i*4+3] = 255;
        }
        sssTexture = new THREE.DataTexture(texData, width, height, THREE.RGBAFormat);
        sssTexture.needsUpdate = true;
    }

    const bathyTexData = new Uint8Array(width * height * 4);
    for (let i = 0; i < bathymetry.length; i++) {
        const c = getTurboR(bathymetry[i]);
        bathyTexData[i*4] = c.r * 255;
        bathyTexData[i*4+1] = c.g * 255;
        bathyTexData[i*4+2] = c.b * 255;
        bathyTexData[i*4+3] = 255;
    }
    const bathyTexture = new THREE.DataTexture(bathyTexData, width, height, THREE.RGBAFormat);
    bathyTexture.needsUpdate = true;

    const posS = new Float32Array(bathymetry.length * 3);
    const posB = new Float32Array(bathymetry.length * 3);
    const posBase = new Float32Array(bathymetry.length * 3);
    let minBedrockZ = Infinity;
    for (let i = 0; i < bathymetry.length; i++) {
        const x = (i % width) * step_m;
        const y = Math.floor(i / width) * step_m;
        const zS = -(bathymetry[i] - meanDepth) * Z_EXAGGERATION;
        const zB = bedrock ? -(bedrock[i] - meanDepth) * Z_EXAGGERATION : zS - 2;
        if (zB < minBedrockZ) minBedrockZ = zB;
        posS[i*3] = x; posS[i*3+1] = zS; posS[i*3+2] = y;
        posB[i*3] = x; posB[i*3+1] = zB; posB[i*3+2] = y;
    }
    const baseZ = minBedrockZ - 10;
    for (let i = 0; i < bathymetry.length; i++) {
        posBase[i*3] = posS[i*3];
        posBase[i*3+1] = baseZ;
        posBase[i*3+2] = posS[i*3+2];
    }

    const matSurface = new THREE.MeshStandardMaterial({ map: bathyTexture, flatShading: true, side: THREE.DoubleSide });
    const matSediment = new THREE.MeshStandardMaterial({ color: new THREE.Color(sedHex), flatShading: true, side: THREE.DoubleSide });
    const matBedrock = new THREE.MeshStandardMaterial({ color: new THREE.Color(bedrockHex), flatShading: true, side: THREE.DoubleSide });

    const createPlane = (posArray) => {
        const geom = new THREE.PlaneGeometry(width * step_m, height * step_m, width - 1, height - 1);
        geom.rotateX(-Math.PI / 2);
        geom.translate((width * step_m) / 2, 0, (height * step_m) / 2);
        geom.attributes.position.array.set(posArray);
        geom.computeVertexNormals();
        return geom;
    };

    const group = new THREE.Group();
    group.position.set(-(width * step_m) / 2, 0, -(height * step_m) / 2);
    scene.add(group);
    group.add(new THREE.Mesh(createPlane(posS), matSurface));
    group.add(new THREE.Mesh(createPlane(posB), matBedrock));

    const createWall = (tArr, bArr, count, indices, mat) => {
        const wallGeom = new THREE.PlaneGeometry(1, 1, count - 1, 1);
        const wallPos = wallGeom.attributes.position.array;
        for (let i = 0; i < count; i++) {
            const idx = indices[i];
            wallPos[i*3] = tArr[idx*3]; wallPos[i*3+1] = tArr[idx*3+1]; wallPos[i*3+2] = tArr[idx*3+2];
            wallPos[(i+count)*3] = bArr[idx*3]; wallPos[(i+count)*3+1] = bArr[idx*3+1]; wallPos[(i+count)*3+2] = bArr[idx*3+2];
        }
        wallGeom.computeVertexNormals();
        return new THREE.Mesh(wallGeom, mat);
    };

    const idxSouth = Array.from({ length: width }, (_, i) => i);
    const idxNorth = Array.from({ length: width }, (_, i) => (height - 1) * width + i);
    const idxWest = Array.from({ length: height }, (_, i) => i * width);
    const idxEast = Array.from({ length: height }, (_, i) => i * width + width - 1);

    group.add(createWall(posS, posB, width, idxSouth, matSediment));
    group.add(createWall(posS, posB, width, idxNorth, matSediment));
    group.add(createWall(posS, posB, height, idxWest, matSediment));
    group.add(createWall(posS, posB, height, idxEast, matSediment));
    group.add(createWall(posB, posBase, width, idxSouth, matBedrock));
    group.add(createWall(posB, posBase, width, idxNorth, matBedrock));
    group.add(createWall(posB, posBase, height, idxWest, matBedrock));
    group.add(createWall(posB, posBase, height, idxEast, matBedrock));

    scene.add(new THREE.BoxHelper(group, 0x475569));

    const infoOverlay = document.createElement('div');
    infoOverlay.className = "absolute top-16 left-4 bg-slate-900/80 border border-slate-600 rounded-lg p-4 z-10 shadow-2xl text-slate-200 text-xs w-64 h-[220px] flex flex-col justify-between";
    infoOverlay.innerHTML = `
        <div class="font-bold text-blue-400 border-b border-slate-600 pb-2 mb-2">📊 Section Metrics</div>
        <div class="flex-1 flex flex-col justify-around">
            <div class="flex justify-between"><span>Area:</span> <span class="font-mono">${(width*step_m).toFixed(0)}x${(height*step_m).toFixed(0)}m</span></div>
            <div class="flex justify-between"><span>Depth:</span> <span class="font-mono">${minD.toFixed(1)}~${maxD.toFixed(1)}m</span></div>
            <div class="flex justify-between"><span>Sediment:</span> <span class="font-bold" style="color:${sedHex}">${sedLabel}</span></div>
        </div>
    `;
    container.appendChild(infoOverlay);

    const uiOverlay = document.createElement('div');
    uiOverlay.className = "absolute top-16 right-4 bg-slate-900/90 border border-slate-600 rounded-lg p-3 z-10 text-slate-200 text-sm";
    uiOverlay.innerHTML = `<label class="flex items-center gap-2 cursor-pointer hover:text-white"><input type="checkbox" id="ctrl-sss" class="accent-blue-500"> SSS Texture</label>`;
    container.appendChild(uiOverlay);

    document.getElementById('ctrl-sss').addEventListener('change', (e) => {
        matSurface.map = e.target.checked ? (sssTexture || bathyTexture) : bathyTexture;
        matSurface.needsUpdate = true;
    });

    function animate() {
        window.currentAnimationId = requestAnimationFrame(animate);
        window.currentControls.update();
        window.currentRenderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', () => {
        const modal = document.getElementById('modal-3d');
        if (modal && !modal.classList.contains('hidden')) {
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            if (window.currentRenderer) {
                window.currentRenderer.setSize(container.clientWidth, container.clientHeight);
            }
        }
    });
}


export function exposeBlock3D() {
    // Expose for region.js (via window.build3DScene), HTML inline onclick (window.close3D),
    // and toolbar.js close-on-tool-switch logic.
    window.build3DScene = build3DScene;
    window.close3D = close3D;
}