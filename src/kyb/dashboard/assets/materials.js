/* Liquid Midnight @7730d0f: open-kyb hue adaptation. Apache-2.0; see NOTICE.md. */
(() => {
    const field = /** @type {HTMLCanvasElement} */ (document.getElementById('field')), cx = field.getContext('2d');
    const rm = matchMedia('(prefers-reduced-motion: reduce)');
    let playing = !rm.matches, time = 0, last = 0, lastPaint = -1, dirty = true, orbs = [];
    const glyphs = [...' .·:-=+*%#@'];
    /** @param {HTMLCanvasElement} canvas @param {boolean} witness */
    function makeOrb(canvas, witness) {
        try {
            const gl = canvas.getContext('webgl', { alpha: true, premultipliedAlpha: true });
            if (!gl) {
                canvas.replaceWith(document.createTextNode(witness ? '◇' : '◉'));
                return null;
            }
            const shader = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
                throw new Error('orb shader unavailable'); return s; };
            const p = gl.createProgram();
            gl.attachShader(p, shader(gl.VERTEX_SHADER, 'attribute vec2 p;void main(){gl_Position=vec4(p,0.,1.);}'));
            gl.attachShader(p, shader(gl.FRAGMENT_SHADER, document.getElementById('orb-source').textContent));
            gl.linkProgram(p);
            if (!gl.getProgramParameter(p, gl.LINK_STATUS))
                return null;
            gl.useProgram(p);
            const buffer = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
            const loc = gl.getAttribLocation(p, 'p');
            gl.enableVertexAttribArray(loc);
            gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
            const u = n => gl.getUniformLocation(p, n);
            gl.uniform1f(u('uViolet'), .08);
            gl.uniform1f(u('uSeed'), .3);
            gl.uniform1f(u('u_speed'), .4);
            return { dispose: () => { gl.deleteBuffer(buffer); gl.deleteProgram(p); gl.getExtension('WEBGL_lose_context')?.loseContext(); }, paint: t => {
                    const day = document.documentElement.dataset.theme === 'light';
                    const colors = witness ? (day ? [[.89, .64, .32], [.49, .3, .1]] : [[1, .88, .63], [.71, .47, .2]]) : (day ? [[1, .47, .74], [.72, 0, .48]] : [[1, .68, .85], [.71, 0, .44]]);
                    gl.uniform3fv(u('uA'), colors[0]);
                    gl.uniform3fv(u('uB'), colors[1]);
                    const size = Math.round(60 * Math.min(2, devicePixelRatio || 1));
                    if (canvas.width !== size) {
                        canvas.width = size;
                        canvas.height = size;
                    }
                    gl.viewport(0, 0, size, size);
                    gl.uniform2f(u('u_resolution'), size, size);
                    gl.uniform1f(u('u_time'), t);
                    gl.drawArrays(gl.TRIANGLES, 0, 6);
                } };
        }
        catch {
            return null;
        }
    }
    function paint(t) {
        const day = document.documentElement.dataset.theme === 'light', w = innerWidth, h = innerHeight;
        const d = Math.min(1.5, devicePixelRatio || 1);
        if (field.width !== Math.round(w * d) || field.height !== Math.round(h * d)) {
            field.width = Math.round(w * d);
            field.height = Math.round(h * d);
        }
        cx.setTransform(d, 0, 0, d, 0, 0);
        cx.clearRect(0, 0, w, h);
        cx.font = '10px monospace';
        cx.fillStyle = day ? '#5a0a46' : '#ffc2ea';
        cx.globalAlpha = day ? .08 : .12;
        for (let y = 0; y < h; y += 17)
            for (let x = 0; x < w; x += 17) {
                const f = Math.sin(x * .011 + Math.sin(y * .008 + t * .19) * 1.7 + t * .13) * .5 + .5, g = Math.sin(y * .015 - t * .11 + Math.sin(x * .005 + t * .07) * 2.2) * .5 + .5, v = Math.min(.72, Math.pow(f * g, 2.2) * 1.25);
                if (v > .02)
                    cx.fillText(glyphs[Math.min(10, 1 + Math.floor(v * 10))], x, y);
            }
        document.documentElement.style.setProperty('--angle', ((t * 40) % 360) + 'deg');
        orbs.forEach(o => o.paint(t));
        dirty = false;
    }
    function tick(now) { if (!last)
        last = now; if (playing && !document.hidden)
        time += Math.min(.05, (now - last) / 1000); last = now; if (dirty || playing && !document.hidden && now - lastPaint > 1000 / 15) {
        paint(time);
        lastPaint = now;
    } requestAnimationFrame(tick); }
    const motion = document.getElementById('motion');
    const motionLabel = () => { motion.dataset.playing = String(playing); motion.setAttribute('aria-label', playing ? 'Pause ambient motion' : 'Play ambient motion'); motion.setAttribute('aria-pressed', String(playing)); };
    motion.onclick = () => { playing = !playing; motionLabel(); dirty = true; };
    rm.addEventListener('change', () => { playing = !rm.matches; dirty = true; motionLabel(); });
    window.kybMaterials = { refresh() { orbs.forEach(o => o.dispose()); orbs = [...document.querySelectorAll('canvas[data-orb]')].map(c => { const canvas = /** @type {HTMLCanvasElement} */ (c); return makeOrb(canvas, canvas.dataset.orb === 'witness'); }).filter(Boolean); dirty = true; }, invalidate() { dirty = true; } };
    window.renderAt = t => { time = Math.max(0, Number(t) || 0); paint(time); };
    addEventListener('resize', () => { dirty = true; });
    motionLabel();
    requestAnimationFrame(tick);
})();
