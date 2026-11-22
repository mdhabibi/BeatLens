const plot = document.querySelector('#plot');
const legend = document.querySelector('#legend');
const interpretation = document.querySelector('#interpretation');
const plotGuide = document.querySelector('#plot-guide');
const buttons = [...document.querySelectorAll('[data-mode]')];
const colors = { r: '#c4ccd1', predicted: '#ff8495', reference: '#89e2cd', lime: '#8bccfa', encoder: '#c4a8ff', raw: '#ffc073' };
const domain = [-150, 500];
const number = (value) => value.toFixed(2);

try {
  const response = await fetch('./beat.json');
  if (!response.ok) throw new Error('Example data request failed');
  const original = await response.json();
  let beat = original;
  const readout = document.querySelector('#readout');
  const cursor = document.querySelector('#cursor');
  const reading = document.querySelector('#cursor-reading');
  function interpolate(samples, ms) {
    const index = ms * beat.fs / 1000 + beat.anchor;
    const start = Math.floor(index);
    return samples[start] + (index - start) * (samples[start + 1] - samples[start]);
  }
  function updateReadout() {
    beat = readout.value === 'interaction' ? original : { ...original, ...original.variants[readout.value] };
    document.querySelector('#prediction').textContent = number(beat.prediction_ms) + ' ms';
    document.querySelector('#annotation').textContent = number(beat.reference_ms) + ' ms';
    document.querySelector('#error').textContent = number(Math.abs(beat.prediction_ms - beat.reference_ms)) + ' ms';
    document.querySelector('#quality').textContent =
      `Reference: ${beat.baseline}. LIME weighted R²: ${beat.waveform_lime.weighted_r2.toFixed(3)} across ${beat.waveform_lime.perturbations} perturbations. IG completeness residual: ${beat.patch_ig.completeness_residual_ms.toExponential(2)} ms.`;
  }
  const time = (sample) => (sample - beat.anchor) * 1000 / beat.fs;
  const tick = (value) => ({ time: value, label: String(value) });
  const pointSeries = (samples, interval = 1) => samples.map((value, i) => ({
    time: time(i * interval + (Math.min(interval, beat.waveform[0].length - i * interval) - 1) / 2),
    value,
  }));
  function render(mode) {
    const width = Math.max(270, Math.round(plot.getBoundingClientRect().width));
    const x = (ms) => 38 + (ms - domain[0]) / (domain[1] - domain[0]) * (width - 62);
    const ticks = Array.from({ length: 14 }, (_, i) => tick(-150 + i * 50))
      .filter(({ time: ms }) => width >= 580 || ms % 100 === 0);
    const visible = (points) => points.filter(({ time: ms }) => ms >= domain[0] && ms <= domain[1]);
    const grid = (top, bottom) => ticks.map(({ time: ms }) =>
      `<path d="M${x(ms)} ${top}V${bottom}" stroke="#553440"/>`).join('');
    const path = (points, y, color) => `<polyline points="${visible(points).map(p => `${x(p.time)},${y(p.value)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2"/>`;
    const charts = [];
    let items;
    plotGuide.textContent = {
      waveform: 'Each trace shows normalized ECG voltage. The red × is the model estimate, the teal dot is the manual T peak, and the diamond is the supplied R anchor. The dotted crosshair follows the time slider.',
      lime: 'Each bar shows the estimated effect of restoring a short segment to its original shape. Bars above zero move the estimate later. Bars below zero move it earlier.',
      ig: 'Each bar shows the signed contribution of a time patch through encoder features. Where shown, the orange line traces contributions through raw ECG inputs. Both are measured in milliseconds.',
    }[mode];
    if (mode === 'waveform') {
      items = [[colors.r, '◆', 'Supplied R anchor'], [colors.predicted, '×', 'Model estimate'], [colors.reference, '●', 'Manual annotation']];
      beat.waveform.forEach((samples, i) => {
        const points = pointSeries(samples);
        const amplitude = Math.max(...visible(points).map(({ value }) => Math.abs(value)));
        const top = 18 + i * 128;
        const y = (value) => top + 55 - value / amplitude * 46;
        const guides = [[0, ...items[0]], [beat.prediction_ms, ...items[1]], [beat.reference_ms, ...items[2]]];
        charts.push(grid(top, top + 105), path(points, y, '#f5d9df'),
          `<text x="0" y="${top + 14}" fill="#fff2f3">${['D3', 'V4'][i]}</text>`);
        for (const [ms, color, shape] of guides) {
          const sample = interpolate(samples, ms);
          charts.push(`<path d="M${x(ms)} ${top}V${top + 105}" stroke="${color}" stroke-dasharray="3 4"/><text x="${x(ms)}" y="${y(sample) + 4}" fill="${color}" text-anchor="middle">${shape}</text>`);
        }
      });
      const offset = beat.prediction_ms - beat.reference_ms;
      interpretation.textContent = `D3 and V4 show the same beat. The known R peak starts the clock. The model’s estimate is ${number(Math.abs(offset))} ms ${offset < 0 ? 'earlier' : 'later'} than the manual annotation.`;
      reading.textContent = beat.waveform.map((samples, i) => `${['D3', 'V4'][i]}: ${number(interpolate(samples, Number(cursor.value)))}`).join(' · ') + ' (normalized amplitude)';
    } else {
      const isLime = mode === 'lime';
      const series = isLime ? beat.waveform_lime : beat.patch_ig;
      const interval = isLime ? series.segment_samples : series.patch_samples;
      const points = pointSeries(series.signed_ms, interval);
      const raw = series.raw_signed_ms ? pointSeries(series.raw_signed_ms, interval) : [];
      const extent = Math.max(...visible([...points, ...raw]).map(({ value }) => Math.abs(value)));
      const y = (value) => 132 - value / extent * 93;
      const barWidth = (x(time(interval)) - x(time(0))) * .82;
      const barColor = isLime ? colors.lime : colors.encoder;
      items = isLime ? [[barColor, '■', 'Estimated segment effect']] : [[barColor, '■', 'Encoder features'], ...(raw.length ? [[colors.raw, '━', 'Raw ECG']] : [])];
      charts.push(grid(30, 238), `<path d="M${x(-150)} 132H${x(500)}" stroke="#ac8593"/>`);
      for (const point of visible(points)) {
        charts.push(`<rect x="${x(point.time) - barWidth / 2}" y="${Math.min(132, y(point.value))}" width="${barWidth}" height="${Math.abs(y(point.value) - 132)}" fill="${barColor}"><title>${number(point.time)} ms · ${number(point.value)} ms effect</title></rect>`);
      }
      if (raw.length) charts.push(path(raw, y, colors.raw));
      charts.push(...[extent, 0, -extent].map(value => `<text x="0" y="${y(value) + 4}" fill="#dfbfc7">${number(value)}</text>`));
      const segmentIndex = Math.floor((Number(cursor.value) * beat.fs / 1000 + beat.anchor) / interval);
      const contributions = [`${isLime ? 'LIME' : 'Encoder IG'}: ${number(series.signed_ms[segmentIndex])} ms`];
      if (raw.length) contributions.push(`Raw ECG IG: ${number(series.raw_signed_ms[segmentIndex])} ms`);
      reading.textContent = contributions.join(' · ');
      if (isLime) {
        const peak = series.signed_ms.reduce((best, value, i, values) => Math.abs(value) > Math.abs(values[best]) ? i : best, 0);
        const first = peak * interval;
        const last = Math.min(first + interval, beat.waveform[0].length) - 1;
        const effect = series.signed_ms[peak];
        interpretation.textContent = `LIME blurs short pieces of the ECG and estimates what each original segment contributes. In the local approximation, restoring the ${time(first).toFixed(0)}–${time(last).toFixed(0)} ms segment shifts the estimate ${number(Math.abs(effect))} ms ${effect < 0 ? 'earlier' : 'later'}.`;
      } else {
        interpretation.textContent = `Integrated Gradients attributes the shift from ${number(beat.baseline_prediction_ms)} ms on the smoothed beat to ${number(beat.prediction_ms)} ms on the original. Positive contributions push the estimate later and negative contributions push it earlier.`;
      }
    }
    const selectedX = x(Number(cursor.value));
    charts.push(`<path d="M${selectedX} 10V250" stroke="#fff2f3" stroke-dasharray="2 5" opacity=".65"/>`);
    document.querySelector('#cursor-value').textContent = `${cursor.value} ms`;
    legend.innerHTML = items.map(([color, sign, label]) => `<span><b style="color:${color}">${sign}</b>${label}</span>`).join('');
    const labels = ticks.map(({ time: ms, label }) => `<text x="${x(ms)}" y="264" text-anchor="middle" fill="#dfbfc7">${label}</text>`);
    plot.innerHTML = `<svg role="img" aria-label="${mode === 'waveform' ? 'ECG waveforms and timing markers' : 'Signed attribution effects in milliseconds'}" viewBox="0 0 ${width} 278" font-size="12">${[...charts, ...labels].join('')}</svg>`;
  }
  let mode = 'waveform';
  buttons.forEach(button => button.addEventListener('click', () => {
    mode = button.dataset.mode;
    buttons.forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    render(mode);
  }));
  readout.addEventListener('change', () => { updateReadout(); render(mode); });
  cursor.addEventListener('input', () => render(mode));
  updateReadout();
  new ResizeObserver(() => render(mode)).observe(plot);
  render(mode);
} catch {
  plot.textContent = 'The saved ECG example could not be loaded. Reload to try again.';
  buttons.forEach(button => { button.disabled = true; });
}
