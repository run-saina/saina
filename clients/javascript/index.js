export class SainaHelmError extends Error {
  constructor(message, status) { super(message); this.name = 'SainaHelmError'; this.status = status; }
}

export { SainaHelm as Saina, SainaHelmError as SainaError };

export class SainaHelm {
  constructor({baseUrl, apiKey, timeout = 60000, fetch: fetchImpl = globalThis.fetch}) {
    const url = new URL(baseUrl);
    if (!(url.protocol === 'https:' || (url.protocol === 'http:' && ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname))) || url.username || url.password || url.search || url.hash)
      throw new TypeError('Use HTTPS, or HTTP on localhost; no credentials/query in URL');
    if (typeof apiKey !== 'string' || !apiKey || /[\r\n]/.test(apiKey)) throw new TypeError('API key required');
    if (!Number.isFinite(timeout) || timeout <= 0) throw new TypeError('Positive timeout required');
    this.baseUrl = baseUrl.replace(/\/+$/, ''); this.apiKey = apiKey; this.timeout = timeout; this.fetch = fetchImpl;
  }
  async ask(request) {
    if (!request || !request.questions || !Object.keys(request.questions).length)
      throw new TypeError('At least one typed question is required');
    let response;
    try {
      response = await this.fetch(`${this.baseUrl}/v1/ask`, {
        method: 'POST', redirect: 'error', signal: AbortSignal.timeout(this.timeout),
        headers: {'Content-Type': 'application/json', Authorization: `Bearer ${this.apiKey}`},
        body: JSON.stringify(request)
      });
    } catch { throw new SainaHelmError('Saina Helm connection failed or timed out'); }
    if (!response.ok) throw new SainaHelmError(`Saina Helm HTTP ${response.status}`, response.status);
    let result;
    try { result = await response.json(); } catch { throw new SainaHelmError('Invalid JSON response'); }
    validateAnswers(request, result);
    return result;
  }
}

function validateAnswers(request, result) {
  const probability = v => typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1;
  const fail = () => { throw new SainaHelmError('Invalid typed answer response'); };
  if (!result || typeof result.model !== 'string' || !result.answers ||
      Object.keys(result.answers).length !== Object.keys(request.questions).length) fail();
  for (const [key, q] of Object.entries(request.questions)) {
    const a = Object.hasOwn(result.answers, key) && result.answers[key];
    if (!a || a.type !== q.type) fail();
    const labels = q.type === 'yes_no' ? ['yes', 'no'] : q.type === 'rating' ? q.levels.map((_, i) => String(i)) : Object.keys(q.options);
    const vector = q.type === 'yes_no' ? {yes: a.yes, no: a.no} : q.type === 'multi_choice' ? a.memberships : a.probabilities;
    if (!vector || Object.keys(vector).length !== labels.length || !labels.every(k => Object.hasOwn(vector, k) && probability(vector[k]))) fail();
    if (q.type !== 'multi_choice' && (Math.abs(labels.reduce((s, k) => s + vector[k], 0) - 1) > 1e-5 || !probability(a.confidence))) fail();
    if (q.type === 'single_choice' && !(labels.includes(a.selection) || request.mode === 'decision' && a.selection === null)) fail();
    if (q.type === 'rating' && (typeof a.expected_level !== 'number' || !Number.isFinite(a.expected_level) || a.expected_level < 0 || a.expected_level > labels.length - 1 || !Array.isArray(a.levels) || a.levels.length !== labels.length)) fail();
    if (request.mode === 'decision') {
      if (!['accepted', 'below_threshold', 'below_margin', 'tie'].includes(a.reason)) fail();
      if (q.type === 'yes_no' && !(a.selected === null || typeof a.selected === 'boolean')) fail();
      if (q.type === 'rating' && !(a.level === null || Number.isInteger(a.level) && a.level >= 0 && a.level < labels.length)) fail();
      if (q.type === 'multi_choice' && (!Array.isArray(a.selections) || new Set(a.selections).size !== a.selections.length || !a.selections.every(k => labels.includes(k)))) fail();
    }
  }
}
