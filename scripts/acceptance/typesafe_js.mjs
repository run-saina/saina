// Acceptance: the official TypeSafe JavaScript SDK, unmodified, against Saina's System One route.
//   node scripts/acceptance/typesafe_js.mjs BASE_URL API_KEY      (BASE_URL without /v1)
import {choice, noul, score, TypeSafeClient} from '@typesafe-ai/sdk';

const [baseURL, apiKey] = process.argv.slice(2);
const results = [];
const check = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${!ok && detail ? ` (${detail})` : ''}`); };
for (const [label, extra] of [['default compression', {}], ['compression off', {compression: false}]]) {
  const client = new TypeSafeClient({apiKey, baseURL, defaultModel: 'saina-helm', maxRetries: 0, ...extra});
  console.log(`${label}: client compression = ${client.compression}`);
  for (const [size, state] of [['small body', 'I was charged twice.'], ['body over 1 KiB', 'x '.repeat(2000)]]) {
    try {
      const {answers} = await client.systemOne({state, questions: {
        billing: noul('Is this about billing?'), tone: choice('Tone?', {calm: null, angry: null}), urgency: score('Urgency?', ['low', 'high'])}});
      check(`${label}: ${size}`, answers.billing.noul >= 0 && ['calm', 'angry'].includes(answers.tone.choice));
    } catch (e) { check(`${label}: ${size}`, false, `${e.constructor.name}: ${String(e.message).slice(0, 120)}`); }
  }
}
const failed = results.filter(ok => !ok).length;
console.log(`${results.length - failed}/${results.length} checks passed`);
process.exit(failed ? 1 : 0);
