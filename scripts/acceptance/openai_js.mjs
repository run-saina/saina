// Acceptance: the official OpenAI JavaScript SDK, unmodified, against a Saina /v1/decisions endpoint.
//   node scripts/acceptance/openai_js.mjs BASE_URL API_KEY     (BASE_URL includes /v1)
// Resolve `openai` from a directory where openai@7.43.0 is installed (NODE_PATH or a local node_modules).
import OpenAI from 'openai';

const [baseURL, apiKey, textModel = 'saina-helm'] = process.argv.slice(2);
const client = new OpenAI({baseURL, apiKey, maxRetries: 0});
const results = [];
const check = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${!ok && detail ? ` (${detail})` : ''}`); };
console.log(`openai-node ${OpenAI.VERSION ?? ''} -> ${baseURL}`);

const decision = await client.decisions.create({
  model: textModel, input: 'I was charged twice for my order.',
  questions: [{type: 'choice', name: 'department', instructions: 'Which department should handle this complaint?',
    choices: [{value: 'billing', description: 'Payments, invoices, and refunds.'}, {value: 'technical', description: 'Problems using the product.'},
              {value: 'shipping', description: 'Delivery and tracking.'}, {value: 'other', description: 'Requests outside these categories.'}]},
              {type: 'predicate', name: 'refund', instructions: 'Does the customer want a refund?'},
              {type: 'score', name: 'severity', instructions: 'How severe?', levels: [{label: 'Low'}, {label: 'High'}]}],
});
const [choice, refund, severity] = decision.answers;
check('choice answer', choice.type === 'choice' && ['billing', 'technical', 'shipping', 'other'].includes(choice.choice));
check('choice probabilities array', Array.isArray(choice.probabilities) && Math.abs(choice.probabilities.reduce((s, p) => s + p.probability, 0) - 1) < 1e-6);
check('predicate probability', refund.type === 'predicate' && refund.probability >= 0 && refund.probability <= 1);
check('score labels', severity.type === 'score' && severity.probabilities.map(p => p.label).join() === 'Low,High');
check('usage', decision.usage.output_tokens === 0 && decision.usage.total_tokens === decision.usage.input_tokens);

const policy = await client.decisions.create({model: textModel, input: 'unclear',
  questions: [{type: 'predicate', name: 'billing', instructions: 'Billing?'}], mode: 'decision', threshold: 0.999});
check('policy refusal', policy.answers[0].type === 'refusal' && policy.answers[0].reason === 'below_threshold');

for (const [name, body, cls] of [
  ['unknown model is NotFoundError', {model: 'gpt-6-luna'}, OpenAI.NotFoundError],
  ['temperature is BadRequestError', {temperature: 0}, OpenAI.BadRequestError]]) {
  try {
    await client.decisions.create({model: textModel, input: 'x', questions: [{type: 'predicate', instructions: 'A?'}], ...body});
    check(name, false, 'no error');
  } catch (e) { check(name, e instanceof cls, e.constructor.name); }
}
const failed = results.filter(ok => !ok).length;
console.log(`${results.length - failed}/${results.length} checks passed`);
process.exit(failed ? 1 : 0);
